"""Tests for the PNA layer: hand-computed forward/backward, numeric
gradient checks (including the std aggregator and the degree scalers),
the multi-aggregator distinction, and training dynamics."""

import math
import random

import pytest

from gnn_fraud import (
    Graph,
    GraphGenerator,
    GraphPNA,
    PNALayer,
    evaluate_all,
    train_pna,
    train_val_split,
)


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def _two_node_graph():
    """2 nodes, 0 -- 1, 1-dim features 1.0 and 2.0."""
    g = Graph(2, 1)
    g.add_edge(0, 1)
    g.set_features(0, [1.0])
    g.set_features(1, [2.0])
    return g


def _selector_layer(in_dim=1, out_dim=1, position=0, delta=1.0):
    """A PNA layer whose W picks exactly one coordinate of z (the
    node's own embedding at 0, then the twelve towers in
    scaler-major, aggregator order) and whose bias is zero."""
    layer = PNALayer(in_dim, out_dim, random.Random(0), delta=delta)
    layer.W = [[0.0] * layer.z_dim for _ in range(out_dim)]
    layer.W[0][position] = 1.0
    layer.b = [0.0] * out_dim
    return layer


def _tower_position(in_dim, scaler_idx, agg_idx):
    """Position of one tower inside z: self features first, then
    scaler-major blocks of four aggregators (mean, max, min, std)."""
    return in_dim + scaler_idx * 4 * in_dim + agg_idx * in_dim


# ---------------------------------------------------------------------------
# forward: hand-computable cases
# ---------------------------------------------------------------------------


def test_pna_layer_forward_self_tower():
    # W selects only the node's own feature: out is the identity.
    g = _two_node_graph()
    layer = _selector_layer(position=0)
    out, _ = layer.forward(g, g.features)
    approx(out[0][0], 1.0)
    approx(out[1][0], 2.0)


def test_pna_layer_forward_mean_tower():
    # W selects the identity-scaler mean tower: node 0 reads its only
    # neighbor's feature (2.0), node 1 reads node 0's (1.0).
    g = _two_node_graph()
    layer = _selector_layer(position=_tower_position(1, 0, 0))
    out, _ = layer.forward(g, g.features)
    approx(out[0][0], 2.0)
    approx(out[1][0], 1.0)


def test_pna_layer_forward_max_min_towers():
    # Node 0 has neighbors with features 1.0 and 3.0: max reads 3.0,
    # min reads 1.0, mean reads 2.0 -- three towers, three answers.
    g = Graph(3, 1)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.set_features(0, [0.0])
    g.set_features(1, [1.0])
    g.set_features(2, [3.0])
    for agg_idx, expected in ((0, 2.0), (1, 3.0), (2, 1.0)):
        layer = _selector_layer(position=_tower_position(1, 0, agg_idx))
        out, _ = layer.forward(g, g.features)
        approx(out[0][0], expected)


def test_pna_layer_forward_std_tower():
    # Neighbors 1.0 and 3.0: mean 2.0, population std 1.0.
    g = Graph(3, 1)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.set_features(0, [0.0])
    g.set_features(1, [1.0])
    g.set_features(2, [3.0])
    layer = _selector_layer(position=_tower_position(1, 0, 3))
    out, _ = layer.forward(g, g.features)
    approx(out[0][0], 1.0)


def test_pna_layer_forward_amplification_scaler():
    # Degree 2, delta 1.0: amplification is log(3); the amplified
    # mean tower reads log(3) * 2.0.
    g = Graph(3, 1)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.set_features(0, [0.0])
    g.set_features(1, [1.0])
    g.set_features(2, [3.0])
    layer = _selector_layer(position=_tower_position(1, 1, 0), delta=1.0)
    out, _ = layer.forward(g, g.features)
    approx(out[0][0], math.log(3) * 2.0)


def test_pna_layer_forward_attenuation_scaler():
    # Degree 2, delta 1.0: attenuation is 1 / log(3).
    g = Graph(3, 1)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.set_features(0, [0.0])
    g.set_features(1, [1.0])
    g.set_features(2, [3.0])
    layer = _selector_layer(position=_tower_position(1, 2, 0), delta=1.0)
    out, _ = layer.forward(g, g.features)
    approx(out[0][0], 2.0 / math.log(3))


def test_pna_layer_isolated_node_reads_only_self():
    # An isolated node has zero aggregators and identity scalers, so
    # every tower is 0 and only the self tower survives.
    g = Graph(3, 1)
    g.add_edge(0, 1)
    g.set_features(0, [1.0])
    g.set_features(1, [2.0])
    g.set_features(2, [3.0])
    layer = _selector_layer(position=0)
    out, _ = layer.forward(g, g.features)
    approx(out[2][0], 3.0)
    # Selecting the mean tower instead must give exactly 0.
    layer = _selector_layer(position=_tower_position(1, 0, 0))
    out, _ = layer.forward(g, g.features)
    approx(out[2][0], 0.0)


def test_pna_layer_std_is_zero_for_constant_neighborhood():
    # Identical neighbors: std is 0, so the std tower reads 0 even
    # though mean/max/min all read the shared value.
    g = Graph(3, 1)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.set_features(0, [0.0])
    g.set_features(1, [2.0])
    g.set_features(2, [2.0])
    layer = _selector_layer(position=_tower_position(1, 0, 3))
    out, _ = layer.forward(g, g.features)
    approx(out[0][0], 0.0)


def test_pna_layer_rejects_nonpositive_delta():
    with pytest.raises(ValueError):
        PNALayer(2, 2, random.Random(0), delta=0.0)


# ---------------------------------------------------------------------------
# backward: numeric gradient checks
# ---------------------------------------------------------------------------


def test_pna_layer_gradient_check():
    # 3-node path 0 -- 1 -- 2, 2-dim features, 2-dim output, delta 1.5
    # so the scalers are non-trivial (log(2)/1.5, 1.5/log(2)).
    g = Graph(3, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    rng = random.Random(3)
    layer = PNALayer(2, 2, rng, delta=1.5)
    # Nonzero biases: with zero biases, dead-activation nodes sit
    # exactly on the ReLU kink (pre == 0), where the numeric gradient
    # is undefined and the analytic 0 disagrees with the finite
    # difference.
    layer.b = [0.1, -0.2]
    d_out = [[0.3, -0.7], [1.1, 0.2], [-0.4, 0.9]]

    def loss_fn():
        out, _ = layer.forward(g, g.features)
        return sum(d * o for d_row, o_row in zip(d_out, out)
                   for d, o in zip(d_row, o_row))

    out, cache = layer.forward(g, g.features)
    d_H = layer.backward(d_out, cache)

    eps = 1e-6
    for i in range(2):
        for j in range(layer.z_dim):
            orig = layer.W[i][j]
            layer.W[i][j] = orig + eps
            plus = loss_fn()
            layer.W[i][j] = orig - eps
            minus = loss_fn()
            layer.W[i][j] = orig
            approx(layer.gW[i][j], (plus - minus) / (2 * eps), tol=1e-5)
    for i in range(2):
        orig = layer.b[i]
        layer.b[i] = orig + eps
        plus = loss_fn()
        layer.b[i] = orig - eps
        minus = loss_fn()
        layer.b[i] = orig
        approx(layer.gb[i], (plus - minus) / (2 * eps), tol=1e-5)
    # Gradients w.r.t. the input embeddings (returned, not accumulated).
    # This checks every aggregator's routing at once: mean spreads,
    # max/min go to the winner, std goes to each neighbor by deviation.
    for u in range(3):
        for j in range(2):
            orig = g.features[u][j]
            g.features[u][j] = orig + eps
            plus = loss_fn()
            g.features[u][j] = orig - eps
            minus = loss_fn()
            g.features[u][j] = orig
            approx(d_H[u][j], (plus - minus) / (2 * eps), tol=1e-5)


def test_pna_model_gradient_check():
    # Full two-layer model + output head, numeric-checked end to end.
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    g.set_features(3, [0.1, 0.4])
    labels = {0: 1, 1: 0, 2: 1, 3: 0}
    model = GraphPNA(2, hidden_dim=3, lr=0.0, seed=5, delta=1.5)

    def loss_fn():
        probs, _ = model.forward(g, [0, 1, 2, 3])
        total = 0.0
        for v in [0, 1, 2, 3]:
            p = min(max(probs[v], 1e-7), 1.0 - 1e-7)
            y = labels[v]
            total += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
        return total / 4

    # lr=0.0, so train_batch accumulates gradients without moving weights.
    model.train_batch(g, [0, 1, 2, 3], labels)
    eps = 1e-6
    for layer in (model.layer1, model.layer2):
        for i in range(len(layer.W)):
            for j in range(len(layer.W[0])):
                orig = layer.W[i][j]
                layer.W[i][j] = orig + eps
                plus = loss_fn()
                layer.W[i][j] = orig - eps
                minus = loss_fn()
                layer.W[i][j] = orig
                approx(layer.gW[i][j], (plus - minus) / (2 * eps), tol=1e-5)
        for i in range(len(layer.b)):
            orig = layer.b[i]
            layer.b[i] = orig + eps
            plus = loss_fn()
            layer.b[i] = orig - eps
            minus = loss_fn()
            layer.b[i] = orig
            approx(layer.gb[i], (plus - minus) / (2 * eps), tol=1e-5)
    for j in range(3):
        orig = model.w_out[j]
        model.w_out[j] = orig + eps
        plus = loss_fn()
        model.w_out[j] = orig - eps
        minus = loss_fn()
        model.w_out[j] = orig
        approx(model.gw_out[j], (plus - minus) / (2 * eps), tol=1e-5)
    orig = model.b_out
    model.b_out = orig + eps
    plus = loss_fn()
    model.b_out = orig - eps
    minus = loss_fn()
    model.b_out = orig
    approx(model.gb_out, (plus - minus) / (2 * eps), tol=1e-5)


def test_pna_checkpoint_round_trip():
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    g.set_features(3, [0.1, 0.4])
    model = GraphPNA(2, hidden_dim=3, seed=5, delta=1.5)
    before = model.predict(g, [0, 1, 2, 3])
    state = model.state_dict()
    model2 = GraphPNA(2, hidden_dim=3, seed=99, delta=2.5)
    model2.load_state_dict(state)
    after = model2.predict(g, [0, 1, 2, 3])
    for v in [0, 1, 2, 3]:
        approx(before[v], after[v])
    assert model2.delta == 1.5


# ---------------------------------------------------------------------------
# training dynamics
# ---------------------------------------------------------------------------


def test_pna_trains_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
    model, hist = train_pna(g, train_ids, val_ids, hidden_dim=16,
                            lr=0.1, batch_size=64, epochs=60,
                            patience=12, seed=7)
    assert min(hist["val_loss"]) < hist["val_loss"][0]
    y_true = [g.labels[v] for v in val_ids]
    scores = [model.predict(g, [v])[v] for v in val_ids]
    auc = evaluate_all(y_true, scores)["auc"]
    assert auc >= 0.90, f"PNA AUC {auc:.3f} below 0.90"

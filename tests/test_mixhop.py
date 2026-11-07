"""Tests for the MixHop layer: hand-computed forward/backward, numeric
gradient checks, hop properties, and training dynamics."""

import math
import random

from gnn_fraud import (
    Graph,
    GraphGenerator,
    GraphMixHop,
    MixHopLayer,
    evaluate_all,
    train_mixhop,
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


def _layer_with_weights(K):
    layer = MixHopLayer(1, 1, K=K, rng=random.Random(0))
    layer.W = [[[0.0]] for _ in range(K + 1)]
    layer.b = [[0.0] for _ in range(K + 1)]
    return layer


# ---------------------------------------------------------------------------
# forward: hand-computable cases
# ---------------------------------------------------------------------------


def test_mixhop_layer_forward_hand_computed():
    # Row-normalized weights are 1/2 each (degree + self-loop = 2):
    # P0 = [1, 2]; P1 = [1.5, 1.5]; P2 = [1.5, 1.5].
    # Each hop lands in its own output block, not summed.
    g = _two_node_graph()
    layer = _layer_with_weights(2)
    layer.W[0] = [[1.0]]
    out, _ = layer.forward(g, g.features)
    assert out[0] == [1.0, 0.0, 0.0]
    assert out[1] == [2.0, 0.0, 0.0]

    layer = _layer_with_weights(2)
    layer.W[1] = [[1.0]]
    out, _ = layer.forward(g, g.features)
    assert out[0] == [0.0, 1.5, 0.0]
    assert out[1] == [0.0, 1.5, 0.0]

    layer = _layer_with_weights(2)
    layer.W[2] = [[1.0]]
    out, _ = layer.forward(g, g.features)
    assert out[0] == [0.0, 0.0, 1.5]
    assert out[1] == [0.0, 0.0, 1.5]

    # All hops together with unit weights: concat, not sum.
    layer = _layer_with_weights(2)
    for k in range(3):
        layer.W[k] = [[1.0]]
    out, _ = layer.forward(g, g.features)
    assert out[0] == [1.0, 1.5, 1.5]
    assert out[1] == [2.0, 1.5, 1.5]


def test_mixhop_layer_k0_is_linear():
    # K=0: only A^0 = I, so the layer is relu(W0 @ x + b0).
    g = _two_node_graph()
    layer = _layer_with_weights(0)
    layer.W[0] = [[2.0]]
    layer.b = [[1.0]]
    out, _ = layer.forward(g, g.features)
    assert out[0] == [3.0]
    assert out[1] == [5.0]


def test_mixhop_layer_isolated_node():
    # Node 2 is isolated: its A row is just the self-loop with weight
    # 1, so every power returns its own feature, no div by 0.
    g = Graph(3, 1)
    g.add_edge(0, 1)
    g.set_features(0, [1.0])
    g.set_features(1, [2.0])
    g.set_features(2, [3.0])
    layer = _layer_with_weights(2)
    layer.W[2] = [[1.0]]
    out, _ = layer.forward(g, g.features)
    assert out[2] == [0.0, 0.0, 3.0]


def test_mixhop_row_norm_weights_sum_to_one():
    # Every row of the row-normalized adjacency sums to 1, including
    # on a hub: the average never blows up with degree.
    g = Graph(5, 1)
    for u in range(1, 5):
        g.add_edge(0, u)
    from gnn_fraud.mixhop import _row_norm_weights
    weights = _row_norm_weights(g)
    for row in weights:
        approx(sum(c for _, c in row), 1.0)
    approx(weights[0][0][1], 0.2)


# ---------------------------------------------------------------------------
# backward: numeric gradient checks
# ---------------------------------------------------------------------------


def test_mixhop_layer_gradient_check():
    g = Graph(3, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.set_features(0, [1.0, 0.5])
    g.set_features(1, [0.0, 2.0])
    g.set_features(2, [-1.0, 1.0])
    layer = MixHopLayer(2, 2, K=2, rng=random.Random(1))
    out, cache = layer.forward(g, g.features)
    d_out = [[1.0, -0.5, 0.25, 1.0, -1.0, 0.5],
             [0.5, 1.0, -0.25, 0.5, 1.0, -0.5],
             [-1.0, 0.5, 1.0, -0.25, 0.5, 1.0]]
    layer.zero_grad()
    d_H = layer.backward(d_out, cache)

    def loss_fn():
        o, _ = layer.forward(g, g.features)
        return sum(d * x for row_d, row_o in zip(d_out, o)
                   for d, x in zip(row_d, row_o))

    eps = 1e-6
    for k in range(3):
        for i in range(2):
            for j in range(2):
                orig = layer.W[k][i][j]
                layer.W[k][i][j] = orig + eps
                l1 = loss_fn()
                layer.W[k][i][j] = orig - eps
                l2 = loss_fn()
                layer.W[k][i][j] = orig
                num = (l1 - l2) / (2 * eps)
                assert abs(num - layer.gW[k][i][j]) < 1e-5, \
                    f"W[{k}][{i}][{j}]: {num} != {layer.gW[k][i][j]}"
    for k in range(3):
        for i in range(2):
            orig = layer.b[k][i]
            layer.b[k][i] = orig + eps
            l1 = loss_fn()
            layer.b[k][i] = orig - eps
            l2 = loss_fn()
            layer.b[k][i] = orig
            num = (l1 - l2) / (2 * eps)
            assert abs(num - layer.gb[k][i]) < 1e-5, \
                f"b[{k}][{i}]: {num} != {layer.gb[k][i]}"

    # d_H: numeric-check the input gradient too, since the transpose
    # path through A^k is the one place the row normalization (not
    # symmetric, unlike TAGCN) can bite.
    H = [list(row) for row in g.features]
    for v in range(3):
        for j in range(2):
            orig = H[v][j]
            H[v][j] = orig + eps
            o, _ = layer.forward(g, H)
            l1 = sum(d * x for row_d, row_o in zip(d_out, o)
                     for d, x in zip(row_d, row_o))
            H[v][j] = orig - eps
            o, _ = layer.forward(g, H)
            l2 = sum(d * x for row_d, row_o in zip(d_out, o)
                     for d, x in zip(row_d, row_o))
            H[v][j] = orig
            num = (l1 - l2) / (2 * eps)
            assert abs(num - d_H[v][j]) < 1e-5, \
                f"d_H[{v}][{j}]: {num} != {d_H[v][j]}"


def test_mixhop_model_gradient_check():
    # Full two-layer model + output head, numeric-checked end to end.
    g = Graph(3, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.set_features(0, [1.0, 0.5])
    g.set_features(1, [0.0, 2.0])
    g.set_features(2, [-1.0, 1.0])
    model = GraphMixHop(2, hidden_dim=3, K=2, lr=0.1, seed=0)
    targets = [0, 1, 2]
    labels = {0: 1, 1: 0, 2: 1}

    def loss_fn():
        probs, _ = model.forward(g, targets)
        total = 0.0
        for v in targets:
            p = min(max(probs[v], 1e-7), 1.0 - 1e-7)
            total += -(labels[v] * math.log(p)
                       + (1 - labels[v]) * math.log(1.0 - p))
        return total / len(targets)

    # Manual backward pass to collect analytic gradients.
    model.layer1.zero_grad()
    model.layer2.zero_grad()
    probs, cache = model.forward(g, targets)
    h2 = cache["h2"]
    dh2 = [[0.0] * 9 for _ in range(3)]
    gw_out = [0.0] * 9
    gb_out = 0.0
    for v in targets:
        d_logit = (probs[v] - labels[v]) / len(targets)
        for j in range(9):
            gw_out[j] += d_logit * h2[v][j]
        gb_out += d_logit
        dh2[v] = [w * d_logit for w in model.w_out]
    dh1 = model.layer2.backward(dh2, cache["cache2"])
    model.layer1.backward(dh1, cache["cache1"])

    eps = 1e-6
    for layer, name, in_dim in ((model.layer1, "layer1", 2),
                                (model.layer2, "layer2", 9)):
        for k in range(3):
            for i in range(3):
                for j in range(in_dim):
                    orig = layer.W[k][i][j]
                    layer.W[k][i][j] = orig + eps
                    l1 = loss_fn()
                    layer.W[k][i][j] = orig - eps
                    l2 = loss_fn()
                    layer.W[k][i][j] = orig
                    num = (l1 - l2) / (2 * eps)
                    assert abs(num - layer.gW[k][i][j]) < 1e-5, \
                        f"{name} W[{k}][{i}][{j}]: {num} != {layer.gW[k][i][j]}"
    for j in range(9):
        orig = model.w_out[j]
        model.w_out[j] = orig + eps
        l1 = loss_fn()
        model.w_out[j] = orig - eps
        l2 = loss_fn()
        model.w_out[j] = orig
        num = (l1 - l2) / (2 * eps)
        assert abs(num - gw_out[j]) < 1e-5, \
            f"w_out[{j}]: {num} != {gw_out[j]}"


def test_mixhop_state_dict_round_trip():
    model = GraphMixHop(2, hidden_dim=3, K=2, lr=0.1, seed=0)
    g = Graph(3, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.set_features(0, [1.0, 0.5])
    g.set_features(1, [0.0, 2.0])
    g.set_features(2, [-1.0, 1.0])
    before = model.predict(g, [0, 1, 2])
    state = model.state_dict()
    model2 = GraphMixHop(2, hidden_dim=3, K=2, lr=0.1, seed=9)
    model2.load_state_dict(state)
    after = model2.predict(g, [0, 1, 2])
    for v in (0, 1, 2):
        approx(before[v], after[v])


# ---------------------------------------------------------------------------
# training dynamics
# ---------------------------------------------------------------------------


def test_mixhop_trains_to_auc_above_baseline():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
    y_true = [g.labels[v] for v in val_ids]
    model, _ = train_mixhop(g, train_ids, val_ids, hidden_dim=16, K=2,
                            lr=0.5, epochs=100, patience=25, seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    auc = evaluate_all(y_true, scores)["auc"]
    assert auc >= 0.90, f"MixHop AUC {auc} below 0.90"

"""Tests for the TAGCN layer: hand-computed forward/backward, numeric
gradient checks, hop properties, and training dynamics."""

import math
import random

from gnn_fraud import (
    Graph,
    GraphGenerator,
    GraphTAG,
    TAGLayer,
    evaluate_all,
    train_tag,
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
    layer = TAGLayer(1, 1, K=K, rng=random.Random(0))
    layer.W = [[[0.0]] for _ in range(K + 1)]
    layer.b = [0.0]
    return layer


# ---------------------------------------------------------------------------
# forward: hand-computable cases
# ---------------------------------------------------------------------------


def test_tag_layer_forward_hand_computed():
    # Degrees with self-loops are 2, so A_hat weights are 1/2 each:
    # P0 = [1, 2]; P1 = [1.5, 1.5]; P2 = [1.5, 1.5] (rows sum to 1).
    g = _two_node_graph()
    layer = _layer_with_weights(2)
    layer.W[0] = [[1.0]]
    out, _ = layer.forward(g, g.features)
    approx(out[0][0], 1.0)
    approx(out[1][0], 2.0)

    layer = _layer_with_weights(2)
    layer.W[1] = [[1.0]]
    out, _ = layer.forward(g, g.features)
    approx(out[0][0], 1.5)
    approx(out[1][0], 1.5)

    layer = _layer_with_weights(2)
    layer.W[2] = [[1.0]]
    out, _ = layer.forward(g, g.features)
    approx(out[0][0], 1.5)
    approx(out[1][0], 1.5)

    # All hops together with unit weights: P0 + P1 + P2.
    layer = _layer_with_weights(2)
    for k in range(3):
        layer.W[k] = [[1.0]]
    out, _ = layer.forward(g, g.features)
    approx(out[0][0], 4.0)
    approx(out[1][0], 5.0)


def test_tag_layer_k0_is_linear():
    # K=0: only A_hat^0 = I, so the layer is relu(W0 @ x + b).
    g = _two_node_graph()
    layer = _layer_with_weights(0)
    layer.W[0] = [[2.0]]
    layer.b = [1.0]
    out, _ = layer.forward(g, g.features)
    approx(out[0][0], 3.0)
    approx(out[1][0], 5.0)


def test_tag_layer_isolated_node():
    # Node 2 is isolated: its A_hat row is just the self-loop with
    # weight 1, so every power returns its own feature, no div by 0.
    g = Graph(3, 1)
    g.add_edge(0, 1)
    g.set_features(0, [1.0])
    g.set_features(1, [2.0])
    g.set_features(2, [3.0])
    layer = _layer_with_weights(2)
    layer.W[2] = [[1.0]]
    out, _ = layer.forward(g, g.features)
    approx(out[2][0], 3.0)


# ---------------------------------------------------------------------------
# backward: numeric gradient checks
# ---------------------------------------------------------------------------


def test_tag_layer_gradient_check():
    g = Graph(3, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.set_features(0, [1.0, 0.5])
    g.set_features(1, [0.0, 2.0])
    g.set_features(2, [-1.0, 1.0])
    layer = TAGLayer(2, 2, K=2, rng=random.Random(1))
    out, cache = layer.forward(g, g.features)
    d_out = [[1.0, -0.5], [0.25, 1.0], [-1.0, 0.5]]
    layer.zero_grad()
    layer.backward(d_out, cache)

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
    for i in range(2):
        orig = layer.b[i]
        layer.b[i] = orig + eps
        l1 = loss_fn()
        layer.b[i] = orig - eps
        l2 = loss_fn()
        layer.b[i] = orig
        num = (l1 - l2) / (2 * eps)
        assert abs(num - layer.gb[i]) < 1e-5, \
            f"b[{i}]: {num} != {layer.gb[i]}"


def test_tag_model_gradient_check():
    # Full two-layer model + output head, numeric-checked end to end.
    g = Graph(3, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.set_features(0, [1.0, 0.5])
    g.set_features(1, [0.0, 2.0])
    g.set_features(2, [-1.0, 1.0])
    model = GraphTAG(2, hidden_dim=3, K=2, lr=0.1, seed=0)
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
    dh2 = [[0.0] * 3 for _ in range(3)]
    gw_out = [0.0] * 3
    gb_out = 0.0
    for v in targets:
        d_logit = (probs[v] - labels[v]) / len(targets)
        for j in range(3):
            gw_out[j] += d_logit * h2[v][j]
        gb_out += d_logit
        dh2[v] = [w * d_logit for w in model.w_out]
    dh1 = model.layer2.backward(dh2, cache["cache2"])
    model.layer1.backward(dh1, cache["cache1"])

    eps = 1e-6
    for layer, name in ((model.layer1, "layer1"), (model.layer2, "layer2")):
        for k in range(3):
            for i in range(3):
                for j in range(2 if name == "layer1" else 3):
                    orig = layer.W[k][i][j]
                    layer.W[k][i][j] = orig + eps
                    l1 = loss_fn()
                    layer.W[k][i][j] = orig - eps
                    l2 = loss_fn()
                    layer.W[k][i][j] = orig
                    num = (l1 - l2) / (2 * eps)
                    assert abs(num - layer.gW[k][i][j]) < 1e-5, \
                        f"{name} W[{k}][{i}][{j}]: {num} != {layer.gW[k][i][j]}"
    for j in range(3):
        orig = model.w_out[j]
        model.w_out[j] = orig + eps
        l1 = loss_fn()
        model.w_out[j] = orig - eps
        l2 = loss_fn()
        model.w_out[j] = orig
        num = (l1 - l2) / (2 * eps)
        assert abs(num - gw_out[j]) < 1e-5, \
            f"w_out[{j}]: {num} != {gw_out[j]}"


def test_tag_state_dict_round_trip():
    model = GraphTAG(2, hidden_dim=3, K=2, lr=0.1, seed=0)
    g = Graph(3, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.set_features(0, [1.0, 0.5])
    g.set_features(1, [0.0, 2.0])
    g.set_features(2, [-1.0, 1.0])
    before = model.predict(g, [0, 1, 2])
    state = model.state_dict()
    model2 = GraphTAG(2, hidden_dim=3, K=2, lr=0.1, seed=9)
    model2.load_state_dict(state)
    after = model2.predict(g, [0, 1, 2])
    for v in (0, 1, 2):
        approx(before[v], after[v])


# ---------------------------------------------------------------------------
# training dynamics
# ---------------------------------------------------------------------------


def test_tag_trains_to_auc_above_baseline():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
    y_true = [g.labels[v] for v in val_ids]
    model, _ = train_tag(g, train_ids, val_ids, hidden_dim=16, K=2,
                         lr=0.5, epochs=100, patience=25, seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    auc = evaluate_all(y_true, scores)["auc"]
    assert auc >= 0.90, f"TAGCN AUC {auc} below 0.90"

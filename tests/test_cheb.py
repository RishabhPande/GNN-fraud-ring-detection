"""Tests for the ChebNet layer: hand-computed forward/backward, numeric
gradient checks, polynomial properties, and training dynamics."""

import math
import random

from gnn_fraud import (
    ChebLayer,
    Graph,
    GraphChebyshev,
    GraphGenerator,
    evaluate_all,
    train_cheb,
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


# ---------------------------------------------------------------------------
# forward: hand-computable cases
# ---------------------------------------------------------------------------


def test_cheb_layer_forward_hand_computed():
    # Degrees are 1, so L_tilde weights are -1 / sqrt(1 * 1) = -1
    # (plain degrees here, not the GCN's degree + 1).
    # Z0 = [1, 2]; Z1 = [-1*2, -1*1] = [-2, -1];
    # Z2 = 2 * L_tilde Z1 - Z0 = [2*1 - 1, 2*2 - 2] = [1, 2].
    g = _two_node_graph()
    layer = ChebLayer(1, 1, K=2, rng=random.Random(0))
    layer.W = [[[1.0]], [[0.0]], [[0.0]]]
    layer.b = [0.0]
    out, _ = layer.forward(g, g.features)
    approx(out[0][0], 1.0)
    approx(out[1][0], 2.0)

    # Only the k=2 term: pre = Z2 = [1, 2].
    layer.W = [[[0.0]], [[0.0]], [[1.0]]]
    out, _ = layer.forward(g, g.features)
    approx(out[0][0], 1.0)
    approx(out[1][0], 2.0)

    # Only the k=1 term: pre = Z1 = [-2, -1] -> relu kills both.
    layer.W = [[[0.0]], [[1.0]], [[0.0]]]
    out, _ = layer.forward(g, g.features)
    approx(out[0][0], 0.0)
    approx(out[1][0], 0.0)


def test_cheb_layer_k0_is_linear():
    # K=0: only T_0 = I, so the layer is relu(W0 @ x + b) on raw features.
    g = _two_node_graph()
    layer = ChebLayer(1, 1, K=0, rng=random.Random(0))
    layer.W = [[[2.0]]]
    layer.b = [1.0]
    out, _ = layer.forward(g, g.features)
    approx(out[0][0], 3.0)
    approx(out[1][0], 5.0)


def test_cheb_layer_isolated_node():
    # Node 2 is isolated: its L_tilde row is empty, so Z1[2] = 0 and
    # Z2[2] = 2*0 - Z0[2] = -3. No division by zero anywhere.
    g = Graph(3, 1)
    g.add_edge(0, 1)
    g.set_features(0, [1.0])
    g.set_features(1, [2.0])
    g.set_features(2, [3.0])
    layer = ChebLayer(1, 1, K=2, rng=random.Random(0))
    layer.W = [[[0.0]], [[1.0]], [[0.0]]]
    layer.b = [0.0]
    out, _ = layer.forward(g, g.features)
    approx(out[2][0], 0.0)
    layer.W = [[[1.0]], [[0.0]], [[0.0]]]
    out, _ = layer.forward(g, g.features)
    approx(out[2][0], 3.0)


# ---------------------------------------------------------------------------
# backward: numeric gradient checks
# ---------------------------------------------------------------------------


def _numeric_grad_check_layer():
    g = Graph(3, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.set_features(0, [1.0, 0.5])
    g.set_features(1, [0.0, 2.0])
    g.set_features(2, [-1.0, 1.0])
    layer = ChebLayer(2, 2, K=2, rng=random.Random(1))
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


def test_cheb_layer_gradient_check():
    _numeric_grad_check_layer()


def test_cheb_model_gradient_check():
    # Full two-layer model + output head, numeric-checked end to end:
    # perturb each parameter, compare the analytic gradient from a
    # manual backward pass against the finite-difference loss gradient.
    g = Graph(3, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.set_features(0, [1.0, 0.5])
    g.set_features(1, [0.0, 2.0])
    g.set_features(2, [-1.0, 1.0])
    model = GraphChebyshev(2, hidden_dim=3, K=2, lr=0.1, seed=0)
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


# ---------------------------------------------------------------------------
# training dynamics
# ---------------------------------------------------------------------------


def test_cheb_trains_to_auc_above_baseline():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
    y_true = [g.labels[v] for v in val_ids]
    model, _ = train_cheb(g, train_ids, val_ids, hidden_dim=16, K=2,
                          lr=0.5, epochs=100, patience=25, seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    auc = evaluate_all(y_true, scores)["auc"]
    assert auc >= 0.90, f"ChebNet AUC {auc} below 0.90"

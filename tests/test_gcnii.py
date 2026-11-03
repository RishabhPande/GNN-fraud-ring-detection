"""Tests for the GCNII layer and model: hand-computed forward/backward,
numeric gradient checks, beta schedule, and training dynamics."""

import math
import random

from gnn_fraud import (
    Graph,
    GraphGCNII,
    GraphGenerator,
    evaluate_all,
    train_gcnii,
    train_val_split,
)
from gnn_fraud.gcn import GCNLayer
from gnn_fraud.gcnii import GCNIIConv


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


def test_gcnii_layer_forward_hand_computed():
    # Self-loop degrees are both 2, so every norm weight is 1/2.
    # agg = [0.5*1 + 0.5*2, 0.5*2 + 0.5*1] = [1.5, 1.5].
    # With alpha = 0.5 and H0 = [3, 4]:
    # mix = [0.5*1.5 + 0.5*3, 0.5*1.5 + 0.5*4] = [2.25, 2.75].
    # With beta = 1 (M = W = [[2.0]]) and b = 0: pre = [4.5, 5.5].
    g = _two_node_graph()
    layer = GCNIIConv(1, alpha=0.5, rng=random.Random(0))
    layer.W = [[2.0]]
    layer.b = [0.0]
    out, _ = layer.forward(g, g.features, [[3.0], [4.0]], beta=1.0)
    approx(out[0][0], 4.5)
    approx(out[1][0], 5.5)

    # beta = 0: M = I, so pre = mix exactly (b = 0).
    out, _ = layer.forward(g, g.features, [[3.0], [4.0]], beta=0.0)
    approx(out[0][0], 2.25)
    approx(out[1][0], 2.75)

    # alpha = 0, beta = 1: the residual drops out and the layer is a
    # plain GCN layer with the same W and b.
    gcn = GCNLayer(1, 1, rng=random.Random(0))
    gcn.W = [[2.0]]
    gcn.b = [0.0]
    expected, _ = gcn.forward(g, g.features)
    out, _ = layer.forward(g, g.features, [[3.0], [4.0]], beta=1.0)
    # Recompute with alpha = 0 on the GCNII side.
    layer0 = GCNIIConv(1, alpha=0.0, rng=random.Random(0))
    layer0.W = [[2.0]]
    layer0.b = [0.0]
    out, _ = layer0.forward(g, g.features, [[3.0], [4.0]], beta=1.0)
    approx(out[0][0], expected[0][0])
    approx(out[1][0], expected[1][0])


def test_gcnii_beta_schedule():
    model = GraphGCNII(4, hidden_dim=8, n_layers=4, lam=0.5, seed=0)
    for l, beta in enumerate(model.betas, start=1):
        approx(beta, math.log(0.5 / l + 1.0))
    # Betas decay with depth: deeper layers stay closer to identity.
    assert model.betas[0] > model.betas[-1] > 0.0


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
    h0 = [[0.5, -1.0], [1.0, 0.0], [0.25, 0.75]]
    layer = GCNIIConv(2, alpha=0.3, rng=random.Random(1))
    out, cache = layer.forward(g, g.features, h0, beta=0.4)
    d_out = [[1.0, -0.5], [0.25, 1.0], [-1.0, 0.5]]
    layer.zero_grad()
    layer.backward(d_out, cache)

    def loss_fn():
        o, _ = layer.forward(g, g.features, h0, beta=0.4)
        return sum(d * x for row_d, row_o in zip(d_out, o)
                   for d, x in zip(row_d, row_o))

    eps = 1e-6
    for i in range(2):
        for j in range(2):
            orig = layer.W[i][j]
            layer.W[i][j] = orig + eps
            l1 = loss_fn()
            layer.W[i][j] = orig - eps
            l2 = loss_fn()
            layer.W[i][j] = orig
            num = (l1 - l2) / (2 * eps)
            assert abs(num - layer.gW[i][j]) < 1e-5, \
                f"W[{i}][{j}]: {num} != {layer.gW[i][j]}"
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


def test_gcnii_layer_gradient_check():
    _numeric_grad_check_layer()


def test_gcnii_model_gradient_check():
    # Full model + output head, numeric-checked end to end: perturb
    # each parameter, compare the analytic gradient from a manual
    # backward pass against the finite-difference loss gradient. The
    # manual pass must reproduce train_batch's residual accumulation:
    # every layer's d_H0 plus the chain gradient into the first layer.
    g = Graph(3, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.set_features(0, [1.0, 0.5])
    g.set_features(1, [0.0, 2.0])
    g.set_features(2, [-1.0, 1.0])
    model = GraphGCNII(2, hidden_dim=3, n_layers=2, lr=0.1, seed=0)
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
    model.input_layer.zero_grad()
    for layer in model.layers:
        layer.zero_grad()
    probs, cache = model.forward(g, targets)
    h = cache["h"]
    dh = [[0.0] * 3 for _ in range(3)]
    gw_out = [0.0] * 3
    gb_out = 0.0
    for v in targets:
        d_logit = (probs[v] - labels[v]) / len(targets)
        for j in range(3):
            gw_out[j] += d_logit * h[v][j]
        gb_out += d_logit
        dh[v] = [w * d_logit for w in model.w_out]
    d_h0 = [[0.0] * 3 for _ in range(3)]
    for layer, layer_cache in zip(reversed(model.layers),
                                  reversed(cache["caches"])):
        dh, d_h0_layer = layer.backward(dh, layer_cache)
        for v in range(3):
            for j in range(3):
                d_h0[v][j] += d_h0_layer[v][j]
    for v in range(3):
        for j in range(3):
            d_h0[v][j] += dh[v][j]
    model.input_layer.backward(d_h0, cache["cache_in"])

    eps = 1e-6
    layer = model.input_layer
    for i in range(3):
        for j in range(2):
            orig = layer.W[i][j]
            layer.W[i][j] = orig + eps
            l1 = loss_fn()
            layer.W[i][j] = orig - eps
            l2 = loss_fn()
            layer.W[i][j] = orig
            num = (l1 - l2) / (2 * eps)
            assert abs(num - layer.gW[i][j]) < 1e-5, \
                f"input W[{i}][{j}]: {num} != {layer.gW[i][j]}"
    for li, layer in enumerate(model.layers):
        for i in range(3):
            for j in range(3):
                orig = layer.W[i][j]
                layer.W[i][j] = orig + eps
                l1 = loss_fn()
                layer.W[i][j] = orig - eps
                l2 = loss_fn()
                layer.W[i][j] = orig
                num = (l1 - l2) / (2 * eps)
                assert abs(num - layer.gW[i][j]) < 1e-5, \
                    f"layer{li} W[{i}][{j}]: {num} != {layer.gW[i][j]}"
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


def test_gcnii_trains_to_auc_above_baseline():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
    y_true = [g.labels[v] for v in val_ids]
    model, _ = train_gcnii(g, train_ids, val_ids, hidden_dim=16,
                           n_layers=4, lr=0.5, epochs=100, patience=25,
                           seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    auc = evaluate_all(y_true, scores)["auc"]
    assert auc >= 0.90, f"GCNII AUC {auc} below 0.90"


def test_gcnii_deep_stack_holds_up():
    # The point of GCNII is depth without collapse: 8 GCNII layers
    # (9 hops with the input layer) must still clear 0.90 on the
    # seed-7 dense ring, where a plain deep GCN would oversmooth.
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
    y_true = [g.labels[v] for v in val_ids]
    model, _ = train_gcnii(g, train_ids, val_ids, hidden_dim=16,
                           n_layers=8, lr=0.5, epochs=100, patience=25,
                           seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    auc = evaluate_all(y_true, scores)["auc"]
    assert auc >= 0.90, f"deep GCNII AUC {auc} below 0.90"

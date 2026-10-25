"""Tests for the GCN layer: hand-computed forward/backward, numeric
gradient checks, normalization properties, and training dynamics."""

import math
import random

from gnn_fraud import (
    GCNLayer,
    Graph,
    GraphConvolution,
    GraphGenerator,
    evaluate_all,
    train_gcn,
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


def test_gcn_layer_forward_hand_computed():
    # Degrees are 1, so d_hat = 2 for both nodes: every weight is
    # 1 / sqrt(2 * 2) = 0.5. agg_0 = agg_1 = 0.5*1 + 0.5*2 = 1.5.
    # W = [[1]], b = [0] -> out = relu(1.5) = 1.5 for both nodes.
    g = _two_node_graph()
    layer = GCNLayer(1, 1, random.Random(0))
    layer.W = [[1.0]]
    layer.b = [0.0]
    out, _ = layer.forward(g, g.features)
    approx(out[0][0], 1.5)
    approx(out[1][0], 1.5)


def test_gcn_layer_forward_bias_and_negative_weight():
    # W = [[2]], b = [1] -> pre = 2*1.5 + 1 = 4 for both nodes.
    # W = [[-2]], b = [1] -> pre = -2*1.5 + 1 = -2 -> relu kills it.
    g = _two_node_graph()
    layer = GCNLayer(1, 1, random.Random(0))
    layer.W = [[2.0]]
    layer.b = [1.0]
    out, _ = layer.forward(g, g.features)
    approx(out[0][0], 4.0)
    approx(out[1][0], 4.0)
    layer.W = [[-2.0]]
    out, _ = layer.forward(g, g.features)
    approx(out[0][0], 0.0)
    approx(out[1][0], 0.0)


def test_gcn_isolated_node_keeps_own_features():
    # An isolated node has d_hat = 1, so its only weight is
    # 1 / sqrt(1 * 1) = 1: the layer sees just its own features.
    g = Graph(3, 2)
    g.add_edge(0, 1)
    g.set_features(0, [1.0, 0.0])
    g.set_features(1, [0.0, 1.0])
    g.set_features(2, [3.0, -2.0])
    layer = GCNLayer(2, 2, random.Random(0))
    layer.W = [[1.0, 0.0], [0.0, 1.0]]
    layer.b = [0.0, 0.0]
    out, _ = layer.forward(g, g.features)
    approx(out[2][0], 3.0)
    approx(out[2][1], 0.0)  # relu(-2) = 0


def test_gcn_normalization_weights_symmetric():
    # c_vu = 1 / sqrt(d_hat_v * d_hat_u) is symmetric by construction,
    # and a hub's messages are down-weighted relative to a leaf's.
    from gnn_fraud.gcn import _norm_weights

    g = Graph(4, 1)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)  # node 0 is a hub of degree 3
    weights = _norm_weights(g)
    w = {v: dict(row) for v, row in enumerate(weights)}
    # Hub-to-leaf weight: 1 / sqrt(4 * 2); leaf self-weight: 1 / 2.
    approx(w[0][1], 1.0 / math.sqrt(4 * 2))
    approx(w[1][1], 1.0 / 2.0)
    approx(w[1][0], w[0][1])  # symmetry
    approx(w[0][0], 1.0 / 4.0)  # hub self-weight is the smallest


# ---------------------------------------------------------------------------
# backward: numeric gradient checks
# ---------------------------------------------------------------------------


def _numeric_grad_check():
    # 3-node path 0 -- 1 -- 2, 2-dim features, 2-dim output.
    g = Graph(3, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    rng = random.Random(3)
    layer = GCNLayer(2, 2, rng)
    d_out = [[0.3, -0.7], [1.1, 0.2], [-0.4, 0.9]]

    def loss_fn():
        out, _ = layer.forward(g, g.features)
        return sum(d * o for d_row, o_row in zip(d_out, out)
                   for d, o in zip(d_row, o_row))

    out, cache = layer.forward(g, g.features)
    d_H = layer.backward(d_out, cache)

    eps = 1e-6
    # Gradients w.r.t. W and b.
    for i in range(2):
        for j in range(2):
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
    for u in range(3):
        for j in range(2):
            orig = g.features[u][j]
            g.features[u][j] = orig + eps
            plus = loss_fn()
            g.features[u][j] = orig - eps
            minus = loss_fn()
            g.features[u][j] = orig
            approx(d_H[u][j], (plus - minus) / (2 * eps), tol=1e-5)
    return d_H


def test_gcn_layer_gradient_check():
    _numeric_grad_check()


def test_gcn_model_gradient_check():
    # Full two-layer model + output head, numeric-checked end to end:
    # perturb each parameter, compare against the gradient train_batch
    # accumulates. Uses a small graph so the check runs fast.
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    g.set_features(3, [0.1, 0.4])
    labels = {0: 1, 1: 0, 2: 1, 3: 0}
    model = GraphConvolution(2, hidden_dim=3, lr=0.0, seed=5)

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
    for layer, gW in ((model.layer1, model.layer1.gW),
                      (model.layer2, model.layer2.gW)):
        for i in range(len(layer.W)):
            for j in range(len(layer.W[0])):
                orig = layer.W[i][j]
                layer.W[i][j] = orig + eps
                plus = loss_fn()
                layer.W[i][j] = orig - eps
                minus = loss_fn()
                layer.W[i][j] = orig
                approx(gW[i][j], (plus - minus) / (2 * eps), tol=1e-5)
    for j in range(3):
        orig = model.w_out[j]
        model.w_out[j] = orig + eps
        plus = loss_fn()
        model.w_out[j] = orig - eps
        minus = loss_fn()
        model.w_out[j] = orig
        approx(model.gw_out[j], (plus - minus) / (2 * eps), tol=1e-5)


# ---------------------------------------------------------------------------
# checkpointing + training dynamics
# ---------------------------------------------------------------------------


def test_gcn_checkpoint_round_trip():
    model = GraphConvolution(6, hidden_dim=4, seed=1)
    state = model.state_dict()
    model2 = GraphConvolution(6, hidden_dim=4, seed=99)
    model2.load_state_dict(state)
    g = _two_node_graph()
    g2 = Graph(2, 6)
    g2.add_edge(0, 1)
    g2.set_features(0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    g2.set_features(1, [0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    p1 = model.predict(g2, [0, 1])
    p2 = model2.predict(g2, [0, 1])
    approx(p1[0], p2[0])
    approx(p1[1], p2[1])


def test_gcn_trains_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
    model, hist = train_gcn(g, train_ids, val_ids, hidden_dim=16,
                            lr=0.5, batch_size=64, epochs=100,
                            patience=25, seed=7)
    assert min(hist["val_loss"]) < hist["val_loss"][0]
    y_true = [g.labels[v] for v in val_ids]
    scores = [model.predict(g, [v])[v] for v in val_ids]
    auc = evaluate_all(y_true, scores)["auc"]
    assert auc >= 0.90, f"GCN AUC {auc:.3f} below 0.90"

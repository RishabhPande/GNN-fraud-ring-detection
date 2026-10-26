"""Tests for the GIN layer: hand-computed forward/backward, numeric
gradient checks (including the learnable eps), the sum-vs-mean
distinction, and training dynamics."""

import math
import random

from gnn_fraud import (
    GINLayer,
    Graph,
    GraphIsomorphism,
    GraphGenerator,
    evaluate_all,
    train_gin,
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


def _identity_layer(in_dim=1, out_dim=1):
    layer = GINLayer(in_dim, out_dim, random.Random(0))
    layer.W1 = [[1.0 if i == j else 0.0 for j in range(in_dim)]
                for i in range(out_dim)]
    layer.b1 = [0.0] * out_dim
    layer.W2 = [[1.0 if i == j else 0.0 for j in range(out_dim)]
                for i in range(out_dim)]
    layer.b2 = [0.0] * out_dim
    layer.eps = 0.0
    return layer


# ---------------------------------------------------------------------------
# forward: hand-computable cases
# ---------------------------------------------------------------------------


def test_gin_layer_forward_hand_computed():
    # eps = 0: agg_0 = 1*1 + 2 = 3, agg_1 = 1*2 + 1 = 3. Identity MLP
    # with ReLU passes 3 through unchanged for both nodes.
    g = _two_node_graph()
    layer = _identity_layer()
    out, _ = layer.forward(g, g.features)
    approx(out[0][0], 3.0)
    approx(out[1][0], 3.0)


def test_gin_layer_eps_weights_self():
    # eps = 1: agg_0 = 2*1 + 2 = 4, agg_1 = 2*2 + 1 = 5.
    g = _two_node_graph()
    layer = _identity_layer()
    layer.eps = 1.0
    out, _ = layer.forward(g, g.features)
    approx(out[0][0], 4.0)
    approx(out[1][0], 5.0)


def test_gin_layer_isolated_node_keeps_own_features():
    # An isolated node sums only itself: (1 + eps) * h_v.
    g = Graph(3, 2)
    g.add_edge(0, 1)
    g.set_features(0, [1.0, 0.0])
    g.set_features(1, [0.0, 1.0])
    g.set_features(2, [3.0, -2.0])
    layer = _identity_layer(in_dim=2, out_dim=2)
    out, _ = layer.forward(g, g.features)
    approx(out[2][0], 3.0)
    approx(out[2][1], 0.0)  # relu(-2) = 0


def test_gin_sum_distinguishes_multiset_size():
    # The point of sum over mean: two nodes whose neighbors have the
    # same *mean* feature but different *counts* must get different
    # embeddings. Node 0 has two neighbors with feature 1.0; node 2
    # has one neighbor with feature 1.0; all self-features are 0.
    g = Graph(4, 1)
    g.add_edge(0, 1)
    g.add_edge(0, 3)
    g.add_edge(2, 3)
    g.set_features(0, [0.0])
    g.set_features(1, [1.0])
    g.set_features(2, [0.0])
    g.set_features(3, [1.0])
    layer = _identity_layer()
    out, _ = layer.forward(g, g.features)
    approx(out[0][0], 2.0)  # two neighbors
    approx(out[2][0], 1.0)  # one neighbor -- a mean would give 1.0 both


# ---------------------------------------------------------------------------
# backward: numeric gradient checks
# ---------------------------------------------------------------------------


def test_gin_layer_gradient_check():
    # 3-node path 0 -- 1 -- 2, 2-dim features, 2-dim output.
    g = Graph(3, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    rng = random.Random(3)
    layer = GINLayer(2, 2, rng)
    layer.eps = 0.3
    # Nonzero biases: with zero biases, dead-activation nodes sit exactly
    # on the ReLU kink (pre2 == 0), where the numeric gradient is
    # undefined and the analytic 0 disagrees with the finite difference.
    layer.b1 = [0.1, -0.2]
    layer.b2 = [0.05, -0.04]
    d_out = [[0.3, -0.7], [1.1, 0.2], [-0.4, 0.9]]

    def loss_fn():
        out, _ = layer.forward(g, g.features)
        return sum(d * o for d_row, o_row in zip(d_out, out)
                   for d, o in zip(d_row, o_row))

    out, cache = layer.forward(g, g.features)
    d_H = layer.backward(d_out, cache)

    eps = 1e-6
    for W, gW in ((layer.W1, layer.gW1), (layer.W2, layer.gW2)):
        for i in range(2):
            for j in range(2):
                orig = W[i][j]
                W[i][j] = orig + eps
                plus = loss_fn()
                W[i][j] = orig - eps
                minus = loss_fn()
                W[i][j] = orig
                approx(gW[i][j], (plus - minus) / (2 * eps), tol=1e-5)
    for b, gb in ((layer.b1, layer.gb1), (layer.b2, layer.gb2)):
        for i in range(2):
            orig = b[i]
            b[i] = orig + eps
            plus = loss_fn()
            b[i] = orig - eps
            minus = loss_fn()
            b[i] = orig
            approx(gb[i], (plus - minus) / (2 * eps), tol=1e-5)
    # Learnable eps.
    orig = layer.eps
    layer.eps = orig + eps
    plus = loss_fn()
    layer.eps = orig - eps
    minus = loss_fn()
    layer.eps = orig
    approx(layer.geps, (plus - minus) / (2 * eps), tol=1e-5)
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


def test_gin_model_gradient_check():
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
    model = GraphIsomorphism(2, hidden_dim=3, lr=0.0, seed=5)

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
        for W, gW in ((layer.W1, layer.gW1), (layer.W2, layer.gW2)):
            for i in range(len(W)):
                for j in range(len(W[0])):
                    orig = W[i][j]
                    W[i][j] = orig + eps
                    plus = loss_fn()
                    W[i][j] = orig - eps
                    minus = loss_fn()
                    W[i][j] = orig
                    approx(gW[i][j], (plus - minus) / (2 * eps), tol=1e-5)
        orig = layer.eps
        layer.eps = orig + eps
        plus = loss_fn()
        layer.eps = orig - eps
        minus = loss_fn()
        layer.eps = orig
        approx(layer.geps, (plus - minus) / (2 * eps), tol=1e-5)
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


def test_gin_checkpoint_round_trip():
    model = GraphIsomorphism(6, hidden_dim=4, seed=1)
    state = model.state_dict()
    model2 = GraphIsomorphism(6, hidden_dim=4, seed=99)
    model2.load_state_dict(state)
    g2 = Graph(2, 6)
    g2.add_edge(0, 1)
    g2.set_features(0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    g2.set_features(1, [0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    p1 = model.predict(g2, [0, 1])
    p2 = model2.predict(g2, [0, 1])
    approx(p1[0], p2[0])
    approx(p1[1], p2[1])
    assert model2.layer1.eps == model.layer1.eps


def test_gin_trains_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
    model, hist = train_gin(g, train_ids, val_ids, hidden_dim=16,
                            lr=0.1, batch_size=64, epochs=60,
                            patience=12, seed=7)
    assert min(hist["val_loss"]) < hist["val_loss"][0]
    y_true = [g.labels[v] for v in val_ids]
    scores = [model.predict(g, [v])[v] for v in val_ids]
    auc = evaluate_all(y_true, scores)["auc"]
    assert auc >= 0.90, f"GIN AUC {auc:.3f} below 0.90"

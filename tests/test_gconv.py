"""Tests for GraphConv: hand-computed layer forward (separate self
and neighbour matrices over a raw sum), isolated-node behaviour,
numeric gradient checks at the layer (including the input gradient)
and end to end, checkpointing, bad-parameter rejection, and training
dynamics on the dense ring."""

import math
import random

from gnn_fraud import (
    Graph,
    GraphConvLayer,
    GraphConvNet,
    GraphGenerator,
    evaluate_all,
    train_gconv,
    train_val_split,
)


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def test_gconv_layer_hand_computed():
    # Identity self matrix, zero neighbour matrix: output is relu(x).
    rng = random.Random(0)
    layer = GraphConvLayer(2, 2, rng)
    layer.W_self = [[1.0, 0.0], [0.0, 1.0]]
    layer.W_neigh = [[0.0, 0.0], [0.0, 0.0]]
    layer.b = [0.0, 0.0]
    g = Graph(2, 2)
    g.add_edge(0, 1)
    out, _ = layer.forward(g, [[1.0, -2.0], [3.0, 4.0]])
    approx(out[0][0], 1.0)
    approx(out[0][1], 0.0)
    approx(out[1][0], 3.0)
    approx(out[1][1], 4.0)


def test_gconv_layer_neighbour_sum_counts():
    # Zero self matrix, identity neighbour matrix: node 0 sums both
    # neighbours (count preserved), a leaf sums its one neighbour.
    rng = random.Random(0)
    layer = GraphConvLayer(1, 1, rng)
    layer.W_self = [[0.0]]
    layer.W_neigh = [[1.0]]
    layer.b = [0.0]
    g = Graph(3, 1)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    out, _ = layer.forward(g, [[10.0], [1.0], [2.0]])
    approx(out[0][0], 3.0)
    approx(out[1][0], 10.0)
    approx(out[2][0], 10.0)


def test_gconv_isolated_node_uses_self_only():
    rng = random.Random(0)
    layer = GraphConvLayer(1, 1, rng)
    layer.W_self = [[2.0]]
    layer.W_neigh = [[5.0]]
    layer.b = [1.0]
    g = Graph(1, 1)
    out, _ = layer.forward(g, [[3.0]])
    approx(out[0][0], 7.0)


def test_gconv_bad_params_rejected():
    try:
        GraphConvNet(6, hidden_dim=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _loss(model, g, labels):
    probs, _ = model.forward(g, [0, 1, 2, 3])
    total = 0.0
    for v in [0, 1, 2, 3]:
        p = min(max(probs[v], 1e-7), 1.0 - 1e-7)
        y = labels[v]
        total += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
    return total / 4


def test_gconv_model_gradient_check():
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    g.set_features(3, [0.1, 0.4])
    labels = {0: 1, 1: 0, 2: 1, 3: 0}
    model = GraphConvNet(2, hidden_dim=3, lr=0.0, seed=5)
    model.train_batch(g, [0, 1, 2, 3], labels)
    eps = 1e-6

    def check(analytic, get, set_):
        orig = get()
        set_(orig + eps)
        plus = _loss(model, g, labels)
        set_(orig - eps)
        minus = _loss(model, g, labels)
        set_(orig)
        approx(analytic, (plus - minus) / (2 * eps), tol=1e-5)

    for layer in (model.layer1, model.layer2):
        for attr, gattr in (("W_self", "gW_self"), ("W_neigh", "gW_neigh")):
            W = getattr(layer, attr)
            G = getattr(layer, gattr)
            for i in range(len(W)):
                for j in range(len(W[i])):
                    check(G[i][j],
                          lambda W=W, i=i, j=j: W[i][j],
                          lambda v, W=W, i=i, j=j: W[i].__setitem__(j, v))
        for i in range(layer.out_dim):
            check(layer.gb[i],
                  lambda layer=layer, i=i: layer.b[i],
                  lambda v, layer=layer, i=i: layer.b.__setitem__(i, v))
    for i in range(model.hidden_dim):
        check(model.gw_out[i],
              lambda i=i: model.w_out[i],
              lambda v, i=i: model.w_out.__setitem__(i, v))
    check(model.gb_out, lambda: model.b_out,
          lambda v: setattr(model, "b_out", v))


def test_gconv_layer_input_gradient_check():
    # d_H from backward must match numeric gradients w.r.t. inputs.
    rng = random.Random(3)
    layer = GraphConvLayer(2, 2, rng)
    g = Graph(3, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    H = [[0.5, -1.0], [1.5, 0.25], [-0.75, 2.0]]
    out, cache = layer.forward(g, H)
    d_out = [[1.0, -0.5], [0.25, 2.0], [-1.0, 0.75]]
    d_H = layer.backward(d_out, cache)
    eps = 1e-6

    def scalar():
        o, _ = layer.forward(g, H)
        return sum(o[v][j] * d_out[v][j]
                   for v in range(3) for j in range(2))

    for v in range(3):
        for j in range(2):
            orig = H[v][j]
            H[v][j] = orig + eps
            plus = scalar()
            H[v][j] = orig - eps
            minus = scalar()
            H[v][j] = orig
            approx(d_H[v][j], (plus - minus) / (2 * eps), tol=1e-5)


def test_gconv_checkpoint_round_trip():
    model = GraphConvNet(6, hidden_dim=8, seed=1)
    state = model.state_dict()
    model2 = GraphConvNet(6, hidden_dim=8, seed=99)
    model2.load_state_dict(state)
    g = Graph(2, 6)
    g.add_edge(0, 1)
    g.set_features(0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    g.set_features(1, [0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    p1 = model.predict(g, [0, 1])
    p2 = model2.predict(g, [0, 1])
    approx(p1[0], p2[0])
    approx(p1[1], p2[1])


def test_gconv_trains_to_high_auc_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    model, _ = train_gconv(g, train_ids, val_ids, hidden_dim=16,
                           lr=0.1, batch_size=64, epochs=60,
                           patience=12, seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    y_true = [g.labels[v] for v in val_ids]
    assert evaluate_all(y_true, scores)["auc"] >= 0.90

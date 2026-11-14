"""Tests for the ARMA-GNN: hand-computed propagation and layer
forward, the iters=0 no-graph equivalence, numeric gradient checks
(layer input gradient and full model end to end, including the
shared recurrent weight), checkpointing, and training dynamics on
the dense ring."""

import math
import random

from gnn_fraud import (
    ARMALayer,
    Graph,
    GraphARMA,
    GraphGenerator,
    evaluate_all,
    normalized_propagate,
    train_arma,
    train_val_split,
)


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def _two_node_graph():
    g = Graph(2, 1)
    g.add_edge(0, 1)
    g.set_features(0, [1.0])
    g.set_features(1, [3.0])
    return g


def _hand_layer(iters):
    """1-dim in, 1-dim out layer with every weight set by hand."""
    layer = ARMALayer(1, 1, iters, random.Random(0))
    layer.W_in = [[1.0]]
    layer.W_rec = [[0.5]]
    layer.b = [0.1]
    return layer


# ---------------------------------------------------------------------------
# hand-computed forward
# ---------------------------------------------------------------------------


def test_arma_propagate_two_node_hand_computed():
    g = _two_node_graph()
    out = normalized_propagate(g, [[1.0], [3.0]])
    # Both hat degrees are 2, so every entry of S_hat is 1/2 and
    # each node gets the plain mean of the two vectors.
    approx(out[0][0], 2.0)
    approx(out[1][0], 2.0)


def test_arma_propagate_isolated_node_is_identity():
    g = Graph(3, 1)
    g.add_edge(0, 1)
    out = normalized_propagate(g, [[1.0], [3.0], [7.0]])
    # Node 2 is isolated: hat degree 1, self-loop weight 1, so its
    # propagated value is exactly its own vector.
    approx(out[2][0], 7.0)


def test_arma_layer_hand_computed():
    g = _two_node_graph()
    layer = _hand_layer(iters=1)
    out, _ = layer.forward(g, g.features)
    # base = [1, 3]; Z^(0) = [1.1, 3.1]; propagated mean = 2.1 for
    # both nodes; Z^(1) = 0.5 * 2.1 + base + 0.1 = [2.15, 4.15];
    # both positive, so the output ReLU changes nothing.
    approx(out[0][0], 2.15)
    approx(out[1][0], 4.15)


def test_arma_iters_zero_is_plain_linear_relu():
    # iters=0 degrades exactly to a linear-plus-ReLU layer over raw
    # features: no propagation, so the graph structure is irrelevant
    # and each node is scored from its own features alone.
    g = _two_node_graph()
    layer = _hand_layer(iters=0)
    out, _ = layer.forward(g, g.features)
    approx(out[0][0], 1.1)
    approx(out[1][0], 3.1)
    g2 = Graph(2, 1)  # same features, no edge at all
    g2.set_features(0, [1.0])
    g2.set_features(1, [3.0])
    out2, _ = layer.forward(g2, g2.features)
    approx(out2[0][0], 1.1)
    approx(out2[1][0], 3.1)


def test_arma_bad_params_rejected():
    for kwargs in ({"in_dim": 0, "out_dim": 2, "iters": 1},
                   {"in_dim": 2, "out_dim": 0, "iters": 1},
                   {"in_dim": 2, "out_dim": 2, "iters": -1}):
        try:
            ARMALayer(rng=random.Random(0), **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")
    for kwargs in ({"hidden_dim": 0}, {"iters": -2}):
        try:
            GraphARMA(6, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


# ---------------------------------------------------------------------------
# numeric gradient checks
# ---------------------------------------------------------------------------


def test_arma_layer_input_gradient_check():
    # Perturb the layer's input vectors directly and compare with
    # the d_H that backward returns for a fixed output weighting.
    # This is the check that validates the transpose propagation
    # through the recursion (S_hat is symmetric, so its transpose
    # is itself) separately from the parameter gradients.
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    H = [[0.5, -1.0], [1.5, 0.25], [-0.75, 2.0], [0.1, 0.4]]
    weights = [[0.3, -0.7, 0.5], [1.1, 0.2, -0.2],
               [-0.4, 0.9, 0.1], [0.6, -1.3, 0.8]]
    layer = ARMALayer(2, 3, 2, random.Random(4))
    layer.zero_grad()
    out, cache = layer.forward(g, H)
    d_H = layer.backward(weights, cache)

    def scalar():
        o, _ = layer.forward(g, H)
        return sum(weights[v][j] * o[v][j]
                   for v in range(4) for j in range(3))

    eps = 1e-6
    for v in range(4):
        for j in range(2):
            orig = H[v][j]
            H[v][j] = orig + eps
            plus = scalar()
            H[v][j] = orig - eps
            minus = scalar()
            H[v][j] = orig
            approx(d_H[v][j], (plus - minus) / (2 * eps), tol=1e-5)


def _loss(model, g, labels):
    probs, _ = model.forward(g, [0, 1, 2, 3])
    total = 0.0
    for v in [0, 1, 2, 3]:
        p = min(max(probs[v], 1e-7), 1.0 - 1e-7)
        y = labels[v]
        total += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
    return total / 4


def test_arma_model_gradient_check():
    # Perturb every parameter of both layers (input matrix, shared
    # recurrent matrix, bias) plus the head, and compare with the
    # gradients train_batch accumulates at lr=0.0. W_rec is shared
    # across iterations, so its analytic gradient is a sum over
    # steps; a backward that only counted one step would fail here.
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    g.set_features(3, [0.1, 0.4])
    labels = {0: 1, 1: 0, 2: 1, 3: 0}
    model = GraphARMA(2, hidden_dim=3, iters=2, lr=0.0, seed=5)
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
        for W, gW in ((layer.W_in, layer.gW_in),
                      (layer.W_rec, layer.gW_rec)):
            for i in range(len(W)):
                for j in range(len(W[i])):
                    check(gW[i][j],
                          lambda W=W, i=i, j=j: W[i][j],
                          lambda v, W=W, i=i, j=j: W[i].__setitem__(j, v))
        for i in range(layer.out_dim):
            check(layer.gb[i],
                  lambda layer=layer, i=i: layer.b[i],
                  lambda v, layer=layer, i=i: layer.b.__setitem__(i, v))
    for j in range(model.hidden_dim):
        check(model.gw_out[j],
              lambda j=j: model.w_out[j],
              lambda v, j=j: model.w_out.__setitem__(j, v))
    check(model.gb_out, lambda: model.b_out,
          lambda v: setattr(model, "b_out", v))


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------


def test_arma_checkpoint_round_trip():
    model = GraphARMA(6, hidden_dim=8, iters=2, seed=1)
    state = model.state_dict()
    model2 = GraphARMA(6, hidden_dim=8, iters=2, seed=99)
    model2.load_state_dict(state)
    g = Graph(2, 6)
    g.add_edge(0, 1)
    g.set_features(0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    g.set_features(1, [0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    p1 = model.predict(g, [0, 1])
    p2 = model2.predict(g, [0, 1])
    approx(p1[0], p2[0])
    approx(p1[1], p2[1])


# ---------------------------------------------------------------------------
# training dynamics
# ---------------------------------------------------------------------------


def test_arma_trains_to_high_auc_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    model, _ = train_arma(g, train_ids, val_ids, hidden_dim=16,
                          iters=3, lr=0.1, batch_size=64, epochs=150,
                          patience=40, seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    y_true = [g.labels[v] for v in val_ids]
    assert evaluate_all(y_true, scores)["auc"] >= 0.90


def test_arma_recurrent_weight_moves_off_init():
    # The recursion is the finding, so pin that training actually
    # moves W_rec: a dead recurrent branch would leave the model a
    # plain two-layer linear-plus-ReLU stack over raw features.
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    fresh = GraphARMA(g.feature_dim, hidden_dim=16, iters=3, seed=7)
    init = sum(abs(x) for row in fresh.layer1.W_rec for x in row)
    model, _ = train_arma(g, train_ids, val_ids, hidden_dim=16,
                          iters=3, lr=0.1, batch_size=64, epochs=150,
                          patience=40, seed=7)
    trained = sum(abs(x) for row in model.layer1.W_rec for x in row)
    assert abs(trained - init) > 0.5

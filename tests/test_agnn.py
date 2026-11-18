"""Tests for AGNN: hand-computed cosine-attention propagation,
beta=0 uniform-mean equivalence, steps=0 MLP equivalence, numeric
gradient checks end to end (including the learnable betas, set off
their init so the attention path is live), checkpointing,
bad-parameter rejection, and training dynamics on the dense ring.
"""

import math

from gnn_fraud import (
    AGNN,
    Graph,
    GraphGenerator,
    attention_weights_for,
    evaluate_all,
    propagate_attention,
    train_agnn,
    train_val_split,
)


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def _two_node_graph():
    g = Graph(2, 2)
    g.add_edge(0, 1)
    g.set_features(0, [1.0, 0.0])
    g.set_features(1, [0.0, 1.0])
    return g


def test_agnn_beta_zero_is_uniform_mean():
    # At beta=0 every weight is 1/2 on the two-node graph, so each
    # node's output is the plain mean of both embeddings.
    g = _two_node_graph()
    H = [[1.0, 0.0], [0.0, 1.0]]
    out, _ = propagate_attention(g, H, 0.0)
    for v in range(2):
        approx(out[v][0], 0.5)
        approx(out[v][1], 0.5)
    w = attention_weights_for(H, [0, 1], 0, 0.0)
    approx(w[0], 0.5)
    approx(w[1], 0.5)


def test_agnn_attention_prefers_similar_neighbor():
    # h0 = [1, 0]; neighbors itself (cos 1) and [0, 1] (cos 0).
    # With beta=2 the self weight is e^2 / (e^2 + 1).
    H = [[1.0, 0.0], [0.0, 1.0]]
    w = attention_weights_for(H, [0, 1], 0, 2.0)
    expected = math.exp(2.0) / (math.exp(2.0) + 1.0)
    approx(w[0], expected)
    approx(w[1], 1.0 - expected)
    approx(sum(w), 1.0)


def test_agnn_propagate_hand_computed():
    g = _two_node_graph()
    H = [[1.0, 0.0], [0.0, 1.0]]
    out, _ = propagate_attention(g, H, 2.0)
    a = math.exp(2.0) / (math.exp(2.0) + 1.0)
    approx(out[0][0], a)
    approx(out[0][1], 1.0 - a)
    approx(out[1][0], 1.0 - a)
    approx(out[1][1], a)


def test_agnn_isolated_node_keeps_own_embedding():
    g = Graph(1, 2)
    H = [[3.0, 4.0]]
    out, _ = propagate_attention(g, H, 5.0)
    approx(out[0][0], 3.0)
    approx(out[0][1], 4.0)


def test_agnn_zero_vector_guard():
    # A zero embedding has no direction: cosine 0 against everything,
    # and propagation must not produce NaNs.
    g = _two_node_graph()
    H = [[0.0, 0.0], [1.0, 0.0]]
    out, _ = propagate_attention(g, H, 3.0)
    for row in out:
        for x in row:
            assert math.isfinite(x)


def test_agnn_steps_zero_is_mlp_on_raw_features():
    g = _two_node_graph()
    model = AGNN(2, hidden_dim=3, steps=0, seed=4)
    probs = model.predict(g, [0, 1])
    # Recompute the encoder-plus-head by hand.
    for v in range(2):
        x = g.features[v]
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b)
             for row, b in zip(model.W1, model.b1)]
        z = sum(w * hj for w, hj in zip(model.w_out, h)) + model.b_out
        expected = 1.0 / (1.0 + math.exp(-z))
        approx(probs[v], expected)


def test_agnn_bad_params_rejected():
    for kwargs in ({"steps": -1}, {"hidden_dim": 0}):
        try:
            AGNN(6, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


def _loss(model, g, labels):
    probs, _ = model.forward(g, [0, 1, 2, 3])
    total = 0.0
    for v in [0, 1, 2, 3]:
        p = min(max(probs[v], 1e-7), 1.0 - 1e-7)
        y = labels[v]
        total += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
    return total / 4


def test_agnn_model_gradient_check():
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(0, 2)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    g.set_features(3, [0.1, 0.4])
    labels = {0: 1, 1: 0, 2: 1, 3: 0}
    model = AGNN(2, hidden_dim=3, steps=2, lr=0.0, seed=5)
    # Betas off their init and distinct, so both attention paths and
    # both beta gradients are live and distinguishable.
    model.betas = [0.7, 1.3]
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

    for i in range(model.hidden_dim):
        for j in range(model.in_dim):
            check(model.gW1[i][j],
                  lambda i=i, j=j: model.W1[i][j],
                  lambda v, i=i, j=j: model.W1[i].__setitem__(j, v))
        check(model.gb1[i],
              lambda i=i: model.b1[i],
              lambda v, i=i: model.b1.__setitem__(i, v))
        check(model.gw_out[i],
              lambda i=i: model.w_out[i],
              lambda v, i=i: model.w_out.__setitem__(i, v))
    for k in range(model.steps):
        check(model.gbetas[k],
              lambda k=k: model.betas[k],
              lambda v, k=k: model.betas.__setitem__(k, v))
    check(model.gb_out, lambda: model.b_out,
          lambda v: setattr(model, "b_out", v))


def test_agnn_checkpoint_round_trip():
    model = AGNN(6, hidden_dim=8, steps=2, seed=1)
    state = model.state_dict()
    model2 = AGNN(6, hidden_dim=8, steps=2, seed=99)
    model2.load_state_dict(state)
    assert model2.betas == model.betas
    g = Graph(2, 6)
    g.add_edge(0, 1)
    g.set_features(0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    g.set_features(1, [0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    p1 = model.predict(g, [0, 1])
    p2 = model2.predict(g, [0, 1])
    approx(p1[0], p2[0])
    approx(p1[1], p2[1])


def test_agnn_attention_weights_sum_to_one():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=50, ring_size=8)
    model = AGNN(g.feature_dim, hidden_dim=8, steps=2, seed=3)
    w = model.attention_weights(g, 0)
    approx(sum(w), 1.0)
    assert len(w) == g.degree(0) + 1


def test_agnn_trains_to_high_auc_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    model, _ = train_agnn(g, train_ids, val_ids, hidden_dim=16,
                          steps=2, lr=0.5, batch_size=64, epochs=100,
                          patience=25, seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    y_true = [g.labels[v] for v in val_ids]
    assert evaluate_all(y_true, scores)["auc"] >= 0.90

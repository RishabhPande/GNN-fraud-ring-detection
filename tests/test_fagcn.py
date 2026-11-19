"""Tests for FAGCN: frequency split identities (low+high == input,
isolated node has zero high part), steps=0 MLP equivalence, numeric
gradient checks end to end including alpha and beta off their init,
checkpointing, bad-parameter rejection, and training on the ring."""

import math

from gnn_fraud import (
    FAGCN,
    Graph,
    GraphGenerator,
    evaluate_all,
    frequency_split,
    train_fagcn,
    train_val_split,
)


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def test_frequency_split_sums_to_input():
    g = Graph(3, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    H = [[1.0, -2.0], [3.0, 4.0], [-0.5, 0.25]]
    low, high = frequency_split(g, H)
    for v in range(3):
        for j in range(2):
            approx(low[v][j] + high[v][j], H[v][j])


def test_frequency_split_isolated_node_high_is_zero():
    g = Graph(2, 1)
    # No edges: node 0 is isolated, its S row is [1.0] on itself.
    low, high = frequency_split(g, [[3.0], [-1.0]])
    approx(low[0][0], 3.0)
    approx(high[0][0], 0.0)


def test_fagcn_steps_zero_is_encoder_only():
    # steps=0: no propagation, predict must not depend on edges.
    g1 = Graph(2, 2)
    g1.set_features(0, [0.5, -1.0])
    g1.set_features(1, [1.5, 0.25])
    g2 = Graph(2, 2)
    g2.add_edge(0, 1)
    g2.set_features(0, [0.5, -1.0])
    g2.set_features(1, [1.5, 0.25])
    m = FAGCN(2, hidden_dim=3, steps=0, seed=5)
    approx(m.predict(g1, [0])[0], m.predict(g2, [0])[0])


def test_fagcn_bad_params_rejected():
    for kwargs in ({"hidden_dim": 0}, {"steps": -1}):
        try:
            FAGCN(6, **kwargs)
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


def test_fagcn_gradient_check():
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    g.set_features(3, [0.1, 0.4])
    labels = {0: 1, 1: 0, 2: 1, 3: 0}
    model = FAGCN(2, hidden_dim=3, steps=2, lr=0.0, seed=5)
    # Move alpha/beta off init so both channels carry gradient.
    model.alpha = 0.7
    model.beta = 0.3
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

    for i in range(len(model.W1)):
        for j in range(len(model.W1[i])):
            check(model.gW1[i][j],
                  lambda i=i, j=j: model.W1[i][j],
                  lambda v, i=i, j=j: model.W1[i].__setitem__(j, v))
    for i in range(model.hidden_dim):
        check(model.gb1[i],
              lambda i=i: model.b1[i],
              lambda v, i=i: model.b1.__setitem__(i, v))
        check(model.gw_out[i],
              lambda i=i: model.w_out[i],
              lambda v, i=i: model.w_out.__setitem__(i, v))
    check(model.galpha, lambda: model.alpha,
          lambda v: setattr(model, "alpha", v))
    check(model.gbeta, lambda: model.beta,
          lambda v: setattr(model, "beta", v))
    check(model.gb_out, lambda: model.b_out,
          lambda v: setattr(model, "b_out", v))


def test_fagcn_checkpoint_round_trip():
    model = FAGCN(6, hidden_dim=8, steps=2, seed=1)
    state = model.state_dict()
    model2 = FAGCN(6, hidden_dim=8, steps=2, seed=99)
    model2.load_state_dict(state)
    g = Graph(2, 6)
    g.add_edge(0, 1)
    g.set_features(0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    g.set_features(1, [0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    p1 = model.predict(g, [0, 1])
    p2 = model2.predict(g, [0, 1])
    approx(p1[0], p2[0])
    approx(p1[1], p2[1])


def test_fagcn_trains_to_high_auc_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    model, _ = train_fagcn(g, train_ids, val_ids, hidden_dim=16, steps=2,
                           lr=0.25, batch_size=64, epochs=100,
                           patience=25, seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    y_true = [g.labels[v] for v in val_ids]
    assert evaluate_all(y_true, scores)["auc"] >= 0.90

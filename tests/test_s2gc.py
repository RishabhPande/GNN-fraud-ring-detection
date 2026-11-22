"""Tests for S2GC: hand-computed self-looped propagation powers
and their average, the isolated-node identity, numeric gradient
checks end to end, checkpointing, bad-parameter rejection, and
training dynamics on the dense ring."""

import math

from gnn_fraud import (
    Graph,
    GraphGenerator,
    S2GC,
    averaged_propagation,
    evaluate_all,
    s2gc_propagation_powers,
    train_s2gc,
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


# ---------------------------------------------------------------------------
# hand-computed powers and average
# ---------------------------------------------------------------------------


def test_s2gc_powers_two_node_hand_computed():
    g = _two_node_graph()
    # Degrees with self-loops are 2 and 2, so S_hat maps
    # [a, b] to [(a + b) / 2, (a + b) / 2]: one step collapses
    # both nodes to the mean, and every later power stays there.
    powers = s2gc_propagation_powers(g, [[1.0], [3.0]], 2)
    approx(powers[0][0][0], 1.0)
    approx(powers[1][0][0], 2.0)
    approx(powers[1][1][0], 2.0)
    approx(powers[2][0][0], 2.0)
    approx(powers[2][1][0], 2.0)


def test_s2gc_average_two_node_hand_computed():
    g = _two_node_graph()
    # Mean of the powers above: node 0 gets (1 + 2 + 2) / 3,
    # node 1 gets (3 + 2 + 2) / 3.
    avg = averaged_propagation(g, [[1.0], [3.0]], 2)
    approx(avg[0][0], 5.0 / 3.0)
    approx(avg[1][0], 7.0 / 3.0)


def test_s2gc_isolated_node_keeps_own_state():
    g = Graph(3, 1)
    g.add_edge(0, 1)
    # Node 2 is isolated: with the self-loop its only neighbor is
    # itself at weight 1, so every power is its own state and the
    # average is exact, where the no-self-loop operators zero it.
    avg = averaged_propagation(g, [[1.0], [3.0], [7.0]], 4)
    approx(avg[2][0], 7.0)


def test_s2gc_K_zero_returns_copy():
    g = _two_node_graph()
    h0 = [[1.0], [3.0]]
    avg = averaged_propagation(g, h0, 0)
    assert avg == h0
    avg[0][0] = 99.0
    assert h0[0][0] == 1.0


def test_s2gc_K_zero_model_ignores_graph():
    # K = 0: Z = h0 exactly, so edges change nothing.
    g = _two_node_graph()
    g2 = Graph(2, 1)
    g2.set_features(0, [1.0])
    g2.set_features(1, [3.0])
    model = S2GC(1, hidden_dim=4, K=0, seed=3)
    p1 = model.predict(g, [0, 1])
    p2 = model.predict(g2, [0, 1])
    approx(p1[0], p2[0])
    approx(p1[1], p2[1])


def test_s2gc_bad_params_rejected():
    for kwargs in ({"K": -1}, {"hidden_dim": 0}):
        try:
            S2GC(6, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")
    for fn in (lambda: averaged_propagation(_two_node_graph(),
                                            [[1.0], [2.0]], -1),
               lambda: s2gc_propagation_powers(_two_node_graph(),
                                               [[1.0]], 1)):
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


# ---------------------------------------------------------------------------
# numeric gradient check
# ---------------------------------------------------------------------------


def _loss(model, g, labels):
    probs, _ = model.forward(g, [0, 1, 2, 3])
    total = 0.0
    for v in [0, 1, 2, 3]:
        p = min(max(probs[v], 1e-7), 1.0 - 1e-7)
        y = labels[v]
        total += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
    return total / 4


def test_s2gc_model_gradient_check():
    # Perturb every parameter (encoder and head; the filter has
    # none) and compare with the gradients train_batch accumulates
    # at lr=0.0. The encoder check is the one that matters: its
    # gradient must run the same averaged filter backwards, and a
    # missing 1 / (K + 1) or a last-power-only backward fails it.
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    g.set_features(3, [0.1, 0.4])
    labels = {0: 1, 1: 0, 2: 1, 3: 0}
    model = S2GC(2, hidden_dim=3, K=3, lr=0.0, seed=5)
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
    for j in range(model.hidden_dim):
        check(model.gw_out[j],
              lambda j=j: model.w_out[j],
              lambda v, j=j: model.w_out.__setitem__(j, v))
    check(model.gb_out, lambda: model.b_out,
          lambda v: setattr(model, "b_out", v))


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------


def test_s2gc_checkpoint_round_trip():
    model = S2GC(6, hidden_dim=8, K=3, seed=1)
    state = model.state_dict()
    model2 = S2GC(6, hidden_dim=8, K=3, seed=99)
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


def test_s2gc_trains_to_high_auc_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    model, _ = train_s2gc(g, train_ids, val_ids, hidden_dim=16,
                          K=5, lr=0.5, batch_size=64, epochs=100,
                          patience=25, seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    y_true = [g.labels[v] for v in val_ids]
    assert evaluate_all(y_true, scores)["auc"] >= 0.90


def test_s2gc_deeper_average_beats_shallow_on_dense_ring():
    # The mechanism claim: on this graph the shallow average
    # (K = 2, AUC 0.870 in the headline run) trails the deeper
    # one (K = 5, AUC 0.935), because the ring's pocket is wider
    # than two hops. Pin the ordering, not the exact numbers.
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    y_true = [g.labels[v] for v in val_ids]
    aucs = {}
    for K in (2, 5):
        model, _ = train_s2gc(g, train_ids, val_ids, hidden_dim=16,
                              K=K, lr=0.5, batch_size=64, epochs=100,
                              patience=25, seed=7)
        scores = [model.predict(g, [v])[v] for v in val_ids]
        aucs[K] = evaluate_all(y_true, scores)["auc"]
    assert aucs[5] > aucs[2]

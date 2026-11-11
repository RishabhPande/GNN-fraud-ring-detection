"""Tests for BernNet: hand-computed Bernstein terms, partition of
unity, filter-coefficient identities, numeric gradient checks,
checkpointing, and training dynamics on the dense ring."""

import math

from gnn_fraud import (
    BernNet,
    Graph,
    GraphGenerator,
    bern_theta_init,
    bernstein_filter_coeffs,
    bernstein_terms,
    evaluate_all,
    train_bern,
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
# Bernstein terms: hand-computable cases
# ---------------------------------------------------------------------------
# On the two-node graph every degree is 1, so S swaps the two values:
# S [1, 2] = [2, 1]. Hence P = (I + S) / 2 maps [1, 2] to [1.5, 1.5]
# and M = (I - S) / 2 maps it to [-0.5, 0.5].


def test_bernstein_terms_k1_hand_computed():
    g = _two_node_graph()
    terms = bernstein_terms(g, [[1.0], [2.0]], 1)
    assert len(terms) == 2
    approx(terms[0][0][0], 1.5)   # B_0 = P h
    approx(terms[0][1][0], 1.5)
    approx(terms[1][0][0], -0.5)  # B_1 = M h
    approx(terms[1][1][0], 0.5)


def test_bernstein_terms_k2_hand_computed():
    g = _two_node_graph()
    terms = bernstein_terms(g, [[1.0], [2.0]], 2)
    assert len(terms) == 3
    # B_0 = P^2 h: P h = [1.5, 1.5] is a fixed point of P.
    approx(terms[0][0][0], 1.5)
    approx(terms[0][1][0], 1.5)
    # B_1 = 2 M P h: P h is constant across nodes, and M kills
    # constants on this graph (M [c, c] = [0, 0]).
    approx(terms[1][0][0], 0.0)
    approx(terms[1][1][0], 0.0)
    # B_2 = M^2 h: M h = [-0.5, 0.5], S swaps it to [0.5, -0.5],
    # so M (M h) = ([-0.5, 0.5] - [0.5, -0.5]) / 2 = [-0.5, 0.5].
    approx(terms[2][0][0], -0.5)
    approx(terms[2][1][0], 0.5)


def test_bernstein_terms_partition_of_unity():
    # sum_k B_k = (M + P)^K = I for every K, so the terms of any
    # input sum back to the input exactly.
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(0, 3)
    h0 = [[0.5, -1.0], [1.5, 0.25], [-0.75, 2.0], [0.1, 0.4]]
    for K in (1, 2, 3, 5):
        terms = bernstein_terms(g, h0, K)
        assert len(terms) == K + 1
        for v in range(4):
            for j in range(2):
                approx(sum(terms[k][v][j] for k in range(K + 1)),
                       h0[v][j], tol=1e-9)


def test_bernstein_terms_k0_is_identity_copy():
    g = _two_node_graph()
    h0 = [[1.0], [2.0]]
    terms = bernstein_terms(g, h0, 0)
    assert len(terms) == 1
    approx(terms[0][0][0], 1.0)
    terms[0][0][0] = 99.0  # mutating the result must not touch h0
    approx(h0[0][0], 1.0)


def test_bernstein_terms_negative_k_rejected():
    g = _two_node_graph()
    try:
        bernstein_terms(g, [[1.0], [2.0]], -1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for K=-1")


def test_bernstein_terms_isolated_node():
    # An isolated node has an empty S row: P and M both halve it, so
    # B_k picks up C(K, k) (1/2)^K of its value at every k and the
    # terms still sum to the input.
    g = Graph(3, 1)
    g.add_edge(0, 1)
    g.set_features(0, [1.0])
    g.set_features(1, [2.0])
    g.set_features(2, [4.0])
    terms = bernstein_terms(g, [[1.0], [2.0], [4.0]], 2)
    for k, c in enumerate((1, 2, 1)):
        approx(terms[k][2][0], c * 0.25 * 4.0)
    approx(sum(terms[k][2][0] for k in range(3)), 4.0)


# ---------------------------------------------------------------------------
# filter coefficients and init
# ---------------------------------------------------------------------------


def test_bern_filter_coeffs_all_ones_is_identity():
    coeffs = bernstein_filter_coeffs([1.0, 1.0, 1.0, 1.0, 1.0])
    approx(coeffs[0], 1.0)
    for c in coeffs[1:]:
        approx(c, 0.0)


def test_bern_filter_coeffs_linear_init_is_half_smoothing():
    # theta_k = 1 - k/K is the Bernstein form of g(lambda) =
    # 1 - lambda/2 with lambda the Laplacian eigenvalue 1 - s, so
    # the operator is I - (I - S)/2 = (I + S)/2: coeffs [0.5, 0.5, 0...].
    coeffs = bernstein_filter_coeffs(bern_theta_init(4))
    approx(coeffs[0], 0.5)
    approx(coeffs[1], 0.5)
    for c in coeffs[2:]:
        approx(c, 0.0)


def test_bern_theta_init_values():
    assert bern_theta_init(0) == [1.0]
    theta = bern_theta_init(4)
    assert len(theta) == 5
    for k, want in enumerate((1.0, 0.75, 0.5, 0.25, 0.0)):
        approx(theta[k], want)


def test_bern_bad_params_rejected():
    for kwargs in ({"K": -1}, {"hidden_dim": 0}):
        try:
            BernNet(6, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


# ---------------------------------------------------------------------------
# model: numeric gradient check + checkpointing
# ---------------------------------------------------------------------------


def test_bern_model_gradient_check():
    # Encoder + Bernstein filter + head: perturb every parameter
    # (W1, b1, theta, w_out, b_out) and compare against the gradients
    # train_batch accumulates (lr=0.0, so the step moves nothing).
    # This is the check that validates the polynomial-transpose
    # encoder backward, not just the theta gradient.
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    g.set_features(3, [0.1, 0.4])
    labels = {0: 1, 1: 0, 2: 1, 3: 0}
    model = BernNet(2, hidden_dim=4, K=3, lr=0.0, seed=5)

    def loss_fn():
        probs, _ = model.forward(g, [0, 1, 2, 3])
        total = 0.0
        for v in [0, 1, 2, 3]:
            p = min(max(probs[v], 1e-7), 1.0 - 1e-7)
            y = labels[v]
            total += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
        return total / 4

    model.train_batch(g, [0, 1, 2, 3], labels)
    eps = 1e-6

    def check(analytic, get, set_):
        orig = get()
        set_(orig + eps)
        plus = loss_fn()
        set_(orig - eps)
        minus = loss_fn()
        set_(orig)
        approx(analytic, (plus - minus) / (2 * eps), tol=1e-5)

    for i in range(4):
        for j in range(2):
            check(model.gW1[i][j],
                  lambda i=i, j=j: model.W1[i][j],
                  lambda v, i=i, j=j: model.W1[i].__setitem__(j, v))
        check(model.gb1[i],
              lambda i=i: model.b1[i],
              lambda v, i=i: model.b1.__setitem__(i, v))
        check(model.gw_out[i],
              lambda i=i: model.w_out[i],
              lambda v, i=i: model.w_out.__setitem__(i, v))
    for k in range(4):
        check(model.gtheta[k],
              lambda k=k: model.theta[k],
              lambda v, k=k: model.theta.__setitem__(k, v))
    check(model.gb_out, lambda: model.b_out,
          lambda v: setattr(model, "b_out", v))


def test_bern_checkpoint_round_trip():
    model = BernNet(6, hidden_dim=8, K=4, seed=1)
    state = model.state_dict()
    model2 = BernNet(6, hidden_dim=8, K=4, seed=99)
    model2.load_state_dict(state)
    g = Graph(2, 6)
    g.add_edge(0, 1)
    g.set_features(0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    g.set_features(1, [0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    p1 = model.predict(g, [0, 1])
    p2 = model2.predict(g, [0, 1])
    approx(p1[0], p2[0])
    approx(p1[1], p2[1])


def test_bern_k0_matches_hand_computed_mlp():
    # With K=0 the filter is theta_0 * I with theta_0 = 1, so the
    # model is exactly its encoder MLP plus the linear head.
    g = _two_node_graph()
    model = BernNet(1, hidden_dim=2, K=0, seed=0)
    model.W1 = [[2.0], [-1.0]]
    model.b1 = [0.5, 0.0]
    model.w_out = [1.0, -2.0]
    model.b_out = 0.25
    probs = model.predict(g, [0, 1])
    # Node 0: pre = [2.5, -1] -> h = [2.5, 0]; logit = 2.75.
    approx(probs[0], 1.0 / (1.0 + math.exp(-2.75)))
    # Node 1: pre = [4.5, -2] -> h = [4.5, 0]; logit = 4.75.
    approx(probs[1], 1.0 / (1.0 + math.exp(-4.75)))


# ---------------------------------------------------------------------------
# training dynamics
# ---------------------------------------------------------------------------


def test_bern_trains_to_high_auc_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    model, _ = train_bern(g, train_ids, val_ids, K=4, hidden_dim=16,
                          lr=0.5, batch_size=64, epochs=100,
                          patience=25, seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    y_true = [g.labels[v] for v in val_ids]
    assert evaluate_all(y_true, scores)["auc"] >= 0.90


def test_bern_theta_moves_from_linear_init():
    # The learned coefficients are the finding, so pin that training
    # actually moves them off the init (a frozen theta would mean the
    # filter gradient is dead).
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    model, _ = train_bern(g, train_ids, val_ids, K=4, hidden_dim=16,
                          lr=0.5, batch_size=64, epochs=100,
                          patience=25, seed=7)
    init = bern_theta_init(4)
    moved = max(abs(a - b) for a, b in zip(model.theta, init))
    assert moved > 0.1

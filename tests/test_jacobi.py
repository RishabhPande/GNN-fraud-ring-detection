"""Tests for JacobiConv: hand-computed Jacobi polynomials (the
Legendre and Chebyshev special cases), hand-computed filter terms
on a two-node graph, the theta init's exact (I + S) / 2 filter,
the filter-coefficient expansion's consistency with the terms,
numeric gradient checks end to end (including theta), checkpointing,
bad-parameter rejection, and training dynamics on the dense ring."""

import math
import random

from gnn_fraud import (
    Graph,
    GraphGenerator,
    JacobiConv,
    evaluate_all,
    jacobi_filter_coeffs,
    jacobi_polys,
    jacobi_terms,
    jacobi_theta_init,
    train_jacobi,
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
# hand-computed polynomials and terms
# ---------------------------------------------------------------------------


def test_jacobi_polys_legendre_hand_computed():
    # a = b = 0 is the Legendre subfamily: P_0 = 1, P_1 = x,
    # P_2 = (3x^2 - 1) / 2, P_3 = (5x^3 - 3x) / 2.
    polys = jacobi_polys(3, 0.0, 0.0)
    assert polys[0] == [1.0]
    assert polys[1] == [0.0, 1.0]
    for got, want in zip(polys[2], [-0.5, 0.0, 1.5]):
        approx(got, want)
    for got, want in zip(polys[3], [0.0, -1.5, 0.0, 2.5]):
        approx(got, want)


def test_jacobi_polys_chebyshev_corner_are_scaled_T():
    # a = b = -0.5: each P_k is a shrinking multiple of the
    # Chebyshev polynomial T_k -- 1/2 for k = 1, 3/8 for k = 2,
    # 5/16 for k = 3. This is why the corner's terms stay bounded
    # and the filter trains at lr 0.5 (see the module docstring).
    polys = jacobi_polys(3, -0.5, -0.5)
    for got, want in zip(polys[1], [0.0, 0.5]):          # (1/2) T_1
        approx(got, want)
    for got, want in zip(polys[2], [-0.375, 0.0, 0.75]):  # (3/8) T_2
        approx(got, want)
    for got, want in zip(polys[3], [0.0, -0.9375, 0.0, 1.25]):  # (5/16) T_3
        approx(got, want)


def test_jacobi_terms_two_node_hand_computed():
    g = _two_node_graph()
    # Degrees are 1 and 1, so S swaps the two vectors: S h = [3, 1].
    terms = jacobi_terms(g, [[1.0], [3.0]], 2, 1.0, 1.0)
    approx(terms[0][0][0], 1.0)
    approx(terms[0][1][0], 3.0)
    # P_1 = 2 S at a = b = 1, so P_1(S) h = [6, 2].
    approx(terms[1][0][0], 6.0)
    approx(terms[1][1][0], 2.0)
    # Recurrence at n = 1, a = b = 1: A = 1.875, B = 0, C = 0.75, so
    # P_2(S) h = 1.875 * S [6, 2] - 0.75 * [1, 3] = [3.0, 9.0].
    approx(terms[2][0][0], 3.0)
    approx(terms[2][1][0], 9.0)


def test_jacobi_terms_isolated_node_sees_constant_term_only():
    g = Graph(3, 1)
    g.add_edge(0, 1)
    h0 = [[1.0], [3.0], [7.0]]
    # Node 2 is isolated: its S row is zero, so P_1(S) h at node 2
    # is exactly P_1's constant term times h -- (a - b) / 2 = 1 at
    # a = 2, b = 0.
    terms = jacobi_terms(g, h0, 1, 2.0, 0.0)
    approx(terms[1][2][0], 7.0)
    # At the symmetric corners (a = b) the constant term is zero,
    # so the isolated node's first-order term vanishes exactly.
    terms = jacobi_terms(g, h0, 1, -0.5, -0.5)
    approx(terms[1][2][0], 0.0)


def test_jacobi_terms_K_zero_returns_copy():
    g = _two_node_graph()
    h0 = [[1.0], [3.0]]
    terms = jacobi_terms(g, h0, 0)
    assert terms == [h0]
    terms[0][0][0] = 99.0
    assert h0[0][0] == 1.0


# ---------------------------------------------------------------------------
# init and filter-coefficient expansion
# ---------------------------------------------------------------------------


def test_jacobi_theta_init_is_exact_half_smoothing():
    # For every (a, b), the init's filter is exactly (I + S) / 2:
    # coefficients [0.5, 0.5, 0, ..., 0] in the monomial basis.
    for a, b in [(1.0, 1.0), (0.0, 0.0), (2.0, 0.0), (0.5, 1.5),
                 (-0.5, -0.5)]:
        coeffs = jacobi_filter_coeffs(jacobi_theta_init(4, a, b), a, b)
        approx(coeffs[0], 0.5)
        approx(coeffs[1], 0.5)
        for c in coeffs[2:]:
            approx(c, 0.0)
    assert jacobi_theta_init(0) == [1.0]


def test_jacobi_filter_coeffs_match_terms():
    # The expansion jacobi_filter_coeffs(theta) must reproduce the
    # filtered output built from jacobi_terms directly: apply the
    # monomial coefficients to the powers of S computed by hand
    # here, and compare against sum_k theta_k * terms[k].
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    h0 = [[0.5, -1.0], [1.5, 0.25], [-0.75, 2.0], [0.1, 0.4]]
    a, b, K = 0.5, 1.5, 3
    theta = [0.3, -0.7, 1.1, 0.2]
    terms = jacobi_terms(g, h0, K, a, b)
    coeffs = jacobi_filter_coeffs(theta, a, b)
    # Powers of S by direct normalized adjacency multiplication.
    d = [g.degree(v) for v in range(g.n)]

    def apply_S(H):
        return [[sum(H[u][j] / math.sqrt(d[v] * d[u])
                      for u in g.neighbors(v)) for j in range(2)]
                for v in range(g.n)]

    powers = [h0]
    for _ in range(K):
        powers.append(apply_S(powers[-1]))
    for v in range(g.n):
        for j in range(2):
            via_terms = sum(theta[k] * terms[k][v][j]
                            for k in range(K + 1))
            via_coeffs = sum(coeffs[t] * powers[t][v][j]
                             for t in range(K + 1))
            approx(via_terms, via_coeffs, tol=1e-9)


def test_jacobi_K_zero_model_ignores_graph():
    # K = 0 degrades exactly to an MLP on raw features: the filter
    # is theta_0 * P_0 = the identity, so edges change nothing.
    g = _two_node_graph()
    g2 = Graph(2, 1)  # same features, no edge at all
    g2.set_features(0, [1.0])
    g2.set_features(1, [3.0])
    model = JacobiConv(1, hidden_dim=4, K=0, seed=3)
    p1 = model.predict(g, [0, 1])
    p2 = model.predict(g2, [0, 1])
    approx(p1[0], p2[0])
    approx(p1[1], p2[1])


def test_jacobi_bad_params_rejected():
    for kwargs in ({"K": -1}, {"hidden_dim": 0}, {"a": -1.0},
                   {"b": -1.5}):
        try:
            JacobiConv(6, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")
    for fn in (lambda: jacobi_polys(-1, 0.0, 0.0),
               lambda: jacobi_theta_init(2, -1.0, 0.0),
               lambda: jacobi_terms(_two_node_graph(), [[1.0], [2.0]],
                                    1, 0.0, -2.0)):
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


def test_jacobi_model_gradient_check():
    # Perturb every parameter (encoder, theta, head) and compare
    # with the gradients train_batch accumulates at lr=0.0. Theta
    # is the check that matters most: its gradient is a dot
    # product with the terms, and the encoder gradient runs the
    # filter polynomial backwards, so a wrong expansion in
    # jacobi_filter_coeffs fails here even if the forward is right.
    # Run at a = 0.5, b = 1.5, away from every symmetric corner.
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    g.set_features(3, [0.1, 0.4])
    labels = {0: 1, 1: 0, 2: 1, 3: 0}
    model = JacobiConv(2, hidden_dim=3, K=3, a=0.5, b=1.5,
                       lr=0.0, seed=5)
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
    for k in range(model.K + 1):
        check(model.gtheta[k],
              lambda k=k: model.theta[k],
              lambda v, k=k: model.theta.__setitem__(k, v))
    for j in range(model.hidden_dim):
        check(model.gw_out[j],
              lambda j=j: model.w_out[j],
              lambda v, j=j: model.w_out.__setitem__(j, v))
    check(model.gb_out, lambda: model.b_out,
          lambda v: setattr(model, "b_out", v))


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------


def test_jacobi_checkpoint_round_trip():
    model = JacobiConv(6, hidden_dim=8, K=3, seed=1)
    state = model.state_dict()
    model2 = JacobiConv(6, hidden_dim=8, K=3, seed=99)
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


def test_jacobi_trains_to_high_auc_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    model, _ = train_jacobi(g, train_ids, val_ids, hidden_dim=16,
                            K=4, lr=0.5, batch_size=64, epochs=150,
                            patience=40, seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    y_true = [g.labels[v] for v in val_ids]
    assert evaluate_all(y_true, scores)["auc"] >= 0.90


def test_jacobi_theta_moves_off_init():
    # The learned filter is the finding, so pin that training
    # actually moves theta: a dead theta branch would leave the
    # model at its (I + S) / 2 init filter throughout.
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    init = jacobi_theta_init(4)
    model, _ = train_jacobi(g, train_ids, val_ids, hidden_dim=16,
                            K=4, lr=0.5, batch_size=64, epochs=150,
                            patience=40, seed=7)
    moved = sum(abs(t - i) for t, i in zip(model.theta, init))
    assert moved > 0.5

"""Tests for ChebNetII: hand-computed Chebyshev nodes and terms,
the interpolation map (values at nodes reproduce exactly, a linear
filter's coefficients come back exact), the gamma init's exact
(I + S) / 2 filter, the filter-coefficient expansion's consistency
with the terms, numeric gradient checks end to end (including
gamma, through the interpolation transpose), checkpointing,
bad-parameter rejection, and training dynamics on the dense ring."""

import math

from gnn_fraud import (
    ChebNetII,
    Graph,
    GraphGenerator,
    chebii_cheb_coeffs,
    chebii_filter_coeffs,
    chebii_gamma_init,
    chebii_nodes,
    chebii_terms,
    evaluate_all,
    train_chebii,
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
# nodes, interpolation, and hand-computed terms
# ---------------------------------------------------------------------------


def test_chebii_nodes_hand_computed():
    # K = 3: nodes cos(pi/8), cos(3pi/8), cos(5pi/8), cos(7pi/8).
    nodes = chebii_nodes(3)
    for got, j in zip(nodes, range(4)):
        approx(got, math.cos((j + 0.5) * math.pi / 4))
    approx(nodes[0], 0.9238795325112867)
    approx(nodes[3], -0.9238795325112867)
    # Nodes are symmetric: x_j = -x_{K-j}.
    for j in range(4):
        approx(nodes[j], -nodes[3 - j])
    assert chebii_nodes(0) == [0.0] or abs(chebii_nodes(0)[0]) < 1e-15


def test_chebii_interpolation_reproduces_values_at_nodes():
    # The defining property: the polynomial built from gamma passes
    # through (x_j, gamma_j) at every node. Evaluate sum_k c_k T_k
    # at each node directly and compare.
    gamma = [0.3, -0.7, 1.1, 0.2, -0.4]
    K = len(gamma) - 1
    coeffs = chebii_cheb_coeffs(gamma)
    nodes = chebii_nodes(K)
    for j, x in enumerate(nodes):
        # T_k(x) by the scalar recurrence.
        t_prev, t_cur = 1.0, x
        val = coeffs[0] * t_prev
        if K >= 1:
            val += coeffs[1] * t_cur
        for k in range(2, K + 1):
            t_prev, t_cur = t_cur, 2.0 * x * t_cur - t_prev
            val += coeffs[k] * t_cur
        approx(val, gamma[j], tol=1e-9)


def test_chebii_interpolation_of_linear_filter_is_exact():
    # g(x) = (1 + x) / 2 has Chebyshev coefficients [0.5, 0.5, 0...]:
    # interpolating its values at any K >= 1 nodes recovers them.
    for K in (1, 2, 4, 7):
        gamma = [(1.0 + x) / 2.0 for x in chebii_nodes(K)]
        coeffs = chebii_cheb_coeffs(gamma)
        approx(coeffs[0], 0.5)
        approx(coeffs[1], 0.5)
        for c in coeffs[2:]:
            approx(c, 0.0)


def test_chebii_terms_two_node_hand_computed():
    g = _two_node_graph()
    # Degrees are 1 and 1, so S swaps the two vectors: S h = [3, 1].
    terms = chebii_terms(g, [[1.0], [3.0]], 3)
    approx(terms[0][0][0], 1.0)
    approx(terms[0][1][0], 3.0)
    approx(terms[1][0][0], 3.0)
    approx(terms[1][1][0], 1.0)
    # T_2 = 2 S^2 - I and S^2 = I here, so T_2(S) h = h.
    approx(terms[2][0][0], 1.0)
    approx(terms[2][1][0], 3.0)
    # T_3 = 4 S^3 - 3 S = S here (S^3 = S), so T_3(S) h = [3, 1].
    approx(terms[3][0][0], 3.0)
    approx(terms[3][1][0], 1.0)


def test_chebii_terms_isolated_node():
    g = Graph(3, 1)
    g.add_edge(0, 1)
    h0 = [[1.0], [3.0], [7.0]]
    terms = chebii_terms(g, h0, 2)
    # Node 2 is isolated: S row is zero, so T_1 term vanishes and
    # T_2 = 2 S^2 - I leaves exactly -h.
    approx(terms[1][2][0], 0.0)
    approx(terms[2][2][0], -7.0)


def test_chebii_terms_K_zero_returns_copy():
    g = _two_node_graph()
    h0 = [[1.0], [3.0]]
    terms = chebii_terms(g, h0, 0)
    assert terms == [h0]
    terms[0][0][0] = 99.0
    assert h0[0][0] == 1.0


# ---------------------------------------------------------------------------
# init and filter-coefficient expansion
# ---------------------------------------------------------------------------


def test_chebii_gamma_init_is_exact_half_smoothing():
    for K in (1, 2, 4, 7):
        coeffs = chebii_filter_coeffs(chebii_gamma_init(K))
        approx(coeffs[0], 0.5)
        approx(coeffs[1], 0.5)
        for c in coeffs[2:]:
            approx(c, 0.0)
    assert chebii_gamma_init(0) == [1.0]


def test_chebii_filter_coeffs_match_terms():
    # chebii_filter_coeffs(gamma) expanded to monomials must
    # reproduce the filtered output built from the Chebyshev terms
    # and the interpolated coefficients directly.
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    h0 = [[0.5, -1.0], [1.5, 0.25], [-0.75, 2.0], [0.1, 0.4]]
    K = 3
    gamma = [0.3, -0.7, 1.1, 0.2]
    terms = chebii_terms(g, h0, K)
    cheb = chebii_cheb_coeffs(gamma)
    coeffs = chebii_filter_coeffs(gamma)
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
            via_terms = sum(cheb[k] * terms[k][v][j]
                            for k in range(K + 1))
            via_coeffs = sum(coeffs[t] * powers[t][v][j]
                             for t in range(K + 1))
            approx(via_terms, via_coeffs, tol=1e-9)


def test_chebii_K_zero_model_ignores_graph():
    # K = 0: the filter is gamma_0 * T_0 = a scalar times the
    # identity, so edges change nothing.
    g = _two_node_graph()
    g2 = Graph(2, 1)
    g2.set_features(0, [1.0])
    g2.set_features(1, [3.0])
    model = ChebNetII(1, hidden_dim=4, K=0, seed=3)
    p1 = model.predict(g, [0, 1])
    p2 = model.predict(g2, [0, 1])
    approx(p1[0], p2[0])
    approx(p1[1], p2[1])


def test_chebii_bad_params_rejected():
    for kwargs in ({"K": -1}, {"hidden_dim": 0}):
        try:
            ChebNetII(6, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")
    for fn in (lambda: chebii_nodes(-1),
               lambda: chebii_gamma_init(-2),
               lambda: chebii_terms(_two_node_graph(), [[1.0], [2.0]], -1),
               lambda: chebii_cheb_coeffs([])):
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


def test_chebii_model_gradient_check():
    # Perturb every parameter (encoder, gamma, head) and compare
    # with the gradients train_batch accumulates at lr=0.0. Gamma
    # is the check that matters most: its gradient runs through the
    # transpose of the interpolation map, and the encoder gradient
    # runs the expanded filter polynomial backwards, so a wrong
    # factor in either map fails here even if the forward is right.
    # Gamma is set off its init so every interpolation row is live.
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    g.set_features(3, [0.1, 0.4])
    labels = {0: 1, 1: 0, 2: 1, 3: 0}
    model = ChebNetII(2, hidden_dim=3, K=3, lr=0.0, seed=5)
    model.gamma = [0.3, -0.7, 1.1, 0.2]
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
        check(model.ggamma[k],
              lambda k=k: model.gamma[k],
              lambda v, k=k: model.gamma.__setitem__(k, v))
    for j in range(model.hidden_dim):
        check(model.gw_out[j],
              lambda j=j: model.w_out[j],
              lambda v, j=j: model.w_out.__setitem__(j, v))
    check(model.gb_out, lambda: model.b_out,
          lambda v: setattr(model, "b_out", v))


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------


def test_chebii_checkpoint_round_trip():
    model = ChebNetII(6, hidden_dim=8, K=3, seed=1)
    state = model.state_dict()
    model2 = ChebNetII(6, hidden_dim=8, K=3, seed=99)
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


def test_chebii_trains_to_high_auc_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    model, _ = train_chebii(g, train_ids, val_ids, hidden_dim=16,
                            K=4, lr=0.5, batch_size=64, epochs=100,
                            patience=25, seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    y_true = [g.labels[v] for v in val_ids]
    assert evaluate_all(y_true, scores)["auc"] >= 0.90


def test_chebii_gamma_moves_off_init():
    # The learned values are the finding, so pin that training
    # actually moves gamma: a dead gamma branch would leave the
    # model at its (I + S) / 2 init filter throughout.
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    init = chebii_gamma_init(4)
    model, _ = train_chebii(g, train_ids, val_ids, hidden_dim=16,
                            K=4, lr=0.5, batch_size=64, epochs=100,
                            patience=25, seed=7)
    moved = sum(abs(t - i) for t, i in zip(model.gamma, init))
    assert moved > 0.5

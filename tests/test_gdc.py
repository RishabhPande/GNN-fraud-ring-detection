"""Tests for GDC: hand-computed heat-kernel coefficients and
propagation powers, the Poisson mass identities, numeric gradient
checks end to end (including the closed-form log-time gradient),
checkpointing, bad-parameter rejection, and training dynamics on
the dense ring."""

import math

from gnn_fraud import (
    Graph,
    GraphDiffusion,
    GraphGenerator,
    diffusion_powers,
    evaluate_all,
    heat_kernel_coeffs,
    train_gdc,
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
# coefficients and hand-computed powers
# ---------------------------------------------------------------------------


def test_heat_kernel_coeffs_hand_computed():
    # t = 1: theta_k = e^-1 / k!.
    coeffs = heat_kernel_coeffs(1.0, 3)
    e = math.exp(-1.0)
    approx(coeffs[0], e)
    approx(coeffs[1], e)
    approx(coeffs[2], e / 2.0)
    approx(coeffs[3], e / 6.0)
    # t = 2, k = 2: e^-2 * 4 / 2 = 2 e^-2.
    approx(heat_kernel_coeffs(2.0, 2)[2], 2.0 * math.exp(-2.0))


def test_heat_kernel_coeffs_mass_and_peak():
    # The untruncated Poisson weights sum to 1, so a long enough
    # truncation is within tail mass of 1, and the peak sits at
    # k near t (the mode of the Poisson distribution).
    coeffs = heat_kernel_coeffs(2.0, 30)
    approx(sum(coeffs), 1.0, tol=1e-9)
    assert max(range(len(coeffs)), key=lambda k: coeffs[k]) in (1, 2)
    # Truncation only ever loses mass, never gains it.
    assert sum(heat_kernel_coeffs(2.0, 3)) < 1.0


def test_diffusion_powers_two_node_hand_computed():
    g = _two_node_graph()
    # Degrees are 1 and 1, so S swaps the two vectors and S^2 = I.
    powers = diffusion_powers(g, [[1.0], [3.0]], 3)
    approx(powers[0][0][0], 1.0)
    approx(powers[1][0][0], 3.0)
    approx(powers[1][1][0], 1.0)
    approx(powers[2][0][0], 1.0)
    approx(powers[2][1][0], 3.0)
    approx(powers[3][0][0], 3.0)


def test_diffusion_powers_isolated_node():
    g = Graph(3, 1)
    g.add_edge(0, 1)
    powers = diffusion_powers(g, [[1.0], [3.0], [7.0]], 2)
    # Node 2 is isolated: every positive power vanishes on it, so
    # its diffused state is exactly theta_0 * h = e^-t * h.
    approx(powers[1][2][0], 0.0)
    approx(powers[2][2][0], 0.0)


def test_diffusion_powers_K_zero_returns_copy():
    g = _two_node_graph()
    h0 = [[1.0], [3.0]]
    powers = diffusion_powers(g, h0, 0)
    assert powers == [h0]
    powers[0][0][0] = 99.0
    assert h0[0][0] == 1.0


def test_gdc_K_zero_model_ignores_graph():
    # K = 0: Z = theta_0 * h0, a positive scalar times the encoder
    # state, so edges change nothing.
    g = _two_node_graph()
    g2 = Graph(2, 1)
    g2.set_features(0, [1.0])
    g2.set_features(1, [3.0])
    model = GraphDiffusion(1, hidden_dim=4, K=0, seed=3)
    p1 = model.predict(g, [0, 1])
    p2 = model.predict(g2, [0, 1])
    approx(p1[0], p2[0])
    approx(p1[1], p2[1])


def test_gdc_bad_params_rejected():
    for kwargs in ({"K": -1}, {"hidden_dim": 0}, {"t_init": 0.0},
                   {"t_init": -1.0}):
        try:
            GraphDiffusion(6, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")
    for fn in (lambda: heat_kernel_coeffs(0.0, 2),
               lambda: heat_kernel_coeffs(1.0, -1),
               lambda: diffusion_powers(_two_node_graph(),
                                         [[1.0], [2.0]], -1)):
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


def test_gdc_model_gradient_check():
    # Perturb every parameter (encoder, log_t, head) and compare
    # with the gradients train_batch accumulates at lr=0.0. log_t
    # is the check that matters most: its gradient chains the
    # coefficient gradient through theta_k (k - t), and a wrong
    # sign or factor there fails even if the forward is right.
    # log_t is set off its init (t = 1.3) so the chain is live at
    # a non-round time.
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    g.set_features(3, [0.1, 0.4])
    labels = {0: 1, 1: 0, 2: 1, 3: 0}
    model = GraphDiffusion(2, hidden_dim=3, K=3, lr=0.0, seed=5)
    model.log_t = math.log(1.3)
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
    check(model.glog_t, lambda: model.log_t,
          lambda v: setattr(model, "log_t", v))
    for j in range(model.hidden_dim):
        check(model.gw_out[j],
              lambda j=j: model.w_out[j],
              lambda v, j=j: model.w_out.__setitem__(j, v))
    check(model.gb_out, lambda: model.b_out,
          lambda v: setattr(model, "b_out", v))


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------


def test_gdc_checkpoint_round_trip():
    model = GraphDiffusion(6, hidden_dim=8, K=3, seed=1)
    state = model.state_dict()
    model2 = GraphDiffusion(6, hidden_dim=8, K=3, seed=99)
    model2.load_state_dict(state)
    g = Graph(2, 6)
    g.add_edge(0, 1)
    g.set_features(0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    g.set_features(1, [0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    p1 = model.predict(g, [0, 1])
    p2 = model2.predict(g, [0, 1])
    approx(p1[0], p2[0])
    approx(p1[1], p2[1])
    approx(model.diffusion_time(), model2.diffusion_time())


# ---------------------------------------------------------------------------
# training dynamics
# ---------------------------------------------------------------------------


def test_gdc_trains_to_high_auc_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    model, _ = train_gdc(g, train_ids, val_ids, hidden_dim=16,
                         K=6, lr=0.5, batch_size=64, epochs=100,
                         patience=25, seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    y_true = [g.labels[v] for v in val_ids]
    assert evaluate_all(y_true, scores)["auc"] >= 0.90


def test_gdc_diffusion_time_moves_off_init():
    # The learned time is the finding, so pin that training
    # actually moves it: a dead log_t branch would leave the
    # model at its t = 2 init throughout.
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    model, _ = train_gdc(g, train_ids, val_ids, hidden_dim=16,
                         K=6, lr=0.5, batch_size=64, epochs=100,
                         patience=25, seed=7)
    assert abs(model.log_t - math.log(2.0)) > 0.05

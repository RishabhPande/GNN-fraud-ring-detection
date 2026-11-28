"""Tests for PairNorm: hand-computed centering and rescaling,
the two invariants (zero mean, exact total pairwise distance),
scale invariance, the collapsed-input edge case, numeric
gradient checks end to end, checkpointing, bad-parameter
rejection, and training dynamics on the dense ring, including
the deep-stack oversmoothing comparison."""

import math

from gnn_fraud import (
    Graph,
    GraphGenerator,
    GraphPairNormGCN,
    evaluate_all,
    pairnorm,
    total_pairwise_sq_distance,
    train_pairnorm,
    train_val_split,
)
from gnn_fraud.pairnorm import pairnorm_backward, pairnorm_forward


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


# ---------------------------------------------------------------------------
# hand-computed map and invariants
# ---------------------------------------------------------------------------


def test_pairnorm_hand_computed():
    # H = [1, 3, 5] in one dimension: mean 3, centered
    # [-2, 0, 2], norm sqrt(8), alpha = sqrt(3) / sqrt(8).
    Y = pairnorm([[1.0], [3.0], [5.0]])
    a = math.sqrt(3.0 / 8.0)
    approx(Y[0][0], -2.0 * a)
    approx(Y[1][0], 0.0)
    approx(Y[2][0], 2.0 * a)


def test_pairnorm_zero_mean_and_exact_total_distance():
    H = [[0.5, -1.0], [1.5, 0.25], [-0.75, 2.0], [0.1, 0.4], [2.0, -0.3]]
    for scale in (1.0, 2.5):
        Y = pairnorm(H, scale=scale)
        for j in range(2):
            approx(sum(row[j] for row in Y), 0.0, tol=1e-9)
        approx(sum(x * x for row in Y for x in row), scale * scale * 5)
        approx(total_pairwise_sq_distance(Y), scale * scale * 25)


def test_pairnorm_scale_invariance():
    H = [[0.5, -1.0], [1.5, 0.25], [-0.75, 2.0]]
    Y1 = pairnorm(H)
    Y2 = pairnorm([[7.0 * x for x in row] for row in H])
    for r1, r2 in zip(Y1, Y2):
        for a, b in zip(r1, r2):
            approx(a, b)


def test_pairnorm_collapsed_input_maps_to_zero():
    # Every row identical: the centered input is exactly zero,
    # the fully oversmoothed state. Output is zeros and the
    # backward pass is zeros (PairNorm prevents collapse layer
    # by layer; it cannot undo an exact one).
    Y, cache = pairnorm_forward([[2.0, 3.0], [2.0, 3.0], [2.0, 3.0]])
    assert Y == [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    d = pairnorm_backward([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], cache)
    assert d == [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]


def test_pairnorm_backward_numeric():
    # Central differences on a random small input through a
    # fixed linear readout, against pairnorm_backward directly.
    rng_rows = [[0.3, -0.8], [1.2, 0.5], [-0.4, 1.1], [0.9, -0.2]]
    w = [0.7, -1.3]

    def loss(H):
        Y, _ = pairnorm_forward(H, scale=1.5)
        return sum(w[j] * Y[v][j] for v in range(4) for j in range(2))

    _, cache = pairnorm_forward(rng_rows, scale=1.5)
    dY = [[w[0], w[1]] for _ in range(4)]
    dH = pairnorm_backward(dY, cache)
    eps = 1e-6
    for v in range(4):
        for j in range(2):
            orig = rng_rows[v][j]
            rng_rows[v][j] = orig + eps
            plus = loss(rng_rows)
            rng_rows[v][j] = orig - eps
            minus = loss(rng_rows)
            rng_rows[v][j] = orig
            approx(dH[v][j], (plus - minus) / (2 * eps), tol=1e-5)


def test_pairnorm_bad_params_rejected():
    for kwargs in ({"hidden_dim": 0}, {"n_layers": 0}, {"scale": 0.0},
                   {"scale": -1.0}):
        try:
            GraphPairNormGCN(6, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")
    for fn in (lambda: pairnorm([[1.0]], scale=0.0),
               lambda: pairnorm([]),
               lambda: pairnorm([[1.0], [1.0, 2.0]])):
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


# ---------------------------------------------------------------------------
# numeric gradient check, end to end
# ---------------------------------------------------------------------------


def _loss(model, g, labels):
    probs, _ = model.forward(g, [0, 1, 2, 3])
    total = 0.0
    for v in [0, 1, 2, 3]:
        p = min(max(probs[v], 1e-7), 1.0 - 1e-7)
        y = labels[v]
        total += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
    return total / 4


def test_pairnorm_model_gradient_check():
    # Perturb every parameter (both GCN layers and the head) and
    # compare with the gradients train_batch accumulates at
    # lr=0.0. The PairNorm backward sits between the head and
    # every layer, so a missing radial term or a missing
    # re-centering fails this check.
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    g.set_features(3, [0.1, 0.4])
    labels = {0: 1, 1: 0, 2: 1, 3: 0}
    model = GraphPairNormGCN(2, hidden_dim=3, n_layers=2, lr=0.0, seed=5)
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

    for layer in model.layers:
        for i in range(len(layer.W)):
            for j in range(len(layer.W[i])):
                check(layer.gW[i][j],
                      lambda layer=layer, i=i, j=j: layer.W[i][j],
                      lambda v, layer=layer, i=i, j=j:
                      layer.W[i].__setitem__(j, v))
        for i in range(len(layer.b)):
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


def test_pairnorm_checkpoint_round_trip():
    model = GraphPairNormGCN(6, hidden_dim=8, n_layers=3, seed=1)
    state = model.state_dict()
    model2 = GraphPairNormGCN(6, hidden_dim=8, n_layers=3, seed=99)
    model2.load_state_dict(state)
    g = Graph(2, 6)
    g.add_edge(0, 1)
    g.set_features(0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    g.set_features(1, [0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    p1 = model.predict(g, [0, 1])
    p2 = model2.predict(g, [0, 1])
    approx(p1[0], p2[0])
    approx(p1[1], p2[1])


def test_pairnorm_embeddings_keep_total_distance():
    # After the last PairNorm the embeddings' total pairwise
    # squared distance is exactly scale^2 * n^2, at any depth:
    # the invariant the whole model is built around.
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=60, ring_size=8)
    for depth in (1, 2, 6):
        model = GraphPairNormGCN(g.feature_dim, hidden_dim=8,
                                 n_layers=depth, scale=1.0, seed=3)
        H = model.embeddings(g)
        approx(total_pairwise_sq_distance(H), float(g.n * g.n), tol=1e-6)


# ---------------------------------------------------------------------------
# training dynamics
# ---------------------------------------------------------------------------


def test_pairnorm_trains_to_high_auc_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    model, _ = train_pairnorm(g, train_ids, val_ids, hidden_dim=16,
                              n_layers=2, lr=0.5, batch_size=64,
                              epochs=100, patience=25, seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    y_true = [g.labels[v] for v in val_ids]
    assert evaluate_all(y_true, scores)["auc"] >= 0.90


def test_pairnorm_deep_stack_trains_on_dense_ring():
    # The mechanism claim: an 8-layer stack still trains, because
    # PairNorm resets the embeddings' total pairwise distance
    # after every layer instead of letting the normalized
    # aggregation collapse it toward zero.
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    model, _ = train_pairnorm(g, train_ids, val_ids, hidden_dim=16,
                              n_layers=8, lr=0.5, batch_size=64,
                              epochs=100, patience=25, seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    y_true = [g.labels[v] for v in val_ids]
    assert evaluate_all(y_true, scores)["auc"] >= 0.90

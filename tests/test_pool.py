"""Tests for the max-pool aggregator: hand-computed forward/backward,
numeric gradient checks, edge cases, checkpoint compatibility, and a
training smoke test on the dense ring."""

import copy
import random

from gnn_fraud import (
    GraphGenerator,
    SAGELayer,
    roc_auc,
    train_sage,
    train_val_split,
)


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def _identity_pool_layer():
    # in_dim=2, out_dim=2, all identity weights, zero biases.
    layer = SAGELayer(2, 2, random.Random(0), aggregator="pool")
    layer.W_self = [[1.0, 0.0], [0.0, 1.0]]
    layer.W_agg = [[1.0, 0.0], [0.0, 1.0]]
    layer.W_pool = [[1.0, 0.0], [0.0, 1.0]]
    layer.b_pool = [0.0, 0.0]
    layer.b = [0.0, 0.0]
    return layer


def test_pool_invalid_aggregator_raises():
    try:
        SAGELayer(2, 2, random.Random(0), aggregator="sum")
    except ValueError:
        return
    raise AssertionError("expected ValueError for aggregator='sum'")


def test_pool_default_is_mean():
    layer = SAGELayer(2, 2, random.Random(0))
    assert layer.aggregator == "mean"
    assert layer.W_pool is None
    assert layer.b_pool is None


def test_pool_forward_hand_computed():
    # t_0 = relu([1,2]) = [1,2]; t_1 = relu([3,0.5]) = [3,0.5].
    # agg = [max(1,3), max(2,0.5)] = [3,2]; argmax = [1,0].
    # pre = h_self + agg = [0.5+3, -0.5+2] = [3.5, 1.5].
    layer = _identity_pool_layer()
    out, cache = layer.forward([0.5, -0.5], [[1.0, 2.0], [3.0, 0.5]])
    approx(out[0], 3.5)
    approx(out[1], 1.5)
    agg, t, argmax = cache[2], cache[3], cache[4]
    approx(agg[0], 3.0)
    approx(agg[1], 2.0)
    assert argmax == [1, 0]


def test_pool_backward_hand_computed():
    # d_out = [1,1], pre > 0 -> d_pre = [1,1].
    # dim 0 routes to neighbor 1 ([3,0.5]); dim 1 routes to neighbor 0 ([1,2]).
    layer = _identity_pool_layer()
    _, cache = layer.forward([0.5, -0.5], [[1.0, 2.0], [3.0, 0.5]])
    d_h_self, d_neighs = layer.backward([1.0, 1.0], cache)
    # Shared params see the pooled agg [3,2], same as the mean path would.
    assert layer.gW_self == [[0.5, -0.5], [0.5, -0.5]]
    assert layer.gW_agg == [[3.0, 2.0], [3.0, 2.0]]
    assert layer.gb == [1.0, 1.0]
    approx(d_h_self[0], 1.0)
    approx(d_h_self[1], 1.0)
    # Pool-only params: each dim's gradient goes to its argmax neighbor's
    # raw features; the bias gets the plain per-dim gradient.
    assert layer.gW_pool == [[3.0, 0.5], [1.0, 2.0]]
    assert layer.gb_pool == [1.0, 1.0]
    # Per-neighbor grads route through W_pool (identity here): neighbor 0
    # gets dim 1's gradient, neighbor 1 gets dim 0's.
    assert len(d_neighs) == 2
    approx(d_neighs[0][0], 0.0)
    approx(d_neighs[0][1], 1.0)
    approx(d_neighs[1][0], 1.0)
    approx(d_neighs[1][1], 0.0)


def test_pool_backward_dead_relu_gets_no_gradient():
    # Negative pool weights kill every transform: the max is 0 with a dead
    # ReLU behind it, so nothing flows to W_pool / b_pool / neighbors.
    layer = SAGELayer(2, 2, random.Random(0), aggregator="pool")
    layer.W_self = [[1.0, 0.0], [0.0, 1.0]]
    layer.W_agg = [[1.0, 0.0], [0.0, 1.0]]
    layer.W_pool = [[-1.0, 0.0], [0.0, -1.0]]
    layer.b_pool = [0.0, 0.0]
    layer.b = [0.0, 0.0]
    _, cache = layer.forward([0.5, 0.5], [[1.0, 1.0]])
    d_h_self, d_neighs = layer.backward([1.0, 1.0], cache)
    assert layer.gW_pool == [[0.0, 0.0], [0.0, 0.0]]
    assert layer.gb_pool == [0.0, 0.0]
    assert d_neighs == [[0.0, 0.0]]
    # The self path is unaffected by the dead pool transform.
    approx(d_h_self[0], 1.0)
    approx(d_h_self[1], 1.0)


def test_pool_empty_neighborhood():
    layer = _identity_pool_layer()
    out, cache = layer.forward([0.5, -0.5], [])
    approx(out[0], 0.5)
    approx(out[1], 0.0)
    _, d_neighs = layer.backward([1.0, 1.0], cache)
    assert d_neighs == []
    assert layer.gW_pool == [[0.0, 0.0], [0.0, 0.0]]
    assert layer.gb_pool == [0.0, 0.0]


def _numeric_gradient_check(layer, h_self, h_neighs, tol=1e-4):
    def loss():
        out, _ = layer.forward(h_self, h_neighs)
        return sum(out)

    layer.zero_grad()
    _, cache = layer.forward(h_self, h_neighs)
    layer.backward([1.0] * layer.out_dim, cache)
    params = [
        ("W_self", layer.W_self, layer.gW_self),
        ("W_agg", layer.W_agg, layer.gW_agg),
        ("b", layer.b, [[b] for b in layer.gb]),
    ]
    if layer.aggregator == "pool":
        params += [
            ("W_pool", layer.W_pool, layer.gW_pool),
            ("b_pool", layer.b_pool, [[b] for b in layer.gb_pool]),
        ]
    eps = 1e-5
    worst = 0.0
    for name, mat, gmat in params:
        for i in range(len(mat)):
            is_vec = isinstance(mat[i], float)
            cols = 1 if is_vec else len(mat[i])
            for j in range(cols):
                if is_vec:
                    old, ana = mat[i], gmat[i][0]

                    def get(v, m=mat, i=i):
                        m[i] = v
                else:
                    old, ana = mat[i][j], gmat[i][j]

                    def get(v, m=mat, i=i, j=j):
                        m[i][j] = v
                get(old + eps)
                lp = loss()
                get(old - eps)
                lm = loss()
                get(old)
                num = (lp - lm) / (2 * eps)
                rel = abs(num - ana) / max(1.0, abs(num), abs(ana))
                worst = max(worst, rel)
                assert rel < tol, (name, i, j, num, ana, rel)
    return worst


def test_pool_gradient_check():
    # Random init (seed 3): the check passes only if the argmax routing,
    # the ReLU masking, and every param gradient are exactly right.
    layer = SAGELayer(3, 2, random.Random(3), aggregator="pool")
    h_self = [0.4, -0.2, 0.7]
    h_neighs = [[0.1, 0.9, -0.3], [0.8, 0.2, 0.5], [-0.4, 0.6, 0.1]]
    worst = _numeric_gradient_check(layer, h_self, h_neighs)
    assert worst < 1e-4


def test_mean_gradient_check_still_exact():
    layer = SAGELayer(3, 2, random.Random(3))
    h_self = [0.4, -0.2, 0.7]
    h_neighs = [[0.1, 0.9, -0.3], [0.8, 0.2, 0.5]]
    worst = _numeric_gradient_check(layer, h_self, h_neighs)
    assert worst < 1e-4


def test_pool_checkpoint_roundtrip():
    layer = SAGELayer(3, 2, random.Random(5), aggregator="pool")
    fresh = SAGELayer(3, 2, random.Random(6), aggregator="pool")
    fresh.load_state_dict(layer.state_dict())
    h_self = [0.4, -0.2, 0.7]
    h_neighs = [[0.1, 0.9, -0.3], [0.8, 0.2, 0.5]]
    out1, _ = layer.forward(h_self, h_neighs)
    out2, _ = fresh.forward(h_self, h_neighs)
    assert out1 == out2
    assert fresh.W_pool == layer.W_pool
    assert fresh.b_pool == layer.b_pool


def test_legacy_mean_checkpoint_loads_into_pool_layer():
    # A checkpoint saved before the pool aggregator existed has no
    # "aggregator" key: loading it restores a mean layer, so old
    # experiments stay usable.
    mean_layer = SAGELayer(2, 2, random.Random(0))
    state = mean_layer.state_dict()
    del state["aggregator"]  # simulate a checkpoint saved by the old code
    assert "W_pool" not in state
    pool_layer = SAGELayer(2, 2, random.Random(1), aggregator="pool")
    pool_layer.load_state_dict(state)
    assert pool_layer.aggregator == "mean"
    assert pool_layer.W_pool is None
    out, _ = pool_layer.forward([0.5, 0.5], [[1.0, 0.0]])
    assert len(out) == 2


def test_pool_step_updates_pool_params():
    layer = _identity_pool_layer()
    _, cache = layer.forward([0.5, -0.5], [[1.0, 2.0], [3.0, 0.5]])
    layer.backward([1.0, 1.0], cache)
    w_before = copy.deepcopy(layer.W_pool)
    b_before = list(layer.b_pool)
    layer.step(0.1)
    assert layer.W_pool != w_before
    assert layer.b_pool != b_before


def test_pool_trains_on_dense_ring():
    # Seed-7 quickstart protocol: the pool model must actually learn the
    # ring (this guards against a silently dead aggregation path).
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
    model, _ = train_sage(g, train_ids, val_ids, hidden_dim=16, k1=5, k2=5,
                          lr=0.1, batch_size=64, epochs=60, patience=12,
                          seed=7, aggregator="pool")
    y_true = [g.labels[v] for v in val_ids]
    scores = [model.predict(g, [v])[v] for v in val_ids]
    assert roc_auc(y_true, scores) > 0.95

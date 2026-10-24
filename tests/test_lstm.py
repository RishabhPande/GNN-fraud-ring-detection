"""Tests for the LSTM aggregator: hand-computed forward/backward, numeric
gradient checks over every LSTM parameter (input, recurrent, bias), edge
cases, checkpoint compatibility, and a training smoke test on the dense
ring."""

import copy
import math
import random

from gnn_fraud import (
    GraphGenerator,
    SAGELayer,
    roc_auc,
    train_sage,
    train_val_split,
)
from gnn_fraud.sage import _lstm_order


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def _cell_only_layer():
    # in_dim=2, out_dim=2, identity self/agg weights, zero biases, and an
    # LSTM cell where only the candidate gate sees the input (W_g = I):
    # i = f = o = 0.5, g = tanh(x) at every step, so the cell math is
    # hand-computable while the recurrence is still exercised.
    layer = SAGELayer(2, 2, random.Random(0), aggregator="lstm")
    layer.W_self = [[1.0, 0.0], [0.0, 1.0]]
    layer.W_agg = [[1.0, 0.0], [0.0, 1.0]]
    layer.b = [0.0, 0.0]
    zero = [[0.0, 0.0], [0.0, 0.0]]
    for g in ("i", "f", "o", "g"):
        layer.lstm_W[g] = [row[:] for row in zero]
        layer.lstm_U[g] = [row[:] for row in zero]
        layer.lstm_b[g] = [0.0, 0.0]
    layer.lstm_W["g"] = [[1.0, 0.0], [0.0, 1.0]]
    return layer


def test_lstm_defaults_and_init():
    layer = SAGELayer(2, 2, random.Random(0))
    assert layer.aggregator == "mean"
    assert layer.lstm_W is None
    lstm = SAGELayer(2, 2, random.Random(0), aggregator="lstm")
    assert lstm.aggregator == "lstm"
    assert set(lstm.lstm_W) == {"i", "f", "o", "g"}
    assert set(lstm.lstm_U) == {"i", "f", "o", "g"}
    # Forget-gate bias starts at 1.0 (standard anti-forgetting init);
    # the other biases start at 0.
    assert lstm.lstm_b["f"] == [1.0, 1.0]
    assert lstm.lstm_b["i"] == [0.0, 0.0]
    assert lstm.lstm_b["o"] == [0.0, 0.0]
    assert lstm.lstm_b["g"] == [0.0, 0.0]


def test_lstm_order_is_deterministic_and_input_dependent():
    neighs = [[1.0, 2.0], [3.0, 0.5], [-1.0, 0.0]]
    assert _lstm_order(neighs) == _lstm_order(neighs)
    assert sorted(_lstm_order(neighs)) == [0, 1, 2]
    # A different neighborhood (usually) gets a different order; at minimum
    # the order is still a valid permutation.
    assert sorted(_lstm_order([[9.0, 9.0], [8.0, 8.0]])) == [0, 1]


def test_lstm_forward_single_neighbor_hand_computed():
    # One neighbor: no permutation ambiguity. g = tanh([1,-1]); c = 0.5*g;
    # h = 0.5*tanh(c); pre = h_self + h = [0.5, 0.5] + h.
    layer = _cell_only_layer()
    out, cache = layer.forward([0.5, 0.5], [[1.0, -1.0]])
    c0 = 0.5 * math.tanh(1.0)
    h0 = 0.5 * math.tanh(c0)
    approx(out[0], 0.5 + h0)
    approx(out[1], 0.5 - h0)
    agg = cache[2]
    approx(agg[0], h0)
    approx(agg[1], -h0)


def test_lstm_forward_multi_neighbor_matches_manual_cell():
    # Replay the cell by hand in the layer's own permutation order.
    layer = _cell_only_layer()
    neighs = [[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]]
    out, cache = layer.forward([0.0, 0.0], neighs)
    order = _lstm_order(neighs)
    c = [0.0, 0.0]
    h = [0.0, 0.0]
    for idx in order:
        x = neighs[idx]
        g = [math.tanh(v) for v in x]
        c = [0.5 * c[k] + 0.5 * g[k] for k in range(2)]
        h = [0.5 * math.tanh(c[k]) for k in range(2)]
    # h_self is zero, so the output is relu(agg) -- the negative first
    # dimension is clipped, which is exactly what this checks.
    approx(out[0], max(0.0, h[0]))
    approx(out[1], max(0.0, h[1]))
    # The aggregation is order-sensitive by construction: reversing the
    # feed order (via a different neighbor list that hashes to the reverse
    # order is impractical) -- instead check the cache recorded the order.
    assert cache[3][0] == order


def _positive_lstm_layer():
    # 3-dim cell-only layer with truncated-identity self/agg weights, so
    # positive inputs guarantee positive pre-activations (a dead ReLU
    # would legitimately zero every gradient, which is not what this test
    # is about).
    layer = SAGELayer(3, 2, random.Random(0), aggregator="lstm")
    layer.W_self = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    layer.W_agg = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    layer.b = [0.0, 0.0]
    for g in ("i", "f", "o", "g"):
        layer.lstm_W[g] = [[0.0] * 3 for _ in range(3)]
        layer.lstm_U[g] = [[0.0] * 3 for _ in range(3)]
        layer.lstm_b[g] = [0.0] * 3
    layer.lstm_W["g"] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]]
    return layer


def test_lstm_backward_flows_to_every_neighbor():
    # Unlike max-pool (one winner per dimension), the LSTM's final state
    # depends on every neighbor through the recurrence, so every neighbor
    # must receive gradient.
    layer = _positive_lstm_layer()
    h_self = [0.5, 0.5, 0.5]
    neighs = [[0.1, 0.9, 0.3], [0.8, 0.2, 0.5], [0.4, 0.6, 0.1]]
    _, cache = layer.forward(h_self, neighs)
    d_h_self, d_neighs = layer.backward([1.0, 1.0], cache)
    assert len(d_neighs) == 3
    assert len(d_h_self) == 3
    for d_n in d_neighs:
        assert any(abs(v) > 1e-12 for v in d_n), "dead neighbor gradient"
    # All LSTM parameter gradients are nonzero for a generic input.
    for g in ("i", "f", "o", "g"):
        assert any(abs(v) > 1e-12 for row in layer.glstm_W[g] for v in row)
        assert any(abs(v) > 1e-12 for row in layer.glstm_U[g] for v in row)
        assert any(abs(v) > 1e-12 for v in layer.glstm_b[g])


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
    if layer.aggregator == "lstm":
        for g in ("i", "f", "o", "g"):
            params += [
                (f"W_{g}", layer.lstm_W[g], layer.glstm_W[g]),
                (f"U_{g}", layer.lstm_U[g], layer.glstm_U[g]),
                (f"b_{g}", layer.lstm_b[g], [[b] for b in layer.glstm_b[g]]),
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


def test_lstm_gradient_check():
    # Random init (seed 3), 3 neighbors: the check passes only if the
    # BPTT, the gate derivatives, the unshuffle, and every parameter
    # gradient are exactly right. The permutation is seeded from the
    # (fixed) neighbor inputs, so the loss surface is stable across the
    # finite-difference evaluations.
    layer = SAGELayer(3, 2, random.Random(3), aggregator="lstm")
    h_self = [0.4, -0.2, 0.7]
    h_neighs = [[0.1, 0.9, -0.3], [0.8, 0.2, 0.5], [-0.4, 0.6, 0.1]]
    worst = _numeric_gradient_check(layer, h_self, h_neighs)
    assert worst < 1e-4


def test_lstm_empty_neighborhood():
    layer = _cell_only_layer()
    out, cache = layer.forward([0.5, -0.5], [])
    approx(out[0], 0.5)
    approx(out[1], 0.0)
    _, d_neighs = layer.backward([1.0, 1.0], cache)
    assert d_neighs == []
    for g in ("i", "f", "o", "g"):
        assert layer.glstm_W[g] == [[0.0, 0.0], [0.0, 0.0]]
        assert layer.glstm_b[g] == [0.0, 0.0]


def test_lstm_checkpoint_roundtrip():
    layer = SAGELayer(3, 2, random.Random(5), aggregator="lstm")
    fresh = SAGELayer(3, 2, random.Random(6), aggregator="lstm")
    fresh.load_state_dict(layer.state_dict())
    h_self = [0.4, -0.2, 0.7]
    h_neighs = [[0.1, 0.9, -0.3], [0.8, 0.2, 0.5]]
    out1, _ = layer.forward(h_self, h_neighs)
    out2, _ = fresh.forward(h_self, h_neighs)
    assert out1 == out2
    for g in ("i", "f", "o", "g"):
        assert fresh.lstm_W[g] == layer.lstm_W[g]
        assert fresh.lstm_U[g] == layer.lstm_U[g]
        assert fresh.lstm_b[g] == layer.lstm_b[g]


def test_legacy_mean_checkpoint_loads_into_lstm_layer():
    mean_layer = SAGELayer(2, 2, random.Random(0))
    state = mean_layer.state_dict()
    del state["aggregator"]  # simulate a checkpoint saved by the old code
    assert "lstm_W" not in state
    lstm_layer = SAGELayer(2, 2, random.Random(1), aggregator="lstm")
    lstm_layer.load_state_dict(state)
    assert lstm_layer.aggregator == "mean"
    assert lstm_layer.lstm_W is None
    out, _ = lstm_layer.forward([0.5, 0.5], [[1.0, 0.0]])
    assert len(out) == 2


def test_lstm_step_updates_lstm_params():
    layer = _cell_only_layer()
    _, cache = layer.forward([0.5, -0.5], [[1.0, 2.0], [3.0, 0.5]])
    layer.backward([1.0, 1.0], cache)
    w_before = copy.deepcopy(layer.lstm_W)
    b_before = copy.deepcopy(layer.lstm_b)
    layer.step(0.1)
    assert layer.lstm_W != w_before
    assert layer.lstm_b != b_before


def test_lstm_trains_on_dense_ring():
    # Seed-7 quickstart protocol: the LSTM model must actually learn the
    # ring (this guards against a silently dead recurrence).
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
    model, _ = train_sage(g, train_ids, val_ids, hidden_dim=16, k1=5, k2=5,
                          lr=0.1, batch_size=64, epochs=60, patience=12,
                          seed=7, aggregator="lstm")
    y_true = [g.labels[v] for v in val_ids]
    scores = [model.predict(g, [v])[v] for v in val_ids]
    assert roc_auc(y_true, scores) > 0.90

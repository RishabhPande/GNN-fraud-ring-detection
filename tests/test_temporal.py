"""Tests for temporal edges: timestamp storage, recency weights, the
temporal layer's hand-computed forward pass, a numeric gradient check on
the learned timescale, burst planting in the generator, and the headline
claim that recency-weighted message passing beats features alone."""

import math
import random

from gnn_fraud import (
    Graph,
    GraphGenerator,
    TemporalGraphSAGE,
    TemporalSAGELayer,
    recency_weights,
    roc_auc,
    train_mlp,
    train_temporal,
    train_val_split,
)


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


# ---------------------------------------------------------------------------
# edge timestamps on the graph
# ---------------------------------------------------------------------------


def test_edge_time_stored_and_undirected():
    g = Graph(3, 1)
    g.add_edge(0, 1, time=2.5)
    g.add_edge(1, 2)  # untimestamped
    approx(g.edge_time(0, 1), 2.5)
    approx(g.edge_time(1, 0), 2.5)  # undirected access
    assert g.edge_time(1, 2) is None


def test_edge_time_missing_edge_raises():
    g = Graph(3, 1)
    g.add_edge(0, 1, time=1.0)
    try:
        g.edge_time(0, 2)
        raise AssertionError("expected IndexError")
    except IndexError:
        pass


def test_max_edge_time():
    g = Graph(3, 1)
    assert g.max_edge_time() is None  # no timestamped edges
    g.add_edge(0, 1)
    assert g.max_edge_time() is None
    g.add_edge(1, 2, time=4.0)
    g.add_edge(0, 2, time=7.5)
    approx(g.max_edge_time(), 7.5)


# ---------------------------------------------------------------------------
# recency weights: hand-computable cases
# ---------------------------------------------------------------------------


def test_recency_weights_hand_computed():
    # times [8, 9], t_now 10 -> deltas [2, 1], tau 2 -> logits [-1, -0.5].
    w, d = recency_weights([8.0, 9.0], 10.0, math.log(2.0))
    assert d == [2.0, 1.0]
    e0, e1 = math.exp(-1.0), math.exp(-0.5)
    approx(w[0], e0 / (e0 + e1))
    approx(w[1], e1 / (e0 + e1))
    approx(sum(w), 1.0)
    # the more recent edge (smaller delta) gets the larger weight
    assert w[1] > w[0]


def test_recency_weights_equal_times_uniform():
    w, _ = recency_weights([5.0, 5.0, 5.0], 10.0, math.log(2.0))
    assert len(w) == 3
    for x in w:
        approx(x, 1.0 / 3.0)


def test_recency_weights_large_tau_near_uniform():
    # tau -> infinity: every logit -> 0, weights -> uniform.
    w, _ = recency_weights([1.0, 5.0, 9.0], 10.0, 10.0)
    for x in w:
        approx(x, 1.0 / 3.0, tol=1e-3)


def test_recency_weights_empty():
    w, d = recency_weights([], 10.0, 0.0)
    assert w == [] and d == []


# ---------------------------------------------------------------------------
# temporal layer: hand-computed forward + numeric gradient check
# ---------------------------------------------------------------------------


def _identity_temporal_layer():
    layer = TemporalSAGELayer(2, 2, random.Random(0))
    layer.W_self = [[1.0, 0.0], [0.0, 1.0]]
    layer.W_agg = [[1.0, 0.0], [0.0, 1.0]]
    layer.b = [0.0, 0.0]
    layer.log_tau = math.log(2.0)
    return layer


def test_temporal_layer_forward_hand_computed():
    layer = _identity_temporal_layer()
    # times [8, 9], t_now 10, tau 2 -> weights softmax(-1, -0.5)
    out, cache = layer.forward_temporal(
        [1.0, 0.0], [[0.0, 2.0], [4.0, 0.0]], [8.0, 9.0], 10.0)
    e0, e1 = math.exp(-1.0), math.exp(-0.5)
    w0, w1 = e0 / (e0 + e1), e1 / (e0 + e1)
    # agg = w0*[0,2] + w1*[4,0] = [4*w1, 2*w0]; out = relu([1,0] + agg)
    approx(out[0], 1.0 + 4.0 * w1)
    approx(out[1], 2.0 * w0)
    # isolated node: empty neighborhood -> self only
    out2, _ = layer.forward_temporal([1.0, 0.0], [], [], 10.0)
    approx(out2[0], 1.0)
    approx(out2[1], 0.0)


def _positive_temporal_layer():
    # All-positive weights/inputs/bias: relu never kinks, so numeric
    # gradient checks see a smooth function.
    layer = TemporalSAGELayer(2, 2, random.Random(0))
    layer.W_self = [[0.5, 0.1], [0.2, 0.6]]
    layer.W_agg = [[0.4, 0.2], [0.1, 0.5]]
    layer.b = [0.1, 0.1]
    layer.log_tau = math.log(3.0)
    return layer


def test_temporal_layer_log_tau_numeric_gradient():
    layer = _positive_temporal_layer()
    h_self = [1.0, 0.5]
    h_neighs = [[0.5, 1.0], [1.5, 0.25]]
    times = [7.0, 9.5]
    t_now = 10.0

    def loss(log_tau):
        layer.log_tau = log_tau
        out, _ = layer.forward_temporal(h_self, h_neighs, times, t_now)
        return sum(out)

    base = math.log(3.0)
    eps = 1e-6
    numeric = (loss(base + eps) - loss(base - eps)) / (2 * eps)

    layer.log_tau = base
    layer.zero_grad()
    out, cache = layer.forward_temporal(h_self, h_neighs, times, t_now)
    assert all(x > 0 for x in out)  # no relu kink
    layer.backward_temporal([1.0, 1.0], cache)
    approx(layer.g_log_tau, numeric, tol=1e-5)

    # The weight gradients must be consistent too (spot-check one entry).
    def loss_w(w000):
        layer.log_tau = base
        layer.W_self[0][0] = w000
        out, _ = layer.forward_temporal(h_self, h_neighs, times, t_now)
        return sum(out)

    w0 = layer.W_self[0][0]
    numeric_w = (loss_w(w0 + eps) - loss_w(w0 - eps)) / (2 * eps)
    approx(layer.gW_self[0][0], numeric_w, tol=1e-5)


def test_temporal_layer_step_moves_log_tau():
    layer = _positive_temporal_layer()
    out, cache = layer.forward_temporal(
        [1.0, 0.5], [[0.5, 1.0], [1.5, 0.25]], [7.0, 9.5], 10.0)
    layer.zero_grad()
    layer.backward_temporal([1.0, 1.0], cache)
    before = layer.log_tau
    g = layer.g_log_tau
    assert g != 0.0  # the timescale actually receives gradient
    layer.step(0.1)
    approx(layer.log_tau, before - 0.1 * g)
    # checkpointing round-trips the timescale
    state = layer.state_dict()
    assert "log_tau" in state
    layer2 = TemporalSAGELayer(2, 2, random.Random(1))
    layer2.load_state_dict(state)
    approx(layer2.log_tau, layer.log_tau)


# ---------------------------------------------------------------------------
# generator: the ring forms in a fast late burst
# ---------------------------------------------------------------------------


def test_generator_ring_forms_in_burst():
    gen = GraphGenerator(seed=42)
    g = gen.generate(n=200, ring_size=18, timestamps=True,
                     timeline_days=30.0, ring_burst_days=3.0)
    ring = set(gen.ring_nodes(g))
    ring_times = [g.edge_time(u, v) for u in ring for v in g.neighbors(u)
                  if v in ring and u < v]
    assert len(ring_times) > 50  # dense burst, not a handful of edges
    # every ring-ring edge lands in the late burst window [27, 30]
    assert min(ring_times) >= 27.0, min(ring_times)
    assert max(ring_times) <= 30.0
    # benign edges spread over the whole timeline
    bg_times = [g.edge_time(u, v) for u in range(g.n) for v in g.neighbors(u)
                if u < v and not (u in ring and v in ring)]
    assert min(bg_times) < 10.0, "benign edges should span the timeline"
    assert max(bg_times) > 20.0


def test_generator_timestamps_do_not_change_structure():
    # Timestamps come from a separate RNG stream: the planted structure is
    # identical whether timestamps are on or off (pinned seeds reproducible).
    g_on = GraphGenerator(seed=9).generate(n=100, ring_size=10,
                                           timestamps=True)
    g_off = GraphGenerator(seed=9).generate(n=100, ring_size=10,
                                            timestamps=False)
    assert g_on.labels == g_off.labels
    assert g_on.features == g_off.features
    assert [g_on.neighbors(u) for u in range(100)] == \
           [g_off.neighbors(u) for u in range(100)]
    assert g_off.max_edge_time() is None
    assert g_on.max_edge_time() is not None


def test_temporal_model_requires_timestamps():
    g = GraphGenerator(seed=1).generate(n=30, ring_size=5, timestamps=False)
    model = TemporalGraphSAGE(g.feature_dim, hidden_dim=4, seed=0)
    try:
        model.predict(g, [0, 1])
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "timestamp" in str(e)


# ---------------------------------------------------------------------------
# training dynamics + headline claim
# ---------------------------------------------------------------------------


def test_temporal_loss_decreases():
    g = GraphGenerator(seed=123).generate(n=60, ring_size=8)
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=1)
    _, hist = train_temporal(g, train_ids, val_ids, hidden_dim=8, k1=3, k2=3,
                             lr=0.1, batch_size=32, epochs=8, patience=8,
                             seed=1)
    assert hist["train_loss"][-1] < hist["train_loss"][0]


def test_temporal_predict_deterministic():
    g = GraphGenerator(seed=123).generate(n=60, ring_size=8)
    train_ids, val_ids = train_val_split(g.labels, seed=1)
    model, _ = train_temporal(g, train_ids, val_ids, epochs=2, patience=2,
                              seed=1)
    assert model.predict(g, val_ids) == model.predict(g, val_ids)


def test_temporal_beats_feature_baseline():
    # Same pinned protocol as the GraphSAGE headline test: recency-weighted
    # message passing over the burst must beat node features alone.
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
    y_true = [g.labels[v] for v in val_ids]

    temporal, _ = train_temporal(g, train_ids, val_ids, hidden_dim=16,
                                 k1=5, k2=5, lr=0.1, batch_size=64,
                                 epochs=40, patience=40, seed=7)
    mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                       hidden_dim=16, lr=0.3, batch_size=32,
                       epochs=100, patience=100, seed=7)
    temporal_auc = roc_auc(
        y_true, [temporal.predict(g, [v])[v] for v in val_ids])
    mlp_auc = roc_auc(y_true, mlp.predict([g.features[v] for v in val_ids]))
    assert temporal_auc > 0.85, f"temporal auc {temporal_auc}"
    assert temporal_auc > mlp_auc + 0.15, \
        f"temporal {temporal_auc} vs mlp {mlp_auc}"

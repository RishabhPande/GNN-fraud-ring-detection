"""Tests for the GGNN: hand-computed gated steps, numeric gradient
checks over every GRU parameter, checkpointing, and training dynamics
on the dense ring."""

import math

from gnn_fraud import GGNN, Graph, evaluate_all, train_ggnn, train_val_split
from gnn_fraud import GraphGenerator


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def _two_node_graph():
    g = Graph(2, 1)
    g.add_edge(0, 1)
    g.set_features(0, [1.0])
    g.set_features(1, [2.0])
    return g


def _zero_gru(model):
    h = model.hidden_dim
    for name in ("W_z", "U_z", "W_r", "U_r", "W_h", "U_h"):
        setattr(model, name, [[0.0] * h for _ in range(h)])
    for name in ("b_z", "b_r", "b_h"):
        setattr(model, name, [0.0] * h)
    return model


# ---------------------------------------------------------------------------
# hand-computed cases
# ---------------------------------------------------------------------------


def test_ggnn_steps0_is_encoder_plus_head():
    # steps=0 skips propagation: the model is exactly relu(W_in x +
    # b_in) scored by the linear head.
    g = _two_node_graph()
    model = GGNN(1, hidden_dim=2, steps=0, seed=0)
    model.W_in = [[2.0], [-1.0]]
    model.b_in = [0.5, 0.0]
    model.w_out = [1.0, -2.0]
    model.b_out = 0.25
    probs = model.predict(g, [0, 1])
    # Node 0: h = [2.5, 0]; logit = 2.75. Node 1: h = [4.5, 0]; 4.75.
    approx(probs[0], 1.0 / (1.0 + math.exp(-2.75)))
    approx(probs[1], 1.0 / (1.0 + math.exp(-4.75)))


def test_ggnn_zero_gru_halves_state_each_step():
    # With every GRU weight and bias at zero, z = r = sigmoid(0) = 0.5
    # and the candidate is tanh(0) = 0, so each step maps
    # h -> (1 - 0.5) * h + 0.5 * 0 = h / 2, whatever the messages are.
    g = _two_node_graph()
    model = _zero_gru(GGNN(1, hidden_dim=1, steps=2, seed=0))
    model.W_in = [[1.0]]
    model.b_in = [0.0]
    model.w_out = [1.0]
    model.b_out = 0.0
    probs = model.predict(g, [0, 1])
    # h0 = [1, 2]; after 2 steps h = [0.25, 0.5]; logits are those.
    approx(probs[0], 1.0 / (1.0 + math.exp(-0.25)))
    approx(probs[1], 1.0 / (1.0 + math.exp(-0.5)))


def test_ggnn_isolated_node_gets_zero_message():
    # An isolated node has an empty neighborhood: its message is the
    # zero vector, so with a zero GRU its state still just halves,
    # exactly like a connected node's (the gates see a = 0 either way
    # when the GRU is zeroed - the point is no crash and no NaN).
    g = Graph(3, 1)
    g.add_edge(0, 1)
    g.set_features(0, [1.0])
    g.set_features(1, [2.0])
    g.set_features(2, [4.0])
    model = _zero_gru(GGNN(1, hidden_dim=1, steps=1, seed=0))
    model.W_in = [[1.0]]
    model.b_in = [0.0]
    model.w_out = [1.0]
    model.b_out = 0.0
    probs = model.predict(g, [2])
    approx(probs[2], 1.0 / (1.0 + math.exp(-2.0)))


def test_ggnn_bad_params_rejected():
    for kwargs in ({"steps": -1}, {"hidden_dim": 0}):
        try:
            GGNN(6, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


def test_ggnn_gate_means_in_unit_interval():
    g = _two_node_graph()
    model = GGNN(1, hidden_dim=4, steps=2, seed=3)
    means = model.gate_means(g)
    assert 0.0 < means["update"] < 1.0
    assert 0.0 < means["reset"] < 1.0


# ---------------------------------------------------------------------------
# numeric gradient check + checkpointing
# ---------------------------------------------------------------------------


def test_ggnn_model_gradient_check():
    # Encoder + 2 gated steps + head: perturb every parameter (W_in,
    # b_in, all six GRU matrices, all three gate biases, w_out,
    # b_out) and compare against the gradients train_batch
    # accumulates (lr=0.0). This validates backprop through time,
    # including the message scatter to neighbors.
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    g.set_features(3, [0.1, 0.4])
    labels = {0: 1, 1: 0, 2: 1, 3: 0}
    model = GGNN(2, hidden_dim=3, steps=2, lr=0.0, seed=5)

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

    def check_mat(P, G):
        for i in range(len(P)):
            for j in range(len(P[i])):
                check(G[i][j],
                      lambda i=i, j=j: P[i][j],
                      lambda v, i=i, j=j: P[i].__setitem__(j, v))

    def check_vec(p, gvec):
        for i in range(len(p)):
            check(gvec[i],
                  lambda i=i: p[i],
                  lambda v, i=i: p.__setitem__(i, v))

    check_mat(model.W_in, model.gW_in)
    check_vec(model.b_in, model.gb_in)
    check_mat(model.W_z, model.gW_z)
    check_mat(model.U_z, model.gU_z)
    check_vec(model.b_z, model.gb_z)
    check_mat(model.W_r, model.gW_r)
    check_mat(model.U_r, model.gU_r)
    check_vec(model.b_r, model.gb_r)
    check_mat(model.W_h, model.gW_h)
    check_mat(model.U_h, model.gU_h)
    check_vec(model.b_h, model.gb_h)
    check_vec(model.w_out, model.gw_out)
    check(model.gb_out, lambda: model.b_out,
          lambda v: setattr(model, "b_out", v))


def test_ggnn_checkpoint_round_trip():
    model = GGNN(6, hidden_dim=8, steps=3, seed=1)
    state = model.state_dict()
    model2 = GGNN(6, hidden_dim=8, steps=3, seed=99)
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


def test_ggnn_trains_to_high_auc_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    model, _ = train_ggnn(g, train_ids, val_ids, steps=3, hidden_dim=16,
                          lr=0.5, batch_size=64, epochs=50,
                          patience=12, seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    y_true = [g.labels[v] for v in val_ids]
    assert evaluate_all(y_true, scores)["auc"] >= 0.90


def test_ggnn_gates_move_from_half():
    # The gates are the mechanism, so pin that training actually
    # moves them: the mean over every node and dimension stays near
    # 0.5 (some dimensions open, others close), but per-dimension
    # mean update gates spread measurably away from the sigmoid(0) =
    # 0.5 start. A frozen gate - dead GRU gradient - would leave
    # every dimension's mean within noise of 0.5 and the spread at
    # its untrained width. Observed on this schedule: max
    # per-dimension deviation 0.159, spread (pstdev over all gate
    # values) 0.061.
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    model, _ = train_ggnn(g, train_ids, val_ids, steps=3, hidden_dim=16,
                          lr=0.5, batch_size=64, epochs=50,
                          patience=12, seed=7)
    _, cache = model.forward(g, [])
    zs = [entry["z"] for sc in cache["steps"] for entry in sc]
    hd = model.hidden_dim
    per_dim = [sum(z[j] for z in zs) / len(zs) for j in range(hd)]
    assert max(abs(x - 0.5) for x in per_dim) > 0.05
    flat = [x for z in zs for x in z]
    mean = sum(flat) / len(flat)
    var = sum((x - mean) ** 2 for x in flat) / len(flat)
    assert var ** 0.5 > 0.03

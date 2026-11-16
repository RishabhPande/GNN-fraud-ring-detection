"""Tests for DAGNN: hand-computed hop attention (uniform at the
zero init, softmax of the projection scores otherwise), its
steps=0 MLP equivalence, numeric gradient checks end to end
(including the projection vector s, through both the direct and
the attention paths), checkpointing, bad-parameter rejection, and
training dynamics on the dense ring."""

import math

from gnn_fraud import (
    DAGNN,
    Graph,
    GraphGenerator,
    evaluate_all,
    hop_attention,
    train_dagnn,
    train_val_split,
)


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def test_hop_attention_uniform_at_zero_projection():
    # s = 0 makes every logit 0, so softmax is exactly uniform,
    # whatever the hidden states are.
    a = hop_attention([[1.0, -2.0], [3.5, 0.25], [-1.0, 1.0]],
                      [0.0, 0.0])
    for w in a:
        approx(w, 1.0 / 3.0)


def test_hop_attention_hand_computed():
    # s . h gives logits [1, 2, 0]; softmax by hand.
    a = hop_attention([[1.0], [2.0], [0.0]], [1.0])
    exps = [math.exp(1.0), math.exp(2.0), math.exp(0.0)]
    total = sum(exps)
    for got, e in zip(a, exps):
        approx(got, e / total)
    approx(sum(a), 1.0)


def test_hop_attention_empty_rejected():
    try:
        hop_attention([], [1.0])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_dagnn_steps_zero_ignores_graph():
    # steps=0: the adaptive sum has one term with weight 1, so the
    # model is exactly an MLP on raw features and edges change
    # nothing.
    g = Graph(2, 1)
    g.add_edge(0, 1)
    g.set_features(0, [1.0])
    g.set_features(1, [3.0])
    g2 = Graph(2, 1)
    g2.set_features(0, [1.0])
    g2.set_features(1, [3.0])
    model = DAGNN(1, hidden_dim=4, steps=0, seed=3)
    p1 = model.predict(g, [0, 1])
    p2 = model.predict(g2, [0, 1])
    approx(p1[0], p2[0])
    approx(p1[1], p2[1])
    assert model.attention_weights(g, 0) == [1.0]


def test_dagnn_attention_weights_sum_to_one():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=50, ring_size=6)
    model = DAGNN(6, hidden_dim=8, steps=4, seed=1)
    model.s = [0.3, -0.2, 0.1, 0.0, 0.5, -0.4, 0.2, 0.1]
    a = model.attention_weights(g, 0)
    assert len(a) == 5
    approx(sum(a), 1.0)
    assert all(w > 0.0 for w in a)


def test_dagnn_bad_params_rejected():
    for kwargs in ({"steps": -1}, {"hidden_dim": 0}):
        try:
            DAGNN(6, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


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


def test_dagnn_model_gradient_check():
    # Perturb every parameter (encoder, projection s, head) and
    # compare with the gradients train_batch accumulates at lr=0.0.
    # The s check only bites with s nonzero: at s = 0 the attention
    # path into the hidden states vanishes, so set s off zero first
    # so a wrong softmax-Jacobian term fails here.
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    g.set_features(3, [0.1, 0.4])
    labels = {0: 1, 1: 0, 2: 1, 3: 0}
    model = DAGNN(2, hidden_dim=3, steps=3, lr=0.0, seed=5)
    model.s = [0.4, -0.3, 0.2]
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
        check(model.gs[i],
              lambda i=i: model.s[i],
              lambda v, i=i: model.s.__setitem__(i, v))
        check(model.gw_out[i],
              lambda i=i: model.w_out[i],
              lambda v, i=i: model.w_out.__setitem__(i, v))
    check(model.gb_out, lambda: model.b_out,
          lambda v: setattr(model, "b_out", v))


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------


def test_dagnn_checkpoint_round_trip():
    model = DAGNN(6, hidden_dim=8, steps=3, seed=1)
    model.s = [0.1] * 8
    state = model.state_dict()
    model2 = DAGNN(6, hidden_dim=8, steps=3, seed=99)
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


def test_dagnn_trains_to_high_auc_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    model, _ = train_dagnn(g, train_ids, val_ids, hidden_dim=16,
                           steps=10, lr=0.5, batch_size=64, epochs=100,
                           patience=25, seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    y_true = [g.labels[v] for v in val_ids]
    assert evaluate_all(y_true, scores)["auc"] >= 0.90


def test_dagnn_attention_moves_off_uniform():
    # The per-node mix is the finding, so pin that training actually
    # moves it: a dead s branch would leave every node at the
    # uniform 1 / (steps + 1) mix throughout.
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    model, _ = train_dagnn(g, train_ids, val_ids, hidden_dim=16,
                           steps=10, lr=0.5, batch_size=64, epochs=100,
                           patience=25, seed=7)
    assert any(abs(x) > 1e-3 for x in model.s)
    spreads = []
    for v in val_ids:
        a = model.attention_weights(g, v)
        spreads.append(max(a) - min(a))
    # The movement is small on this generator (max spread about
    # 0.014, the same near-uniform story as GAT's attention), so
    # pin movement off uniform, not a large specialization.
    assert max(spreads) > 0.01

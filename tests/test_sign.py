"""Tests for SIGN: hand-computed feature concatenation, numeric gradient
checks, checkpointing, and training dynamics on the dense ring."""

import math

from gnn_fraud import (
    Graph,
    GraphGenerator,
    SIGN,
    evaluate_all,
    sign_features,
    train_sgc,
    train_sign,
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
# concatenation: hand-computable cases
# ---------------------------------------------------------------------------


def test_sign_features_zero_hops_is_raw_features():
    g = _two_node_graph()
    out = sign_features(g, 0)
    assert len(out[0]) == 1
    approx(out[0][0], 1.0)
    approx(out[1][0], 2.0)
    out[0][0] = 99.0  # mutating the result must not touch the graph
    approx(g.features[0][0], 1.0)


def test_sign_features_one_hop_hand_computed():
    # Degrees are 1, so d_hat = 2 for both nodes: every weight is
    # 1 / sqrt(2 * 2) = 0.5. Hop-1 block is 0.5*1 + 0.5*2 = 1.5 for both.
    g = _two_node_graph()
    out = sign_features(g, 1)
    assert len(out[0]) == 2
    approx(out[0][0], 1.0)  # hop 0: own feature
    approx(out[0][1], 1.5)  # hop 1: propagated feature
    approx(out[1][0], 2.0)
    approx(out[1][1], 1.5)


def test_sign_features_two_hops_stacks_three_blocks():
    # Same graph: hop 2 is a fixed point at 1.5, so all three blocks
    # are known exactly.
    g = _two_node_graph()
    out = sign_features(g, 2)
    assert len(out[0]) == 3
    approx(out[0][0], 1.0)
    approx(out[0][1], 1.5)
    approx(out[0][2], 1.5)
    approx(out[1][0], 2.0)
    approx(out[1][1], 1.5)
    approx(out[1][2], 1.5)


def test_sign_features_negative_hops_rejected():
    g = _two_node_graph()
    try:
        sign_features(g, -1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for hops=-1")


def test_sign_bad_hidden_dim_rejected():
    try:
        SIGN(6, hops=2, hidden_dim=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for hidden_dim=0")


# ---------------------------------------------------------------------------
# model: numeric gradient check + checkpointing
# ---------------------------------------------------------------------------


def test_sign_model_gradient_check():
    # Two-layer MLP head over concatenated features: perturb every
    # parameter and the biases, compare against the gradient
    # train_batch accumulates (lr=0.0, so the step moves nothing).
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    g.set_features(3, [0.1, 0.4])
    labels = {0: 1, 1: 0, 2: 1, 3: 0}
    model = SIGN(2, hops=2, hidden_dim=4, lr=0.0, seed=5)

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
        for j in range(6):
            check(model.gW1[i][j],
                  lambda i=i, j=j: model.W1[i][j],
                  lambda v, i=i, j=j: model.W1[i].__setitem__(j, v))
        check(model.gb1[i],
              lambda i=i: model.b1[i],
              lambda v, i=i: model.b1.__setitem__(i, v))
    for i in range(4):
        check(model.gW2[i],
              lambda i=i: model.W2[i],
              lambda v, i=i: model.W2.__setitem__(i, v))
    check(model.gb2, lambda: model.b2, lambda v: setattr(model, "b2", v))


def test_sign_checkpoint_round_trip():
    model = SIGN(6, hops=2, hidden_dim=8, seed=1)
    state = model.state_dict()
    model2 = SIGN(6, hops=2, hidden_dim=8, seed=99)
    model2.load_state_dict(state)
    g = Graph(2, 6)
    g.add_edge(0, 1)
    g.set_features(0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    g.set_features(1, [0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    p1 = model.predict(g, [0, 1])
    p2 = model2.predict(g, [0, 1])
    approx(p1[0], p2[0])
    approx(p1[1], p2[1])


def test_sign_hops_zero_matches_mlp_on_raw_features():
    # With hops=0 the concatenated input is the raw features, so two
    # models with identical weights must score identically to a
    # hand-computed two-layer MLP on the raw features.
    g = _two_node_graph()
    model = SIGN(1, hops=0, hidden_dim=2, seed=0)
    model.W1 = [[2.0], [-1.0]]
    model.b1 = [0.5, 0.0]
    model.W2 = [1.0, -2.0]
    model.b2 = 0.25
    probs = model.predict(g, [0, 1])
    # Node 0: pre = [2*1+0.5, -1*1+0] = [2.5, -1] -> h = [2.5, 0];
    # logit = 2.5 + 0.25 = 2.75.
    approx(probs[0], 1.0 / (1.0 + math.exp(-2.75)))
    # Node 1: pre = [4.5, -2] -> h = [4.5, 0]; logit = 4.75.
    approx(probs[1], 1.0 / (1.0 + math.exp(-4.75)))


# ---------------------------------------------------------------------------
# training dynamics
# ---------------------------------------------------------------------------


def test_sign_trains_to_high_auc_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    model, _ = train_sign(g, train_ids, val_ids, hops=2, hidden_dim=16,
                          lr=0.5, batch_size=64, epochs=100,
                          patience=25, seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    y_true = [g.labels[v] for v in val_ids]
    assert evaluate_all(y_true, scores)["auc"] >= 0.90


def test_sign_matches_sgc_protocol_on_dense_ring():
    # Same graph, same split, same schedule as the SGC headline: SIGN
    # should not fall behind the model it generalizes.
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    y_true = [g.labels[v] for v in val_ids]
    sign, _ = train_sign(g, train_ids, val_ids, hops=2, hidden_dim=16,
                        lr=0.5, batch_size=64, epochs=100,
                        patience=25, seed=7)
    sgc, _ = train_sgc(g, train_ids, val_ids, hops=2,
                       lr=0.5, batch_size=64, epochs=100,
                       patience=25, seed=7)
    sign_auc = evaluate_all(
        y_true, [sign.predict(g, [v])[v] for v in val_ids])["auc"]
    sgc_auc = evaluate_all(
        y_true, [sgc.predict(g, [v])[v] for v in val_ids])["auc"]
    assert sign_auc >= sgc_auc - 0.05

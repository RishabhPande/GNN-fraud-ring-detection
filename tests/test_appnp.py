"""Tests for APPNP: hand-computed personalized PageRank propagation,
numeric gradient checks through propagation and the MLP, checkpointing,
and training dynamics on the dense ring."""

import math

from gnn_fraud import (
    APPNP,
    Graph,
    GraphGenerator,
    evaluate_all,
    propagate_predictions,
    train_appnp,
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
# propagation: hand-computable cases
# ---------------------------------------------------------------------------


def test_propagate_zero_steps_is_initial_logits():
    g = _two_node_graph()
    out = propagate_predictions(g, [1.0, 2.0], steps=0, alpha=0.1)
    approx(out[0], 1.0)
    approx(out[1], 2.0)
    out[0] = 99.0  # mutating the result must not touch the input
    # (a second call still sees the original logits)
    out2 = propagate_predictions(g, [1.0, 2.0], steps=0, alpha=0.1)
    approx(out2[0], 1.0)


def test_propagate_one_step_hand_computed():
    # Degrees are 1, so d_hat = 2 for both nodes: every weight is 0.5,
    # so S z = [1.5, 1.5] for z0 = [1, 2]. With alpha = 0.1:
    # node 0: 0.9 * 1.5 + 0.1 * 1 = 1.45, node 1: 0.9 * 1.5 + 0.1 * 2 = 1.55.
    g = _two_node_graph()
    out = propagate_predictions(g, [1.0, 2.0], steps=1, alpha=0.1)
    approx(out[0], 1.45)
    approx(out[1], 1.55)


def test_propagate_alpha_one_is_identity():
    g = _two_node_graph()
    out = propagate_predictions(g, [1.0, 2.0], steps=10, alpha=1.0)
    approx(out[0], 1.0)
    approx(out[1], 2.0)


def test_propagate_alpha_zero_is_plain_propagation():
    # alpha = 0 drops the teleport, so one step is S z = [1.5, 1.5].
    g = _two_node_graph()
    out = propagate_predictions(g, [1.0, 2.0], steps=1, alpha=0.0)
    approx(out[0], 1.5)
    approx(out[1], 1.5)


def test_propagate_isolated_node_keeps_own_logit():
    # An isolated node has d_hat = 1, so its only weight is 1: the
    # propagation term and the teleport term agree, at any alpha.
    g = Graph(3, 1)
    g.add_edge(0, 1)
    g.set_features(0, [0.0])
    g.set_features(1, [0.0])
    g.set_features(2, [0.0])
    out = propagate_predictions(g, [1.0, -1.0, 3.0], steps=5, alpha=0.1)
    approx(out[2], 3.0)


def test_propagate_invalid_args_rejected():
    g = _two_node_graph()
    for kwargs in ({"steps": -1, "alpha": 0.1},
                   {"steps": 1, "alpha": -0.1},
                   {"steps": 1, "alpha": 1.1}):
        try:
            propagate_predictions(g, [1.0, 2.0], **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"invalid args did not raise: {kwargs}")
    try:
        propagate_predictions(g, [1.0], steps=1, alpha=0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong-length z0 did not raise")


# ---------------------------------------------------------------------------
# model: numeric gradient check + checkpointing
# ---------------------------------------------------------------------------


def test_appnp_model_gradient_check():
    # Perturb every parameter, compare against the gradient train_batch
    # accumulates (lr=0.0, so the step moves nothing). Propagation mixes
    # every node's initial logit into every target's score, so this
    # checks the propagation backward as well as the MLP backward.
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    g.set_features(3, [0.1, 0.4])
    labels = {0: 1, 1: 0, 2: 1, 3: 0}
    model = APPNP(2, hidden_dim=3, steps=3, alpha=0.2, lr=0.0, seed=5)

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
    for j in range(3):
        for k in range(2):
            orig = model.W1[j][k]
            model.W1[j][k] = orig + eps
            plus = loss_fn()
            model.W1[j][k] = orig - eps
            minus = loss_fn()
            model.W1[j][k] = orig
            approx(model.gW1[j][k], (plus - minus) / (2 * eps), tol=1e-5)
        orig = model.b1[j]
        model.b1[j] = orig + eps
        plus = loss_fn()
        model.b1[j] = orig - eps
        minus = loss_fn()
        model.b1[j] = orig
        approx(model.gb1[j], (plus - minus) / (2 * eps), tol=1e-5)
        orig = model.w_out[j]
        model.w_out[j] = orig + eps
        plus = loss_fn()
        model.w_out[j] = orig - eps
        minus = loss_fn()
        model.w_out[j] = orig
        approx(model.gw_out[j], (plus - minus) / (2 * eps), tol=1e-5)
    orig = model.b_out
    model.b_out = orig + eps
    plus = loss_fn()
    model.b_out = orig - eps
    minus = loss_fn()
    model.b_out = orig
    approx(model.gb_out, (plus - minus) / (2 * eps), tol=1e-5)


def test_appnp_steps_zero_matches_its_own_mlp():
    # With steps=0, propagation is the identity, so the score must equal
    # a hand-computed MLP forward on the raw features.
    g = _two_node_graph()
    model = APPNP(1, hidden_dim=2, steps=0, alpha=0.1, seed=0)
    model.W1 = [[2.0], [-1.0]]
    model.b1 = [0.5, 0.0]
    model.w_out = [1.0, 1.0]
    model.b_out = 0.0
    probs = model.predict(g, [0, 1])
    # Node 0, x=1: hidden = [relu(2.5), relu(-1)] = [2.5, 0], logit 2.5.
    approx(probs[0], 1.0 / (1.0 + math.exp(-2.5)))
    # Node 1, x=2: hidden = [relu(4.5), relu(-2)] = [4.5, 0], logit 4.5.
    approx(probs[1], 1.0 / (1.0 + math.exp(-4.5)))


def test_appnp_checkpoint_round_trip():
    model = APPNP(6, hidden_dim=4, steps=5, alpha=0.1, seed=1)
    state = model.state_dict()
    model2 = APPNP(6, hidden_dim=4, steps=5, alpha=0.1, seed=99)
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


def test_appnp_trains_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
    model, hist = train_appnp(g, train_ids, val_ids, hidden_dim=16,
                              steps=10, alpha=0.1, lr=0.5,
                              batch_size=64, epochs=100,
                              patience=25, seed=7)
    assert min(hist["val_loss"]) < hist["val_loss"][0]
    y_true = [g.labels[v] for v in val_ids]
    scores = [model.predict(g, [v])[v] for v in val_ids]
    auc = evaluate_all(y_true, scores)["auc"]
    assert auc >= 0.90, f"APPNP AUC {auc:.3f} below 0.90"

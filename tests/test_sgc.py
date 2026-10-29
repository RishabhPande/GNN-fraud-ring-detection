"""Tests for SGC: hand-computed propagation, numeric gradient checks,
checkpointing, and training dynamics on the dense ring."""

import math

from gnn_fraud import (
    Graph,
    GraphGenerator,
    SimpleGraphConvolution,
    evaluate_all,
    propagate_features,
    train_sgc,
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


def test_propagate_zero_hops_is_raw_features():
    g = _two_node_graph()
    out = propagate_features(g, 0)
    approx(out[0][0], 1.0)
    approx(out[1][0], 2.0)
    out[0][0] = 99.0  # mutating the result must not touch the graph
    approx(g.features[0][0], 1.0)


def test_propagate_one_hop_hand_computed():
    # Degrees are 1, so d_hat = 2 for both nodes: every weight is
    # 1 / sqrt(2 * 2) = 0.5. Both nodes see 0.5*1 + 0.5*2 = 1.5.
    g = _two_node_graph()
    out = propagate_features(g, 1)
    approx(out[0][0], 1.5)
    approx(out[1][0], 1.5)
    # A second hop is a fixed point here: both rows are already 1.5.
    out2 = propagate_features(g, 2)
    approx(out2[0][0], 1.5)
    approx(out2[1][0], 1.5)


def test_propagate_star_hand_computed():
    # Hub 0 (degree 3, d_hat 4) with leaves 1, 2, 3 (d_hat 2).
    # Hub self-weight 1/4, hub-leaf 1/sqrt(8), leaf self-weight 1/2.
    g = Graph(4, 1)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)
    g.set_features(0, [0.0])
    g.set_features(1, [1.0])
    g.set_features(2, [0.0])
    g.set_features(3, [0.0])
    out = propagate_features(g, 1)
    approx(out[0][0], 1.0 / math.sqrt(8))
    approx(out[1][0], 0.5)
    approx(out[2][0], 0.0)
    approx(out[3][0], 0.0)


def test_propagate_isolated_node_keeps_own_features():
    # An isolated node has d_hat = 1, so its only weight is 1.
    g = Graph(3, 2)
    g.add_edge(0, 1)
    g.set_features(0, [1.0, 0.0])
    g.set_features(1, [0.0, 1.0])
    g.set_features(2, [3.0, -2.0])
    out = propagate_features(g, 1)
    approx(out[2][0], 3.0)
    approx(out[2][1], -2.0)


def test_propagate_negative_hops_rejected():
    g = _two_node_graph()
    try:
        propagate_features(g, -1)
    except ValueError:
        return
    raise AssertionError("negative hops did not raise")


# ---------------------------------------------------------------------------
# model: numeric gradient check + checkpointing
# ---------------------------------------------------------------------------


def test_sgc_model_gradient_check():
    # Logistic regression on propagated features: perturb each weight
    # and the bias, compare against the gradient train_batch accumulates
    # (lr=0.0, so the step moves nothing).
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    g.set_features(3, [0.1, 0.4])
    labels = {0: 1, 1: 0, 2: 1, 3: 0}
    model = SimpleGraphConvolution(2, hops=2, lr=0.0, seed=5)

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
    for j in range(2):
        orig = model.w[j]
        model.w[j] = orig + eps
        plus = loss_fn()
        model.w[j] = orig - eps
        minus = loss_fn()
        model.w[j] = orig
        approx(model.gw[j], (plus - minus) / (2 * eps), tol=1e-5)
    orig = model.b
    model.b = orig + eps
    plus = loss_fn()
    model.b = orig - eps
    minus = loss_fn()
    model.b = orig
    approx(model.gb, (plus - minus) / (2 * eps), tol=1e-5)


def test_sgc_checkpoint_round_trip():
    model = SimpleGraphConvolution(6, hops=2, seed=1)
    state = model.state_dict()
    model2 = SimpleGraphConvolution(6, hops=2, seed=99)
    model2.load_state_dict(state)
    g = Graph(2, 6)
    g.add_edge(0, 1)
    g.set_features(0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    g.set_features(1, [0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    p1 = model.predict(g, [0, 1])
    p2 = model2.predict(g, [0, 1])
    approx(p1[0], p2[0])
    approx(p1[1], p2[1])


def test_sgc_hops_zero_matches_logistic_on_raw_features():
    # With hops=0, propagation is the identity, so two models with the
    # same seed and weights must score identically to a hand-computed
    # logistic regression on the raw features.
    g = _two_node_graph()
    model = SimpleGraphConvolution(1, hops=0, seed=0)
    model.w = [2.0]
    model.b = 1.0
    probs = model.predict(g, [0, 1])
    approx(probs[0], 1.0 / (1.0 + math.exp(-3.0)))
    approx(probs[1], 1.0 / (1.0 + math.exp(-5.0)))


# ---------------------------------------------------------------------------
# training dynamics
# ---------------------------------------------------------------------------


def test_sgc_trains_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
    model, hist = train_sgc(g, train_ids, val_ids, hops=2,
                            lr=0.5, batch_size=64, epochs=100,
                            patience=25, seed=7)
    assert min(hist["val_loss"]) < hist["val_loss"][0]
    y_true = [g.labels[v] for v in val_ids]
    scores = [model.predict(g, [v])[v] for v in val_ids]
    auc = evaluate_all(y_true, scores)["auc"]
    assert auc >= 0.90, f"SGC AUC {auc:.3f} below 0.90"

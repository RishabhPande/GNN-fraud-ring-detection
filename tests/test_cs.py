"""Tests for Correct and Smooth: hand-computed one-step
diffusion on a two-node graph, the limiting cases (alpha = 0,
steps = 0, isolated node, no known labels), the anchor reset
in the Smooth stage, clipping, validation of bad inputs, and
the headline integration claim: C&S applied to the feature-only
MLP's scores, using training labels only, lifts validation AUC
to the repo's ceiling on the seed-7 dense ring."""

from gnn_fraud import (
    Graph,
    GraphGenerator,
    correct_and_smooth,
    correct_scores,
    evaluate_all,
    smooth_scores,
    train_mlp,
    train_val_split,
)


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def _two_node():
    g = Graph(2, 1)
    g.add_edge(0, 1)
    return g


def test_correct_one_step_hand_computed():
    # Residuals are [0.8, 0.0]; one undamped round replaces each
    # entry with the mean over the node and its neighbour, 0.4,
    # so the corrected scores are [0.2 + 0.4, 0.5 + 0.4].
    out = correct_scores(_two_node(), [0.2, 0.5], {0: 1},
                         alpha=1.0, steps=1)
    approx(out[0], 0.6)
    approx(out[1], 0.9)


def test_smooth_one_step_hand_computed():
    # Anchoring resets node 0 to its label, giving [1.0, 0.5];
    # one undamped round averages the pair to 0.75 at both ends.
    out = smooth_scores(_two_node(), [0.2, 0.5], {0: 1},
                        alpha=1.0, steps=1)
    approx(out[0], 0.75)
    approx(out[1], 0.75)


def test_correct_alpha_zero_sets_labeled_node_to_its_label():
    # No propagation happens, so the diffused residual is the
    # residual itself: the labeled node lands exactly on its
    # label and its neighbour is untouched.
    out = correct_scores(_two_node(), [0.2, 0.5], {0: 1},
                         alpha=0.0, steps=20)
    approx(out[0], 1.0)
    approx(out[1], 0.5)


def test_smooth_steps_zero_keeps_only_the_anchor_reset():
    out = smooth_scores(_two_node(), [0.2, 0.5], {0: 1},
                        alpha=0.8, steps=0)
    approx(out[0], 1.0)
    approx(out[1], 0.5)


def test_no_known_labels_correct_is_identity_smooth_still_diffuses():
    g = _two_node()
    # With no labeled node every residual is 0, so Correct has
    # nothing to spread and returns the base scores exactly.
    out = correct_scores(g, [0.2, 0.5], {}, alpha=0.8, steps=20)
    approx(out[0], 0.2)
    approx(out[1], 0.5)
    # Smooth with no anchors is plain damped diffusion of the
    # scores themselves: the pair move toward each other.
    sm = smooth_scores(g, [0.2, 0.5], {}, alpha=0.8, steps=20)
    assert 0.2 < sm[0] < sm[1] < 0.5


def test_isolated_node_keeps_its_base_score():
    # Node 2 has no edges: its propagation row is the one-hot on
    # itself, so neither stage can move its score, while its
    # wired neighbours still move.
    g = Graph(3, 1)
    g.add_edge(0, 1)
    out = correct_and_smooth(g, [0.2, 0.5, 0.3], {0: 1})
    approx(out[2], 0.3)
    assert out[0] > 0.2
    assert out[1] > 0.5


def test_correct_clips_at_one():
    # A confirmed fraudster scored at 0.9 has residual 0.1, and
    # its neighbour scored at 0.95 receives a positive spread;
    # the result must clip at 1.0, never exceed it.
    out = correct_scores(_two_node(), [0.9, 0.95], {0: 1},
                         alpha=1.0, steps=5)
    for p in out:
        assert 0.0 <= p <= 1.0
    approx(out[1], 1.0)


def test_correct_pulls_neighbour_of_confirmed_fraud_up():
    g = _two_node()
    out = correct_scores(g, [0.2, 0.2], {0: 1}, alpha=0.8, steps=20)
    assert out[1] > 0.2
    # The labeled node itself ends closest to its label.
    assert out[0] > out[1]


def test_smooth_pulls_neighbour_toward_anchor_label():
    g = _two_node()
    out = smooth_scores(g, [0.9, 0.9], {0: 0}, alpha=0.8, steps=20)
    assert out[1] < 0.9
    assert out[0] < out[1]


def test_bad_inputs_rejected():
    g = _two_node()
    cases = [
        lambda: correct_scores(g, [0.2], {0: 1}),
        lambda: correct_scores(g, [0.2, 1.5], {0: 1}),
        lambda: correct_scores(g, [0.2, 0.5], {0: 2}),
        lambda: correct_scores(g, [0.2, 0.5], {0: 1}, alpha=1.5),
        lambda: smooth_scores(g, [0.2, 0.5], {0: 1}, steps=-1),
    ]
    for fn in cases:
        try:
            fn()
        except (ValueError, IndexError):
            continue
        raise AssertionError("expected a validation error")
    try:
        correct_scores(g, [0.2, 0.5], {7: 1})
    except IndexError:
        return
    raise AssertionError("expected IndexError for out-of-range label node")


def test_correct_and_smooth_is_the_composition():
    g = _two_node()
    scores = [0.2, 0.5]
    known = {0: 1}
    staged = smooth_scores(
        g, correct_scores(g, scores, known, alpha=0.5, steps=3),
        known, alpha=0.5, steps=3)
    direct = correct_and_smooth(g, scores, known,
                                alpha_correct=0.5, steps_correct=3,
                                alpha_smooth=0.5, steps_smooth=3)
    for a, b in zip(staged, direct):
        approx(a, b)


def test_cs_lifts_mlp_scores_on_dense_ring():
    # Training labels only: the known set is exactly the train
    # split, so the validation AUC below is not label leakage.
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                       hidden_dim=16, lr=0.3, batch_size=32,
                       epochs=150, patience=20, seed=7)
    base = mlp.predict(g.features)
    known = {v: g.labels[v] for v in train_ids}
    fixed = correct_and_smooth(g, base, known)
    y_true = [g.labels[v] for v in val_ids]
    base_auc = evaluate_all(y_true, [base[v] for v in val_ids])["auc"]
    fixed_auc = evaluate_all(y_true, [fixed[v] for v in val_ids])["auc"]
    assert fixed_auc >= 0.99
    assert fixed_auc > base_auc

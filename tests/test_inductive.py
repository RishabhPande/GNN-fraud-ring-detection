"""Tests for inductive evaluation: the frozen-threshold protocol, holdout
scoring mechanics, and the headline claim that GraphSAGE's ranking
transfers to a freshly generated graph (while its calibration does not).
"""

import pytest

from gnn_fraud import (
    Graph,
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    inductive_evaluate,
    score_new_nodes,
    train_mlp,
    train_sage,
    train_val_split,
)


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def _tiny_graph():
    g = Graph(4, 2)
    g.set_label(0, 0)
    g.set_label(1, 1)
    g.set_label(2, -1)  # unlabeled node type (e.g. a device in a hetero graph)
    g.set_label(3, 0)
    return g


def _dummy_predict(graph, ids):
    return {v: 0.1 * v for v in ids}


# ---------------------------------------------------------------------------
# holdout scoring mechanics
# ---------------------------------------------------------------------------


def test_score_new_nodes_defaults_to_labeled():
    g = _tiny_graph()
    y_true, y_score, ids = score_new_nodes(_dummy_predict, g)
    assert ids == [0, 1, 3]  # the -1 node is skipped
    assert y_true == [0, 1, 0]
    approx(y_score[0], 0.0)
    approx(y_score[1], 0.1)
    approx(y_score[2], 0.3)


def test_score_new_nodes_explicit_ids_keep_order():
    g = _tiny_graph()
    y_true, y_score, ids = score_new_nodes(_dummy_predict, g, node_ids=[3, 1])
    assert ids == [3, 1]
    assert y_true == [0, 1]
    approx(y_score[0], 0.3)
    approx(y_score[1], 0.1)


def test_score_new_nodes_no_labeled_raises():
    g = Graph(3, 2)
    for v in range(3):
        g.set_label(v, -1)
    with pytest.raises(ValueError, match="no labeled nodes"):
        score_new_nodes(_dummy_predict, g)


# ---------------------------------------------------------------------------
# frozen-threshold protocol
# ---------------------------------------------------------------------------


def test_inductive_evaluate_frozen_threshold_matches_evaluate_all():
    g = _tiny_graph()
    m = inductive_evaluate(_dummy_predict, g, threshold=0.25)
    ref = evaluate_all([0, 1, 0], [0.0, 0.1, 0.3], threshold=0.25)
    approx(m["precision"], ref["precision"])
    approx(m["recall"], ref["recall"])
    approx(m["f1"], ref["f1"])
    approx(m["auc"], ref["auc"])
    approx(m["threshold"], 0.25)
    assert m["threshold_frozen"] is True
    assert m["n_scored"] == 3
    assert m["n_positive"] == 1


def test_inductive_evaluate_tunes_threshold_when_unset():
    g = _tiny_graph()
    m = inductive_evaluate(_dummy_predict, g, threshold=None)
    best_thr, _ = best_threshold_f1([0, 1, 0], [0.0, 0.1, 0.3])
    approx(m["threshold"], best_thr)
    assert m["threshold_frozen"] is False


# ---------------------------------------------------------------------------
# headline: ranking transfers to a fresh graph, calibration does not
# ---------------------------------------------------------------------------


def _trained_on_a():
    g_a = GraphGenerator(seed=7).generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g_a.labels, val_frac=0.25, seed=7)
    sage, _ = train_sage(g_a, train_ids, val_ids, hidden_dim=16,
                         k1=5, k2=5, lr=0.1, batch_size=64,
                         epochs=60, patience=12, seed=7)
    y_val = [g_a.labels[v] for v in val_ids]
    thr, _ = best_threshold_f1(
        y_val, [sage.predict(g_a, [v])[v] for v in val_ids])
    return sage, thr, g_a


def test_inductive_ranking_transfers_to_fresh_graph():
    # Pinned by seed -- every RNG involved is seeded, so these numbers are
    # exact, not statistical. Re-run examples/inductive_example.py if they
    # ever drift.
    sage, frozen_thr, g_a = _trained_on_a()
    # Exact value, pinned by seed (examples/inductive_example.py prints it
    # rounded to 2 decimals as 0.28).
    approx(frozen_thr, 0.2837065558789484)

    g_b = GraphGenerator(seed=99).generate(n=200, ring_size=18)
    m = inductive_evaluate(sage.predict, g_b, threshold=frozen_thr)

    # The honest inductive protocol: frozen threshold, no peeking at B.
    assert m["threshold_frozen"] is True
    approx(m["threshold"], frozen_thr)
    assert m["n_scored"] == 200
    assert m["n_positive"] == 18
    # Ranking transfers: inductive AUC stays strong ...
    assert m["auc"] >= 0.80, f"inductive auc {m['auc']}"
    # ... but calibration does not: the frozen threshold's F1 collapses.
    assert m["f1"] <= 0.15, f"frozen-thr f1 {m['f1']} -- expected collapse"


def test_retuning_on_holdout_recovers_f1_but_is_optimistic():
    sage, frozen_thr, _ = _trained_on_a()
    g_b = GraphGenerator(seed=99).generate(n=200, ring_size=18)
    frozen = inductive_evaluate(sage.predict, g_b, threshold=frozen_thr)
    retuned = inductive_evaluate(sage.predict, g_b, threshold=None)

    assert retuned["threshold_frozen"] is False
    # Same ranking (AUC identical), better operating point.
    approx(retuned["auc"], frozen["auc"])
    assert retuned["f1"] >= 0.40, f"retuned f1 {retuned['f1']}"
    assert retuned["f1"] > frozen["f1"]


def test_inductive_graph_signal_beats_feature_signal():
    # Both models jump to the fresh graph; the GNN's structural signal
    # transfers better than the MLP's per-account tells.
    sage, frozen_thr, g_a = _trained_on_a()
    train_ids, val_ids = train_val_split(g_a.labels, val_frac=0.25, seed=7)
    mlp, _ = train_mlp(g_a.features, g_a.labels, train_ids, val_ids,
                       hidden_dim=16, lr=0.3, batch_size=32,
                       epochs=150, patience=20, seed=7)

    def mlp_predict(g, ids):
        return {v: p for v, p in
                zip(ids, mlp.predict([g.features[v] for v in ids]))}

    y_val = [g_a.labels[v] for v in val_ids]
    mlp_thr, _ = best_threshold_f1(
        y_val, mlp.predict([g_a.features[v] for v in val_ids]))

    g_b = GraphGenerator(seed=99).generate(n=200, ring_size=18)
    sage_m = inductive_evaluate(sage.predict, g_b, threshold=frozen_thr)
    mlp_m = inductive_evaluate(mlp_predict, g_b, threshold=mlp_thr)
    assert sage_m["auc"] > mlp_m["auc"] + 0.05, \
        f"sage {sage_m['auc']} vs mlp {mlp_m['auc']}"


def test_single_unseen_node_scores_without_retraining():
    # The core inductive property: a trained model scores a node it never
    # saw, using only learned aggregation weights -- no per-node params.
    sage, _, _ = _trained_on_a()
    g_b = GraphGenerator(seed=1234).generate(n=50, ring_size=6)
    y_true, y_score, ids = score_new_nodes(sage.predict, g_b, node_ids=[0])
    assert ids == [0]
    assert y_true == [g_b.labels[0]]
    assert 0.0 <= y_score[0] <= 1.0

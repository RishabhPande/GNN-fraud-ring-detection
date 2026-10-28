"""Seed-robustness tests: GIN and LSTM must hold up on the seed where
mean aggregation collapses hardest.

Seed 9 is the worst mean seed in the sweep (AUC 0.539, F1 0.240 --
see examples/seed_robustness_example.py): a degenerate checkpoint
that looks fine on validation BCE but ranks backwards. If GIN's sum
or the LSTM aggregator shared that fragility, seed 9 is where it
would show. Both held AUC 1.000 there, so the bar here is the repo's
usual headline threshold, 0.90, not a re-fit to the measured 1.000.
"""

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_gin,
    train_sage,
    train_val_split,
)


def _seed9_split():
    gen = GraphGenerator(seed=9)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=9)
    return g, train_ids, val_ids


def test_gin_holds_on_mean_collapse_seed():
    g, train_ids, val_ids = _seed9_split()
    model, _ = train_gin(g, train_ids, val_ids, hidden_dim=16,
                         lr=0.1, batch_size=64, epochs=60,
                         patience=12, seed=9)
    y_true = [g.labels[v] for v in val_ids]
    scores = [model.predict(g, [v])[v] for v in val_ids]
    _, met = best_threshold_f1(y_true, scores)
    auc = evaluate_all(y_true, scores)["auc"]
    assert auc >= 0.90, f"GIN seed-9 AUC {auc:.3f} below 0.90"
    assert met["f1"] >= 0.90, f"GIN seed-9 F1 {met['f1']:.3f} below 0.90"


def test_lstm_holds_on_mean_collapse_seed():
    g, train_ids, val_ids = _seed9_split()
    model, _ = train_sage(g, train_ids, val_ids, hidden_dim=16,
                          k1=5, k2=5, lr=0.1, batch_size=64,
                          epochs=60, patience=12, seed=9,
                          aggregator="lstm")
    y_true = [g.labels[v] for v in val_ids]
    scores = [model.predict(g, [v])[v] for v in val_ids]
    _, met = best_threshold_f1(y_true, scores)
    auc = evaluate_all(y_true, scores)["auc"]
    assert auc >= 0.90, f"LSTM seed-9 AUC {auc:.3f} below 0.90"
    assert met["f1"] >= 0.90, f"LSTM seed-9 F1 {met['f1']:.3f} below 0.90"

"""Seed-robustness tests: GIN and LSTM must hold up on the seed where
mean aggregation collapses hardest.

Seed 9 is the worst mean seed in the sweep (AUC 0.539, F1 0.240 --
see examples/seed_robustness_example.py): a degenerate checkpoint
that looks fine on validation BCE but ranks backwards. If GIN's sum
or the LSTM aggregator shared that fragility, seed 9 is where it
would show. Both held AUC 1.000 there, so the bar here is the repo's
usual headline threshold, 0.90, not a re-fit to the measured 1.000.

The extended sweep (panel 2 of the example) found the same split in
the later models: PNA and H2GCN hold AUC/F1 1.000 on seed 9, and
ChebNet (AUC 0.944) and JK-GCN (0.911) clear the AUC bar with F1
below it (0.750 and 0.667), so their tests assert AUC only. The rest
of the family does not clear 0.90 on seed 9 (GCN 0.889, APPNP 0.894,
TAGCN 0.894, SIGN 0.883, GPR-GNN 0.822, MixHop 0.867, GATv2 0.689,
GCNII-4 0.506, SGC 0.411) and earns no passing test here: the sweep
table and the README report those numbers instead of a lowered bar.
"""

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_cheb,
    train_gin,
    train_h2gcn,
    train_jk,
    train_pna,
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


def _seed9_auc_f1(model_fn, **kwargs):
    """Train one model on the seed-9 split and return (AUC, F1)."""
    g, train_ids, val_ids = _seed9_split()
    model, _ = model_fn(g, train_ids, val_ids, seed=9, **kwargs)
    y_true = [g.labels[v] for v in val_ids]
    scores = [model.predict(g, [v])[v] for v in val_ids]
    _, met = best_threshold_f1(y_true, scores)
    return evaluate_all(y_true, scores)["auc"], met["f1"]


def test_pna_holds_on_mean_collapse_seed():
    auc, f1 = _seed9_auc_f1(train_pna, hidden_dim=16, lr=0.1,
                            batch_size=64, epochs=60, patience=12)
    assert auc >= 0.90, f"PNA seed-9 AUC {auc:.3f} below 0.90"
    assert f1 >= 0.90, f"PNA seed-9 F1 {f1:.3f} below 0.90"


def test_h2gcn_holds_on_mean_collapse_seed():
    auc, f1 = _seed9_auc_f1(train_h2gcn, hidden_dim=16, lr=0.5,
                            batch_size=64, epochs=100, patience=25)
    assert auc >= 0.90, f"H2GCN seed-9 AUC {auc:.3f} below 0.90"
    assert f1 >= 0.90, f"H2GCN seed-9 F1 {f1:.3f} below 0.90"


def test_jk_holds_auc_on_mean_collapse_seed():
    # AUC clears the bar (0.911 measured); F1 does not (0.667), so
    # only AUC is asserted here. See the module docstring.
    auc, _ = _seed9_auc_f1(train_jk, hidden_dim=16, lr=0.5,
                           batch_size=64, epochs=100, patience=25)
    assert auc >= 0.90, f"JK-GCN seed-9 AUC {auc:.3f} below 0.90"


def test_chebnet_holds_auc_on_mean_collapse_seed():
    # AUC clears the bar (0.944 measured); F1 does not (0.750), so
    # only AUC is asserted here. See the module docstring.
    auc, _ = _seed9_auc_f1(train_cheb, hidden_dim=16, K=2, lr=0.5,
                           batch_size=64, epochs=100, patience=25)
    assert auc >= 0.90, f"ChebNet seed-9 AUC {auc:.3f} below 0.90"

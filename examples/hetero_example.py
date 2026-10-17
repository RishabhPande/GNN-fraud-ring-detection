"""Heterogeneous example: a relational GraphSAGE over accounts, devices,
and merchants vs an MLP on account features alone.

The generator plants a ring whose account features are camouflaged -- but
the ring shares devices and merchants. The MLP sees only account
features; the relational model learns per-relation message weights, so a
shared device counts for more than a one-off counterparty."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    HeteroGenerator,
    best_threshold_f1,
    evaluate_all,
    train_hetero,
    train_mlp,
    train_val_split,
)

gen = HeteroGenerator(seed=7)
g = gen.generate(n_accounts=200, ring_size=18)
ring = gen.ring_nodes(g)
shared_devices = gen.ring_devices(g)
counts = g.type_counts()
print(f"generated {counts[0]} accounts, {counts[1]} devices, "
      f"{counts[2]} merchants; {len(ring)} planted ring members")
print(f"ring shares {len(shared_devices)} devices "
      f"(each used by >= 3 ring members)")

accounts = g.account_ids()
labels = [g.labels[v] for v in accounts]
tr, va = train_val_split(labels, val_frac=0.25, seed=7)
train_ids = [accounts[i] for i in tr]
val_ids = [accounts[i] for i in va]

print("\ntraining relational GraphSAGE (typed neighbor sampling)...")
hetero, hetero_hist = train_hetero(g, train_ids, val_ids, hidden_dim=16,
                                   k1=5, k2=5, lr=0.1, batch_size=64,
                                   epochs=60, patience=12, seed=7)
print(f"  epochs run: {len(hetero_hist['train_loss'])}, "
      f"best val loss: {min(hetero_hist['val_loss']):.4f}")

print("training MLP baseline (account features only)...")
mlp, mlp_hist = train_mlp([g.features[v] for v in accounts], labels, tr, va,
                          hidden_dim=16, lr=0.3, batch_size=32,
                          epochs=150, patience=20, seed=7)
print(f"  epochs run: {len(mlp_hist['train_loss'])}, "
      f"best val loss: {min(mlp_hist['val_loss']):.4f}")

y_true = [g.labels[v] for v in val_ids]
hetero_scores = [hetero.predict(g, [v])[v] for v in val_ids]
mlp_scores = mlp.predict([g.features[v] for v in val_ids])

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<34}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
for name, scores in [("Relational GraphSAGE (hetero)", hetero_scores),
                     ("MLP (account features only)", mlp_scores)]:
    thr, m = best_threshold_f1(y_true, scores)
    auc = evaluate_all(y_true, scores)["auc"]
    print(f"  {name:<34}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")
print("\nThe MLP sees camouflaged account features one row at a time; the "
      "relational model pools evidence across typed relations -- shared "
      "devices and merchants light up the ring.")

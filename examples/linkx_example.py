"""LINKX vs GraphSAGE vs the feature-only baseline on the same
fraud-ring graph, plus the head's branch split.

LINKX never message-passes: a feature MLP and a structure MLP
over the node's own adjacency row are concatenated at the head.
The adjacency row names *which* accounts a node is wired to, so
the structure branch learns the wiring as a feature. The branch
norms below read the split back off the trained head.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_linkx,
    train_mlp,
    train_sage,
    train_val_split,
)

gen = GraphGenerator(seed=7)
g = gen.generate(n=200, ring_size=18)
ring = set(gen.ring_nodes(g))
print(f"generated {g.n} accounts, {g.edge_count()} edges, "
      f"{len(ring)} planted ring members")

train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
y_true = [g.labels[v] for v in val_ids]

print("\ntraining LINKX (feature MLP + adjacency-row MLP)...")
linkx, linkx_hist = train_linkx(g, train_ids, val_ids, hidden_dim=16,
                                lr=0.5, batch_size=64,
                                epochs=100, patience=25, seed=7)
print(f"  epochs run: {len(linkx_hist['train_loss'])}, "
      f"best val loss: {min(linkx_hist['val_loss']):.4f}")

print("training GraphSAGE (2-layer, mean agg)...")
sage, _ = train_sage(g, train_ids, val_ids, hidden_dim=16,
                     k1=5, k2=5, lr=0.1, batch_size=64,
                     epochs=60, patience=12, seed=7)

print("training MLP baseline (node features only)...")
mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                   hidden_dim=16, lr=0.3, batch_size=32,
                   epochs=150, patience=20, seed=7)

scores = {
    "LINKX (features + adjacency row)": [linkx.predict(g, [v])[v] for v in val_ids],
    "GraphSAGE (2-layer, mean agg)": [sage.predict(g, [v])[v] for v in val_ids],
    "MLP (features only)": mlp.predict([g.features[v] for v in val_ids]),
}

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<34}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
for name, s in scores.items():
    thr, m = best_threshold_f1(y_true, s)
    auc = evaluate_all(y_true, s)["auc"]
    print(f"  {name:<34}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")

feat_norm, adj_norm = linkx.branch_norms()
print(f"\nhead branch norms: feature block {feat_norm:.3f}, "
      f"structure block {adj_norm:.3f}")
print("\nThe two views never mix before the head, so the split "
      "above is the model's own accounting of how much of the "
      "decision is the profile and how much is the wiring.")

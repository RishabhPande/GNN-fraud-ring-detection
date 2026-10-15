"""Quickstart: generate a transaction graph, train GraphSAGE and an
MLP-on-features baseline, and compare. The ring's features are camouflaged,
so the MLP should lag -- the GNN wins on structure alone."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_mlp,
    train_sage,
    train_val_split,
)

gen = GraphGenerator(seed=7)
g = gen.generate(n=200, ring_size=18)
ring = gen.ring_nodes(g)
print(f"generated {g.n} accounts, {g.edge_count()} edges, "
      f"{len(ring)} planted ring members")
print(f"ring subgraph density: {g.density(ring):.2f} "
      f"(rest of graph is far sparser)")

train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)

print("\ntraining GraphSAGE (neighbor-sampled mini-batches)...")
sage, sage_hist = train_sage(g, train_ids, val_ids, hidden_dim=16,
                             k1=5, k2=5, lr=0.1, batch_size=64,
                             epochs=60, patience=12, seed=7)
print(f"  epochs run: {len(sage_hist['train_loss'])}, "
      f"best val loss: {min(sage_hist['val_loss']):.4f}")

print("training MLP baseline (node features only)...")
mlp, mlp_hist = train_mlp(g.features, g.labels, train_ids, val_ids,
                          hidden_dim=16, lr=0.3, batch_size=32,
                          epochs=150, patience=20, seed=7)
print(f"  epochs run: {len(mlp_hist['train_loss'])}, "
      f"best val loss: {min(mlp_hist['val_loss']):.4f}")

# Score on held-out validation nodes. Fraud scores need a tuned operating
# point under class imbalance, so pick each model's best-F1 threshold.
y_true = [g.labels[v] for v in val_ids]
sage_scores = [sage.predict(g, [v])[v] for v in val_ids]
mlp_scores = mlp.predict([g.features[v] for v in val_ids])

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<28}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
for name, scores in [("GraphSAGE (2-layer, mean agg)", sage_scores),
                     ("MLP (features only)", mlp_scores)]:
    thr, m = best_threshold_f1(y_true, scores)
    auc = evaluate_all(y_true, scores)["auc"]
    print(f"  {name:<28}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")
print("\nThe GNN pools weak tells over the ring's dense neighborhood; "
      "the MLP sees them one noisy account at a time.")

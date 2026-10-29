"""SGC vs GCN vs the feature-only baseline on the same fraud-ring graph.

SGC is the GCN with its hidden layer deleted: propagate the features
twice through the normalized adjacency, then logistic regression. If it
matches the GCN, the ring's signal lives in the propagated features
themselves, not in a learned nonlinear transform -- and the cheaper
model is the right one to ship. If it lags, the gap measures exactly
what the hidden layer buys.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_gcn,
    train_mlp,
    train_sgc,
    train_val_split,
)

gen = GraphGenerator(seed=7)
g = gen.generate(n=200, ring_size=18)
ring = set(gen.ring_nodes(g))
print(f"generated {g.n} accounts, {g.edge_count()} edges, "
      f"{len(ring)} planted ring members")

train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
y_true = [g.labels[v] for v in val_ids]

print("\ntraining SGC (2-hop propagation + logistic regression)...")
sgc, sgc_hist = train_sgc(g, train_ids, val_ids, hops=2,
                          lr=0.5, batch_size=64, epochs=100,
                          patience=25, seed=7)
print(f"  epochs run: {len(sgc_hist['train_loss'])}, "
      f"best val loss: {min(sgc_hist['val_loss']):.4f}")

print("training GCN (2-layer, symmetric normalization)...")
gcn, _ = train_gcn(g, train_ids, val_ids, hidden_dim=16,
                   lr=0.5, batch_size=64, epochs=100,
                   patience=25, seed=7)

print("training MLP baseline (node features only)...")
mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                   hidden_dim=16, lr=0.3, batch_size=32,
                   epochs=150, patience=20, seed=7)

scores = {
    "SGC (2-hop propagation)": [sgc.predict(g, [v])[v] for v in val_ids],
    "GCN (2-layer, normalized)": [gcn.predict(g, [v])[v] for v in val_ids],
    "MLP (features only)": mlp.predict([g.features[v] for v in val_ids]),
}

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<28}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
for name, s in scores.items():
    thr, m = best_threshold_f1(y_true, s)
    auc = evaluate_all(y_true, s)["auc"]
    print(f"  {name:<28}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")
print("\nSGC has no hidden layer to hide behind: whatever it scores, the "
      "propagated features alone earned it.")

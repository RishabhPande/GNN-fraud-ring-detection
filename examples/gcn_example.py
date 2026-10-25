"""GCN vs GraphSAGE on the same fraud-ring graph.

Honest headline: on the dense ring, GCN lands just under SAGE's mean
model (AUC ~0.94 vs 0.98) -- the degree normalization that defines GCN
also dilutes the ring's own signal, since ring members are the
highest-degree nodes in the graph and their messages get down-weighted
the most. GCN still crushes the feature-only baseline, and the gap is a
real architectural difference, not a tuning artifact: both models train
at their own best learning rate here.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_gcn,
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

print("\ntraining GCN (2-layer, symmetric normalization)...")
# GCN needs a larger step than SAGE: the normalized aggregation shrinks
# gradients, and lr=0.1 underfits badly (AUC ~0.48-0.71 vs 0.94 at 0.5).
gcn, gcn_hist = train_gcn(g, train_ids, val_ids, hidden_dim=16,
                           lr=0.5, batch_size=64, epochs=100,
                           patience=25, seed=7)
print(f"  epochs run: {len(gcn_hist['train_loss'])}, "
      f"best val loss: {min(gcn_hist['val_loss']):.4f}")

print("training GraphSAGE (2-layer, mean agg)...")
sage, _ = train_sage(g, train_ids, val_ids, hidden_dim=16,
                     k1=5, k2=5, lr=0.1, batch_size=64,
                     epochs=60, patience=12, seed=7)

scores = {
    "GCN (2-layer, normalized)": [gcn.predict(g, [v])[v] for v in val_ids],
    "GraphSAGE (2-layer, mean agg)": [sage.predict(g, [v])[v]
                                      for v in val_ids],
}

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<28}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
for name, s in scores.items():
    thr, m = best_threshold_f1(y_true, s)
    auc = evaluate_all(y_true, s)["auc"]
    print(f"  {name:<28}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")
print("\nGCN's normalization is a double-edged sword on fraud graphs: it "
      "stops hubs from shouting down the neighborhood, but ring members "
      "are the hubs, so it also turns down the ring's own tell.")

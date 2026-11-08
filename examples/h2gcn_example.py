"""H2GCN vs the plain GCN it extends, SAGE mean, and the feature-only
baseline on the same fraud-ring graph.

H2GCN's bet is that a fraud graph is heterophilic at the boundary:
a ring member's neighbors are a mix of fellow ring members and
ordinary accounts, so averaging the node into its neighborhood (the
GCN's move) dilutes the pocket signal, while keeping the ego
embedding in its own block and reading the 1-hop and 2-hop
neighborhoods separately lets the model weight "I look odd"
differently from "my counterparties look odd". The head-weight
norms per layer block at the end show which view the head actually
leaned on: the embedding view (r0), the 1-hop view (r1), or the
2-hop view (r2).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_gcn,
    train_h2gcn,
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

print("\ntraining H2GCN (ego/neighbor separation, 1- and 2-hop blocks, "
      "all-layer concat)...")
h2, h2_hist = train_h2gcn(g, train_ids, val_ids, hidden_dim=16,
                          lr=0.5, batch_size=64, epochs=100,
                          patience=25, seed=7)
print(f"  epochs run: {len(h2_hist['train_loss'])}, "
      f"best val loss: {min(h2_hist['val_loss']):.4f}")

print("training GCN (2-layer, normalized)...")
gcn, _ = train_gcn(g, train_ids, val_ids, hidden_dim=16,
                   lr=0.5, batch_size=64, epochs=100,
                   patience=25, seed=7)

print("training GraphSAGE (2-layer, mean agg)...")
sage, _ = train_sage(g, train_ids, val_ids, hidden_dim=16,
                     lr=0.1, batch_size=64, epochs=100,
                     patience=25, seed=7)

print("training MLP baseline (node features only)...")
mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                   hidden_dim=16, lr=0.3, batch_size=32,
                   epochs=150, patience=20, seed=7)

scores = {
    "H2GCN (ego + 1/2-hop, concat)": [h2.predict(g, [v])[v]
                                      for v in val_ids],
    "GCN (2-layer, normalized)": [gcn.predict(g, [v])[v]
                                  for v in val_ids],
    "GraphSAGE (2-layer, mean agg)": [sage.predict(g, [v])[v]
                                      for v in val_ids],
    "MLP (features only)": mlp.predict([g.features[v] for v in val_ids]),
}

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<32}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
for name, s in scores.items():
    thr, m = best_threshold_f1(y_true, s)
    auc = evaluate_all(y_true, s)["auc"]
    print(f"  {name:<32}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")

norms = h2.layer_block_norms()
print("\nH2GCN head weight norm per final-representation block:")
print(f"  r0 (embedding view): {norms[0]:.3f}")
print(f"  r1 (1-hop view):     {norms[1]:.3f}")
print(f"  r2 (2-hop view):     {norms[2]:.3f}")
print("\nThe ego/neighbor separation is only worth its extra weights "
      "if the head actually uses the separate blocks; a head that "
      "leans entirely on r0 has learned to ignore the graph.")

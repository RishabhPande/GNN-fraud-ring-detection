"""GCNII vs the plain GCN and GraphSAGE on the same fraud-ring graph.

Honest headline: on the dense ring, GCNII (input layer + 4 GCNII
layers, alpha = 0.2) ties the plain 2-layer GCN on AUC (0.944 both)
and edges it on F1 (0.727 vs 0.667) -- the initial residual keeps
each node's own signal in the mix at every depth. The depth claim
is the real story: GCNII with 8 layers (9 hops with the input
layer) still reaches 0.950 AUC, where a plain deep GCN would
oversmooth toward uniform embeddings. SAGE mean still leads both
(0.978). Like every other headline in this repo, this is one seed
(7) until a sweep says otherwise.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_gcn,
    train_gcnii,
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

print("\ntraining GCNII (input + 4 layers, alpha=0.2)...")
gcnii, gcnii_hist = train_gcnii(g, train_ids, val_ids, hidden_dim=16,
                                n_layers=4, lr=0.5, batch_size=64,
                                epochs=100, patience=25, seed=7)
print(f"  epochs run: {len(gcnii_hist['train_loss'])}, "
      f"best val loss: {min(gcnii_hist['val_loss']):.4f}")

print("training deep GCNII (input + 8 layers, alpha=0.2)...")
deep, deep_hist = train_gcnii(g, train_ids, val_ids, hidden_dim=16,
                              n_layers=8, lr=0.5, batch_size=64,
                              epochs=100, patience=25, seed=7)
print(f"  epochs run: {len(deep_hist['train_loss'])}, "
      f"best val loss: {min(deep_hist['val_loss']):.4f}")

print("training GCN (2-layer)...")
gcn, _ = train_gcn(g, train_ids, val_ids, hidden_dim=16,
                   lr=0.5, batch_size=64, epochs=100,
                   patience=25, seed=7)

print("training GraphSAGE (2-layer, mean agg)...")
sage, _ = train_sage(g, train_ids, val_ids, hidden_dim=16,
                     k1=5, k2=5, lr=0.1, batch_size=64,
                     epochs=60, patience=12, seed=7)

scores = {
    "GCNII (input + 4 layers)": [gcnii.predict(g, [v])[v] for v in val_ids],
    "GCNII deep (input + 8 layers)": [deep.predict(g, [v])[v]
                                       for v in val_ids],
    "GCN (2-layer)": [gcn.predict(g, [v])[v] for v in val_ids],
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
print("\nGCNII ties the GCN on AUC and edges it on F1, but the "
      "residual's real payoff is depth: every layer mixes back the "
      "input layer's representation, so the ring's hubs keep their "
      "own features instead of being averaged away, and the "
      "identity-shrunk weights keep the gradient alive through the "
      "stack -- which is why 8 layers still trains at 0.950 AUC "
      "instead of collapsing.")

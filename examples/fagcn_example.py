"""FAGCN vs GPR-GNN and the MLP baseline on the same fraud-ring graph.

FAGCN (Bo et al.) splits every propagation step into a low-pass
channel (the neighbour mix, S H) and a high-pass channel (the
disagreement, H - S H) and learns one scalar weight per channel
per step. The question is whether the high-pass channel, which
plain GCN propagation throws away, earns anything on a graph
where ring members mostly agree with their neighbours and only
bridge accounts disagree.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_fagcn,
    train_gpr,
    train_mlp,
    train_val_split,
)

gen = GraphGenerator(seed=7)
g = gen.generate(n=200, ring_size=18)
ring = set(gen.ring_nodes(g))
print(f"generated {g.n} accounts, {g.edge_count()} edges, "
      f"{len(ring)} planted ring members")

train_ids, val_ids = train_val_split(g.labels, seed=7)
y_true = [g.labels[v] for v in val_ids]

print("\ntraining FAGCN (2 frequency-adaptive steps)...")
model, hist = train_fagcn(g, train_ids, val_ids, hidden_dim=16, steps=2,
                          lr=0.25, batch_size=64, epochs=100,
                          patience=25, seed=7)
print(f"  epochs run: {len(hist['train_loss'])}, "
      f"best val loss: {min(hist['val_loss']):.4f}")
print(f"  learned alpha (low-pass):  {model.alpha:.3f}")
print(f"  learned beta (high-pass):  {model.beta:.3f}")

print("training GPR-GNN (learned gamma)...")
gpr, _ = train_gpr(g, train_ids, val_ids, hidden_dim=16, steps=10,
                   lr=0.25, batch_size=64, epochs=100, patience=25, seed=7)

print("training MLP (features only)...")
mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids, hidden_dim=16,
                   lr=0.1, batch_size=64, epochs=60, patience=12, seed=7)

scores = {
    "FAGCN (low+high-pass)": [model.predict(g, [v])[v] for v in val_ids],
    "GPR-GNN (learned gamma)": [gpr.predict(g, [v])[v] for v in val_ids],
    "MLP (features only)": mlp.predict([g.features[v] for v in val_ids]),
}

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<32}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
for name, s in scores.items():
    thr, m = best_threshold_f1(y_true, s)
    auc = evaluate_all(y_true, s)["auc"]
    print(f"  {name:<32}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")

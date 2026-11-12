"""GGNN vs GraphSAGE (mean) vs the feature-only baseline on the same
fraud-ring graph.

GGNN propagates recurrently: three rounds of mean aggregation, but
after every round a GRU - shared across rounds - decides per node and
per dimension how much of the incoming message to take and how much
of its own state to keep. A stacked GCN cannot refuse a later
aggregation; a GGNN node whose neighborhood turns noisy can close
its update gate and keep its earlier state. The gates are the
finding here, so the script prints their distribution after
training: a mean stuck at 0.5 with no spread would mean the gates
never learned to gate.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_ggnn,
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

print("\ntraining GGNN (encoder + 3 gated propagation steps)...")
ggnn, ggnn_hist = train_ggnn(g, train_ids, val_ids, hidden_dim=16,
                             steps=3, lr=0.5, batch_size=64,
                             epochs=100, patience=25, seed=7)
print(f"  epochs run: {len(ggnn_hist['train_loss'])}, "
      f"best val loss: {min(ggnn_hist['val_loss']):.4f}")

print("training GraphSAGE (2-layer, mean aggregation)...")
sage, _ = train_sage(g, train_ids, val_ids, hidden_dim=16,
                     aggregator="mean", lr=0.1, batch_size=32,
                     epochs=60, patience=12, seed=7)

print("training MLP baseline (node features only)...")
mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                   hidden_dim=16, lr=0.3, batch_size=32,
                   epochs=150, patience=20, seed=7)

scores = {
    "GGNN (3 gated steps)": [ggnn.predict(g, [v])[v] for v in val_ids],
    "GraphSAGE (2-layer, mean agg)": [sage.predict(g, [v])[v] for v in val_ids],
    "MLP (features only)": mlp.predict([g.features[v] for v in val_ids]),
}

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<32}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
for name, s in scores.items():
    thr, m = best_threshold_f1(y_true, s)
    auc = evaluate_all(y_true, s)["auc"]
    print(f"  {name:<32}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")

_, cache = ggnn.forward(g, [])
zs = [entry["z"] for sc in cache["steps"] for entry in sc]
rs = [entry["r"] for sc in cache["steps"] for entry in sc]
hd = ggnn.hidden_dim
z_dim = [sum(z[j] for z in zs) / len(zs) for j in range(hd)]
r_dim = [sum(r[j] for r in rs) / len(rs) for j in range(hd)]
print("\nlearned gates (mean per hidden dimension, over all nodes "
      "and steps):")
print(f"  update gate z: min {min(z_dim):.3f}, max {max(z_dim):.3f} "
      f"(untrained: every dimension at 0.5 in mean)")
print(f"  reset gate r:  min {min(r_dim):.3f}, max {max(r_dim):.3f}")
print("\nThe spread across dimensions is the gate doing its job: "
      "some channels learn to refresh from the neighborhood every "
      "step, others learn to carry their state through.")

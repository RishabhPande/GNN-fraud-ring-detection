"""DAGNN vs GPR-GNN vs the feature-only baseline on the same
fraud-ring graph.

GPR-GNN learns one global weight per propagation step. DAGNN
learns one projection vector instead, and each node takes its own
softmax mix over the steps: a ring member deep in the pocket and
a bridge account on its edge can weight the same propagated views
differently. If the trained attention stays near uniform for
everyone, the global mix was enough for this graph; if ring and
background accounts separate, per-node depth is doing real work.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_dagnn,
    train_gpr,
    train_mlp,
    train_val_split,
)

gen = GraphGenerator(seed=7)
g = gen.generate(n=200, ring_size=18)
ring = set(gen.ring_nodes(g))
print(f"generated {g.n} accounts, {g.edge_count()} edges, "
      f"{len(ring)} planted ring members")

train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
y_true = [g.labels[v] for v in val_ids]

print("\ntraining DAGNN (MLP + 10 propagation steps, per-node hop attention)...")
dag, dag_hist = train_dagnn(g, train_ids, val_ids, hidden_dim=16,
                            steps=10, lr=0.5,
                            batch_size=64, epochs=100,
                            patience=25, seed=7)
print(f"  epochs run: {len(dag_hist['train_loss'])}, "
      f"best val loss: {min(dag_hist['val_loss']):.4f}")

print("training GPR-GNN (MLP + 10 propagation steps, learned global gamma)...")
gpr, _ = train_gpr(g, train_ids, val_ids, hidden_dim=16,
                   steps=10, alpha=0.1, lr=0.5,
                   batch_size=64, epochs=100,
                   patience=25, seed=7)

print("training MLP baseline (node features only)...")
mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                   hidden_dim=16, lr=0.3, batch_size=32,
                   epochs=150, patience=20, seed=7)

scores = {
    "DAGNN (per-node hop attention)": [dag.predict(g, [v])[v]
                                       for v in val_ids],
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

print("\nmean hop attention by group (steps 0 to 10):")
for label, nodes in (("ring", sorted(ring)),
                     ("background", [v for v in range(g.n)
                                     if v not in ring])):
    means = [0.0] * 11
    for v in nodes:
        a = dag.attention_weights(g, v)
        for k in range(11):
            means[k] += a[k] / len(nodes)
    print(f"  {label:<11}" + "".join(f"{m:>6.3f}" for m in means))
print(f"  {'uniform':<11}" + "".join(f"{1 / 11:>6.3f}" for _ in range(11)))
print("\nDAGNN's question is whether those two rows differ: one global "
      "mix cannot serve both groups if they do.")

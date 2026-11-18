"""AGNN vs DAGNN vs the feature-only baseline on the same
fraud-ring graph.

APPNP, GPR-GNN, and DAGNN all propagate with the same fixed
normalized adjacency for every edge. AGNN re-weights every edge at
every step by the cosine similarity of its endpoints' current
hidden states, scaled by one learnable temperature beta per step.
The question is whether listening selectively beats the fixed mix
on a ring whose members are camouflaged to look like everyone
else, and what the trained betas and edge weights say.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_agnn,
    train_dagnn,
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

print("\ntraining AGNN (MLP encoder + 2 cosine-attention steps)...")
agnn, agnn_hist = train_agnn(g, train_ids, val_ids, hidden_dim=16,
                             steps=2, lr=0.5,
                             batch_size=64, epochs=100,
                             patience=25, seed=7)
print(f"  epochs run: {len(agnn_hist['train_loss'])}, "
      f"best val loss: {min(agnn_hist['val_loss']):.4f}")
print(f"  learned betas: {[round(b, 3) for b in agnn.betas]} "
      f"(init 1.0 each)")

print("training DAGNN (MLP + 10 steps, per-node hop attention)...")
dag, _ = train_dagnn(g, train_ids, val_ids, hidden_dim=16,
                     steps=10, lr=0.5,
                     batch_size=64, epochs=100,
                     patience=25, seed=7)

print("training MLP baseline (node features only)...")
mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                   hidden_dim=16, lr=0.3, batch_size=32,
                   epochs=150, patience=20, seed=7)

scores = {
    "AGNN (cosine-attention prop)": [agnn.predict(g, [v])[v]
                                     for v in val_ids],
    "DAGNN (per-node hop attention)": [dag.predict(g, [v])[v]
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

print("\nmean first-step attention weight by counterpart type:")
for label, nodes in (("ring", sorted(ring)),
                     ("background", [v for v in range(g.n)
                                     if v not in ring])):
    self_w, ring_w, bg_w, counts = 0.0, 0.0, 0.0, 0
    for v in nodes:
        nbrs = sorted(set(g.neighbors(v)) | {v})
        w = agnn.attention_weights(g, v)
        for a, u in zip(w, nbrs):
            counts += 1
            if u == v:
                self_w += a
            elif u in ring:
                ring_w += a
            else:
                bg_w += a
    n = len(nodes)
    print(f"  {label:<11} self {self_w / n:.3f}  ring {ring_w / n:.3f}  "
          f"background {bg_w / n:.3f}  (mean over {n} accounts)")
print("\nAGNN's question is whether ring members listen mostly to "
      "each other: if the ring and background rows look alike, "
      "cosine listening has nothing to grab under camouflage.")

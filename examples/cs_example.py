"""Correct and Smooth on top of the feature-only MLP baseline.

The MLP never sees the graph, so its scores carry only the
camouflaged per-account tells. C&S then post-processes those
scores with the training labels only: the Correct stage spreads
the model's residuals on the confirmed cases to their
counterparties, and the Smooth stage diffuses the corrected
scores with the confirmed cases anchored at their labels.
No validation label is ever passed in - that would be leakage.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    correct_and_smooth,
    correct_scores,
    evaluate_all,
    smooth_scores,
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
known = {v: g.labels[v] for v in train_ids}

print("\ntraining MLP baseline (node features only)...")
mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                   hidden_dim=16, lr=0.3, batch_size=32,
                   epochs=150, patience=20, seed=7)
base = mlp.predict(g.features)

variants = {
    "MLP (features only)": base,
    "MLP + Correct": correct_scores(g, base, known),
    "MLP + Smooth": smooth_scores(g, base, known),
    "MLP + Correct and Smooth": correct_and_smooth(g, base, known),
}

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<28}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
for name, s in variants.items():
    vals = [s[v] for v in val_ids]
    thr, m = best_threshold_f1(y_true, vals)
    auc = evaluate_all(y_true, vals)["auc"]
    print(f"  {name:<28}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")

print("\nThe jump comes from the confirmed cases, not from a "
      "better feature model: the ring members the MLP scored "
      "like background accounts sit next to confirmed ring "
      "members, and the two diffusion stages carry that "
      "information the few hops it needs to travel.")

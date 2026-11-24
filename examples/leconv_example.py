"""LEConv vs GraphConv vs the feature-only baseline on the same
fraud-ring graph, plus a difference-signal readout.

LEConv aggregates signed differences, deg(v) * h_v minus the
neighbour sum (the unnormalised Laplacian), instead of the
neighbours themselves. A node that matches its neighbourhood has
a zero difference term; a node at a boundary - a bridge account,
or a camouflaged ring member among ring neighbours - is a local
extremum, and the difference term is where that shows up.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    difference_sums,
    evaluate_all,
    train_gconv,
    train_leconv,
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

print("\ntraining LEConv (2-layer, difference aggregation)...")
leconv, le_hist = train_leconv(g, train_ids, val_ids, hidden_dim=16,
                               lr=0.5, batch_size=64,
                               epochs=100, patience=25, seed=7)
print(f"  epochs run: {len(le_hist['train_loss'])}, "
      f"best val loss: {min(le_hist['val_loss']):.4f}")

print("training GraphConv (2-layer, sum aggregation, its own schedule)...")
gconv, _ = train_gconv(g, train_ids, val_ids, hidden_dim=16,
                       lr=0.1, batch_size=64, epochs=60,
                       patience=12, seed=7)

print("training MLP baseline (node features only)...")
mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                   hidden_dim=16, lr=0.3, batch_size=32,
                   epochs=150, patience=20, seed=7)

scores = {
    "LEConv (2-layer, difference)": [leconv.predict(g, [v])[v] for v in val_ids],
    "GraphConv (2-layer, sum)": [gconv.predict(g, [v])[v] for v in val_ids],
    "MLP (features only)": mlp.predict([g.features[v] for v in val_ids]),
}

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<32}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
for name, s in scores.items():
    thr, m = best_threshold_f1(y_true, s)
    auc = evaluate_all(y_true, s)["auc"]
    print(f"  {name:<32}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")

# Difference readout on the raw features: how large is each node's
# Laplacian difference, ring vs background?
diffs = difference_sums(g, g.features)
def _norm(row):
    return sum(x * x for x in row) ** 0.5
ring_norms = [_norm(diffs[v]) for v in sorted(ring)]
bg_norms = [_norm(diffs[v]) for v in range(g.n) if v not in ring]
print(f"\nraw-feature difference norm: ring mean "
      f"{sum(ring_norms) / len(ring_norms):.3f}, background mean "
      f"{sum(bg_norms) / len(bg_norms):.3f}")
print("The raw difference norm is mostly a degree signal in "
      "disguise: the operator is unnormalised, so a ring member's "
      "dozen small per-neighbour differences add up where a "
      "background account's one or two cancel or stay small. The "
      "trained layers apply the same operator to learned "
      "embeddings, where the per-dimension sign pattern, not just "
      "the magnitude, is available to the head.")

"""Temporal example: rings form fast, so recency-weighted message passing
should beat the plain mean.

The generator plants ring edges in a 3-day burst at the end of a 30-day
timeline; benign edges spread uniformly. A TemporalGraphSAGE layer weights
neighbors by recency with a learned decay timescale, concentrating each
ring member's aggregation on its burst-coordinated neighbors. Compare
against plain GraphSAGE and the feature-only MLP on the same graph."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_mlp,
    train_sage,
    train_temporal,
    train_val_split,
)

gen = GraphGenerator(seed=7)
g = gen.generate(n=200, ring_size=18, timestamps=True,
                 timeline_days=30.0, ring_burst_days=3.0)
ring = gen.ring_nodes(g)

ring_edge_times = [g.edge_time(u, v) for u in ring for v in g.neighbors(u)
                   if v in ring and u < v]
print(f"generated {g.n} accounts, {g.edge_count()} edges, "
      f"{len(ring)} planted ring members")
print(f"ring edges formed in days {min(ring_edge_times):.1f}-"
      f"{max(ring_edge_times):.1f} of a 30-day timeline "
      f"({len(ring_edge_times)} ring edges in the burst)")

train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
y_true = [g.labels[v] for v in val_ids]

print("\ntraining TemporalGraphSAGE (learned recency timescale)...")
temporal, _ = train_temporal(g, train_ids, val_ids, hidden_dim=16,
                             k1=5, k2=5, lr=0.1, batch_size=64,
                             epochs=60, patience=12, seed=7)
print(f"  learned decay timescales (days): "
      f"{', '.join(f'{t:.2f}' for t in temporal.taus)}")

print("training GraphSAGE (plain mean aggregation)...")
sage, _ = train_sage(g, train_ids, val_ids, hidden_dim=16,
                     k1=5, k2=5, lr=0.1, batch_size=64,
                     epochs=60, patience=12, seed=7)

print("training MLP baseline (node features only)...")
mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                   hidden_dim=16, lr=0.3, batch_size=32,
                   epochs=150, patience=20, seed=7)

temporal_scores = [temporal.predict(g, [v])[v] for v in val_ids]
sage_scores = [sage.predict(g, [v])[v] for v in val_ids]
mlp_scores = mlp.predict([g.features[v] for v in val_ids])

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<34}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
for name, scores in [
    ("TemporalGraphSAGE (recency-weighted)", temporal_scores),
    ("GraphSAGE (2-layer, mean agg)", sage_scores),
    ("MLP (features only)", mlp_scores),
]:
    thr, m = best_threshold_f1(y_true, scores)
    auc = evaluate_all(y_true, scores)["auc"]
    print(f"  {name:<34}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")
print("\nThe ring's burst is recent, so recency concentrates aggregation on "
      "the ring; benign neighborhoods average over the whole timeline.")

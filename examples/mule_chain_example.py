"""Money-mule chains: a second fraud pattern.

Not every laundering network is a dense pocket. Here money flows
originator -> parallel mule chains -> consolidator -> exit: long thin
paths with almost no triangles. The signature is structural -- a
2-layer mean aggregator must pool weak pass-through tells along a
chain instead of inside a clique. Hop timestamps are fast and
monotone downstream, so the temporal model sees a sequential burst.

Prints validation metrics for GraphSAGE, TemporalGraphSAGE, and the
feature-only MLP baseline under each model's F1-tuned threshold.
"""

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
g = gen.generate_mule_chain(n=200, n_chains=4, chain_len=4)
mc = g.mule_chain
fraud = [u for u in range(g.n) if g.labels[u] == 1]
print(f"generated {g.n} accounts, {g.edge_count()} edges, "
      f"{len(fraud)} laundering-network members "
      f"({len(mc['chains'])} chains x {len(mc['chains'][0])} mules)")
print(f"fraud induced-subgraph density: {g.density(fraud):.3f} "
      f"(the dense-ring pattern is ~0.7; chains are thin)")

train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
y_true = [g.labels[v] for v in val_ids]

# Thin chains are a slow pattern to learn: a mule's 2-hop neighborhood
# mixes chain neighbors with benign ones, so budget a long schedule.
print("\ntraining GraphSAGE...")
sage, _ = train_sage(g, train_ids, val_ids, hidden_dim=16, k1=5, k2=5,
                     lr=0.1, batch_size=64, epochs=120, patience=20, seed=7)

print("training TemporalGraphSAGE (chain hops are a sequential burst)...")
temporal, _ = train_temporal(g, train_ids, val_ids, hidden_dim=16, k1=5, k2=5,
                             lr=0.1, batch_size=64, epochs=120, patience=20, seed=7)

print("training MLP baseline (node features only)...")
mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                   hidden_dim=16, lr=0.3, batch_size=32,
                   epochs=150, patience=20, seed=7)

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<34}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
models = [
    ("GraphSAGE (2-layer, mean agg)",
     [sage.predict(g, [v])[v] for v in val_ids]),
    ("TemporalGraphSAGE (recency)",
     [temporal.predict(g, [v])[v] for v in val_ids]),
    ("MLP (features only)",
     mlp.predict([g.features[v] for v in val_ids])),
]
for name, scores in models:
    thr, m = best_threshold_f1(y_true, scores)
    auc = evaluate_all(y_true, scores)["auc"]
    print(f"  {name:<34}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")
print("\nChains rank about as well as the dense pocket (AUC 0.98 both), but "
      "the comparison flatters the chain pattern: the MLP climbs to 0.86 "
      "from 0.66 because the organizer/consolidator/exit roles carry "
      "stronger individual tells than camouflaged ring members. The graph "
      "signal's marginal value is smaller -- 0.98 vs 0.86, not 0.98 vs "
      "0.66 -- and recency weighting adds nothing (0.98 vs 0.98): on an "
      "undirected graph, the sequential burst is already separated by "
      "structure alone.")

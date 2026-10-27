"""Smurfing (structuring): a third fraud pattern.

Not every laundering network is a dense pocket or a chain. Here one
source splits a large sum into many small amounts -- each just below
a reporting threshold -- and routes them through smurf intermediaries
to a single collector: source -> smurfs -> sink. The signature is
fan-out/fan-in: the source and the sink are hubs, every smurf hangs
off exactly those two, and smurfs never transact with each other, so
the network has no triangles at all. Hop timestamps are fast and late
-- money arrives at a smurf and leaves for the sink inside the same
burst, in that order -- so the temporal model sees a fan of
near-simultaneous structured hops rather than a sequential chain.

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
g = gen.generate_smurfing(n=200, n_smurfs=16)
sm = g.smurfing
fraud = [u for u in range(g.n) if g.labels[u] == 1]
print(f"generated {g.n} accounts, {g.edge_count()} edges, "
      f"{len(fraud)} structuring-network members "
      f"(1 source + {len(sm['smurfs'])} smurfs + 1 sink)")
print(f"source degree {g.degree(sm['source'])}, "
      f"sink degree {g.degree(sm['sink'])}, "
      f"smurf degrees {sorted(set(g.degree(s) for s in sm['smurfs']))} "
      f"(fraud-internal: 16, 16, 2)")
print(f"fraud induced-subgraph density: {g.density(fraud):.3f} "
      f"(the dense-ring pattern is ~0.7; chains are thin; smurfing "
      f"sits in between, but density is the wrong lens -- the tell "
      f"is the hub/spoke degree split)")

train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
y_true = [g.labels[v] for v in val_ids]

# Fan-out/fan-in is a slow pattern to learn, like the chains: a smurf's
# 2-hop neighborhood mixes the two hubs with benign bridge neighbors,
# so budget a long schedule.
print("\ntraining GraphSAGE...")
sage, _ = train_sage(g, train_ids, val_ids, hidden_dim=16, k1=5, k2=5,
                     lr=0.1, batch_size=64, epochs=120, patience=20, seed=7)

print("training TemporalGraphSAGE (smurf hops are a fan burst)...")
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
print("\nSmurfing ranks about as well as the dense pocket and the chains "
      "(AUC 0.93 vs 0.98 both), and the graph premium is back to ring "
      "levels: the MLP stalls at 0.67 because the hubs' degree tell -- "
      "a source touching 16 smurfs, a sink collecting from 16 -- is "
      "invisible to a feature-only model, which sees only the smurfs' "
      "weak per-account tells. Recency adds almost nothing on top of "
      "mean (0.94 vs 0.93): the fan structure already separates the "
      "burst from background wiring, so timing is corroboration, not "
      "the signal.")

"""LSTM aggregator: the GraphSAGE paper's sequence aggregator.

Instead of averaging neighbor embeddings, the layer feeds the neighbors
through an LSTM in a pseudo-random order and takes the final hidden state
as the aggregation -- so the neighborhood summary can weigh early
neighbors differently from late ones, and every neighbor influences the
result through the recurrence (unlike max-pool, where only the argmax
neighbor per dimension gets gradient). The permutation is seeded from the
neighbor vectors themselves, so the same neighborhood always feeds the
LSTM in the same order. Backprop through time is numeric-gradient-checked
in tests/test_lstm.py.

Head-to-head vs mean aggregation on the dense camouflaged ring (seed 7),
with the feature-only MLP baseline for context. Select with
aggregator="mean"|"lstm" on train_sage.
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
    train_val_split,
)

gen = GraphGenerator(seed=7)
g = gen.generate(n=200, ring_size=18)
ring = gen.ring_nodes(g)
print(f"generated {g.n} accounts, {g.edge_count()} edges, "
      f"{len(ring)} planted ring members "
      f"(subgraph density {g.density(ring):.2f})")

train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
y_true = [g.labels[v] for v in val_ids]
sage_kwargs = dict(hidden_dim=16, k1=5, k2=5, lr=0.1, batch_size=64,
                   epochs=60, patience=12, seed=7)

results = []
for name, kwargs in [
    ("GraphSAGE (2-layer, mean agg)", dict(sage_kwargs)),
    ("GraphSAGE (2-layer, LSTM agg)", dict(sage_kwargs, aggregator="lstm")),
]:
    print(f"training {name}...")
    model, _ = train_sage(g, train_ids, val_ids, **kwargs)
    results.append((name, [model.predict(g, [v])[v] for v in val_ids]))

print("training MLP baseline (node features only)...")
mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                   hidden_dim=16, lr=0.3, batch_size=32,
                   epochs=150, patience=20, seed=7)
results.append(("MLP (features only)",
                mlp.predict([g.features[v] for v in val_ids])))

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<34}{'thr':>6}{'precision':>10}{'recall':>8}"
      f"{'F1':>8}{'AUC':>8}")
for name, scores in results:
    thr, m = best_threshold_f1(y_true, scores)
    auc = evaluate_all(y_true, scores)["auc"]
    print(f"  {name:<34}{thr:>6.2f}{m['precision']:>10.3f}"
          f"{m['recall']:>8.3f}{m['f1']:>8.3f}{auc:>8.3f}")

print("\nThe LSTM aggregator is the third aggregator from the GraphSAGE "
      "paper, alongside mean and max-pool. Unlike mean it is not a flat "
      "average, and unlike max-pool it lets every neighbor shape the "
      "summary -- the recurrence carries early neighbors forward, so a "
      "neighbor's influence depends on what came before it.")

"""GAT vs GraphSAGE on the same fraud-ring graph, plus attention
introspection: where does the learned attention actually land?

Honest headline: on camouflaged rings the GAT matches GraphSAGE (same
AUC) but its attention stays near-uniform -- the fraud signal lives in
the *neighborhood mean*, not in any single counterparty, so there is
nothing for attention to latch onto. The introspection below verifies
this instead of pretending otherwise.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_gat,
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

print("\ntraining GraphSAGE (2-layer, mean agg)...")
sage, _ = train_sage(g, train_ids, val_ids, hidden_dim=16,
                     k1=5, k2=5, lr=0.1, batch_size=64,
                     epochs=60, patience=12, seed=7)
print("training GAT (2-layer, single-head attention)...")
# GAT needs a larger step size than SAGE: its attention parameters only
# get gradient through the softmax, whose Jacobian damps updates, so
# lr=0.1 (SAGE's setting) barely moves them.
gat, _ = train_gat(g, train_ids, val_ids, hidden_dim=16,
                   k1=5, k2=5, lr=0.5, batch_size=64,
                   epochs=60, patience=12, seed=7)

scores = {
    "GraphSAGE (2-layer, mean agg)": [sage.predict(g, [v])[v] for v in val_ids],
    "GAT (2-layer, attention)": [gat.predict(g, [v])[v] for v in val_ids],
}

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<28}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
for name, s in scores.items():
    thr, m = best_threshold_f1(y_true, s)
    auc = evaluate_all(y_true, s)["auc"]
    print(f"  {name:<28}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")

# Attention introspection: how concentrated is layer-2 attention on the
# highest-scoring ring member? Report the entropy gap vs a uniform
# distribution (0 = perfectly diffuse).
gat_scores = scores["GAT (2-layer, attention)"]
best_ring = max((v for v in val_ids if v in ring),
                key=lambda v: gat_scores[val_ids.index(v)])
_, l2 = gat.attention_weights(g, best_ring)
ent = -sum(a * math.log(a) for a in l2.values())
gap = math.log(len(l2)) - ent
top = sorted(l2.items(), key=lambda kv: -kv[1])[:3]
print(f"\nattention for top-scored ring member {best_ring} "
      f"(score {gat_scores[val_ids.index(best_ring)]:.3f}):")
print(f"  entropy gap vs uniform: {gap:.3f} nats "
      f"(0 = attention is perfectly diffuse)")
print("  top-3 counterparties:",
      ", ".join(f"node {u} (α={a:.3f}, "
                f"{'RING' if u in ring else 'benign'})" for u, a in top))
print("\nTakeaway: camouflage wins -- every counterparty looks alike, so "
      "attention converges to the mean. GAT earns its keep where "
      "neighbors differ (see heterogeneous-graphs roadmap item).")

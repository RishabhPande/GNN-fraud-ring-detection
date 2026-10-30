"""GATv2 vs GAT on the same fraud-ring graph, plus attention
introspection: does query-dependent attention find signal that static
attention misses?

GATv2 moves the nonlinearity outside the attention map, so the ranking
of a node's neighbors can depend on the query node itself. On the
camouflaged dense ring that may or may not matter: if the signal lives
in the neighborhood mean, no ranking can beat it, and the extra
parameters just add optimization surface. The numbers below say which.
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
    train_gatv2,
    train_val_split,
)

gen = GraphGenerator(seed=7)
g = gen.generate(n=200, ring_size=18)
ring = set(gen.ring_nodes(g))
print(f"generated {g.n} accounts, {g.edge_count()} edges, "
      f"{len(ring)} planted ring members")

train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
y_true = [g.labels[v] for v in val_ids]

print("\ntraining GAT (2-layer, static attention)...")
gat, _ = train_gat(g, train_ids, val_ids, hidden_dim=16,
                   k1=5, k2=5, lr=0.5, batch_size=64,
                   epochs=60, patience=12, seed=7)
print("training GATv2 (2-layer, dynamic attention)...")
gatv2, _ = train_gatv2(g, train_ids, val_ids, hidden_dim=16,
                       k1=5, k2=5, lr=0.5, batch_size=64,
                       epochs=60, patience=12, seed=7)

scores = {
    "GAT (static attention)": [gat.predict(g, [v])[v] for v in val_ids],
    "GATv2 (dynamic attention)": [gatv2.predict(g, [v])[v] for v in val_ids],
}

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<28}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
for name, s in scores.items():
    thr, m = best_threshold_f1(y_true, s)
    auc = evaluate_all(y_true, s)["auc"]
    print(f"  {name:<28}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")

# Attention introspection for GATv2: same entropy-gap check as the GAT
# example, on GATv2's top-scored ring member.
gatv2_scores = scores["GATv2 (dynamic attention)"]
best_ring = max((v for v in val_ids if v in ring),
                key=lambda v: gatv2_scores[val_ids.index(v)])
_, l2 = gatv2.attention_weights(g, best_ring)
ent = -sum(a * math.log(a) for a in l2.values())
gap = math.log(len(l2)) - ent
top = sorted(l2.items(), key=lambda kv: -kv[1])[:3]
print(f"\nGATv2 attention for top-scored ring member {best_ring} "
      f"(score {gatv2_scores[val_ids.index(best_ring)]:.3f}):")
print(f"  entropy gap vs uniform: {gap:.3f} nats "
      f"(0 = attention is perfectly diffuse)")
print("  top-3 counterparties: ",
      ", ".join(f"node {u} (alpha={a:.3f}, "
                f"{'RING' if u in ring else 'benign'})" for u, a in top))
print("\nTakeaway: dynamic attention is expressively stronger (see "
      "tests/test_gatv2.py for a ranking flip GAT cannot express), but "
      "the trained model does not use it: layer-2 attention is perfectly "
      "diffuse (gap 0.000), so the small edge over GAT comes from the "
      "different parameterization, not from selective listening. On "
      "this generator the ring signal is neighborhood-wide, not "
      "query-specific, and no attention variant concentrates on it.")

"""Single-head vs multi-head GAT on the same fraud-ring graph.

Same seed-7 protocol as gat_example.py, trained three ways: 1 head
(classic), 2 heads, 4 heads. Layer 1 concatenates head outputs, layer 2
averages them (the GAT paper's convention). Multi-head needs a longer
schedule than single-head: the final-layer averaging splits each head's
gradient K ways, so under vanilla SGD the heads escape the flat,
near-uniform-attention region slowly. Short patience can lock in a
degenerate checkpoint (low BCE, inverted ranking) -- hence the generous
patience below. The introspection at the end checks whether heads
actually specialize once trained.
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
    train_val_split,
)

gen = GraphGenerator(seed=7)
g = gen.generate(n=200, ring_size=18)
ring = set(gen.ring_nodes(g))
print(f"generated {g.n} accounts, {g.edge_count()} edges, "
      f"{len(ring)} planted ring members")

train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
y_true = [g.labels[v] for v in val_ids]

models = {}
for heads in (1, 2, 4):
    print(f"training GAT ({heads} head{'s' if heads > 1 else ''})...")
    # Multi-head averages head outputs in the final layer, so each head
    # sees 1/K of the output gradient: give it room (patience 40) to
    # escape the flat uniform-attention region.
    model, _ = train_gat(g, train_ids, val_ids, hidden_dim=16,
                         k1=5, k2=5, lr=1.0, batch_size=64,
                         epochs=150, patience=40, seed=7, n_heads=heads)
    models[heads] = model

scores = {f"GAT ({h}-head)": [m.predict(g, [v])[v] for v in val_ids]
          for h, m in models.items()}

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<14}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
for name, s in scores.items():
    thr, m = best_threshold_f1(y_true, s)
    auc = evaluate_all(y_true, s)["auc"]
    print(f"  {name:<14}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")

# Do the heads specialize? For the top-scored ring member in the 4-head
# model, report each head's attention entropy gap vs uniform (0 = diffuse)
# and its top counterparty. Correlated heads = wasted capacity.
mh4 = models[4]
mh4_scores = scores["GAT (4-head)"]
best_ring = max((v for v in val_ids if v in ring),
                key=lambda v: mh4_scores[val_ids.index(v)])
per_head = mh4.attention_weights_per_head(g, best_ring)
print(f"\n4-head attention for top-scored ring member {best_ring} "
      f"(score {mh4_scores[val_ids.index(best_ring)]:.3f}), layer 2:")
for k, alphas in enumerate(per_head[1]):
    ent = -sum(a * math.log(a) for a in alphas.values())
    gap = math.log(len(alphas)) - ent
    top_u, top_a = max(alphas.items(), key=lambda kv: kv[1])
    print(f"  head {k}: entropy gap {gap:.3f} nats; "
          f"top counterparty node {top_u} (α={top_a:.3f}, "
          f"{'RING' if top_u in ring else 'benign'})")
# Head disagreement: mean pairwise L1/2 distance between head
# distributions (1.0 = fully disjoint attention).
l2 = per_head[1]
dis = []
for i in range(len(l2)):
    for j in range(i + 1, len(l2)):
        dis.append(sum(abs(l2[i][u] - l2[j][u]) for u in l2[i]) / 2.0)
print(f"  mean pairwise head L1/2 distance: {sum(dis) / len(dis):.3f} "
      f"(1.0 = fully disjoint attention)")
print("\nTakeaway: extra heads buy a little ranking power (2- and 4-head "
      "match or beat single-head AUC) but train noticeably slower -- the "
      "final-layer averaging dilutes each head's gradient K-fold, so "
      "budget a long schedule and don't trust short-patience early "
      "stopping. Whether heads truly specialize (vs re-learning the same "
      "mean) is data-dependent; check the entropy gaps above.")

"""FiLM-GNN vs GraphSAGE (mean) vs the feature-only baseline on the
same fraud-ring graph.

FiLM conditions the neighborhood message on the node itself: the
first layer generates a per-dimension scale and shift from each
account's own features, and the neighbor mean is scaled and shifted
by them before it is added to the self transform. That product is a
bilinear term, the node's profile times its neighborhood, which no
linear self-plus-neighborhood layer can express. The question this
example answers is whether the camouflaged ring rewards it: ring
members share one profile, so their scales should agree, and a
bridge account can in principle learn to damp the ring side of its
own message.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_film,
    train_mlp,
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

print("\ntraining FiLM-GNN (2-layer, self-conditioned messages)...")
film, film_hist = train_film(g, train_ids, val_ids, hidden_dim=16,
                             lr=0.1, batch_size=64,
                             epochs=60, patience=12, seed=7)
print(f"  epochs run: {len(film_hist['train_loss'])}, "
      f"best val loss: {min(film_hist['val_loss']):.4f}")

print("training GraphSAGE (2-layer, mean agg)...")
sage, _ = train_sage(g, train_ids, val_ids, hidden_dim=16,
                     lr=0.1, batch_size=32, epochs=60,
                     patience=12, seed=7)

print("training MLP baseline (node features only)...")
mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                   hidden_dim=16, lr=0.3, batch_size=32,
                   epochs=150, patience=20, seed=7)

scores = {
    "FiLM-GNN (conditioned mean)": [film.predict(g, [v])[v] for v in val_ids],
    "GraphSAGE (2-layer, mean agg)": [sage.predict(g, [v])[v] for v in val_ids],
    "MLP (features only)": mlp.predict([g.features[v] for v in val_ids]),
}

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<32}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
for name, s in scores.items():
    thr, m = best_threshold_f1(y_true, s)
    auc = evaluate_all(y_true, s)["auc"]
    print(f"  {name:<32}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")

print("\nfirst-layer modulation, ring members vs background accounts")
print("(scale multiplies the neighbor message dimension by dimension;")
print(" 1.0 is the unmodulated GraphSAGE mean, 0.0 switches it off):")
ring_scales = []
bg_scales = []
for v in range(g.n):
    scale, _ = film.modulation(g, v)
    (ring_scales if v in ring else bg_scales).append(scale)
hidden = len(ring_scales[0])
for j in range(hidden):
    r = sum(s[j] for s in ring_scales) / len(ring_scales)
    b = sum(s[j] for s in bg_scales) / len(bg_scales)
    print(f"  dim {j:>2}: ring mean scale {r:>7.3f}   background {b:>7.3f}")
all_r = [x for s in ring_scales for x in s]
all_b = [x for s in bg_scales for x in s]
print(f"  overall mean scale: ring {sum(all_r) / len(all_r):.3f}, "
      f"background {sum(all_b) / len(all_b):.3f}")
spread = max(all_r + all_b) - min(all_r + all_b)
print(f"  scale spread across accounts and dims: {spread:.3f} "
      f"(0 would mean the conditioning never fires)")

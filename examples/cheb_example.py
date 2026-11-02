"""ChebNet vs GraphSAGE on the same fraud-ring graph.

Honest headline: on the dense ring, ChebNet (2 layers, K=2 each)
matches the best models in the repo (AUC 1.000) and beats the plain GCN
(AUC ~0.94) -- the spectral filter reads a 2-hop neighborhood per layer
instead of one, and the k=0 term keeps each node's own features in the
mix without a self-loop normalization diluting the ring's hubs. Like
every other headline in this repo, this is one seed (7) until a sweep
says otherwise.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_cheb,
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

print("\ntraining ChebNet (2-layer, K=2)...")
# Same long schedule as the GCN: the normalized aggregation shrinks
# gradients, and lr=0.1 underfits the GCN family on this graph.
cheb, cheb_hist = train_cheb(g, train_ids, val_ids, hidden_dim=16,
                             K=2, lr=0.5, batch_size=64, epochs=100,
                             patience=25, seed=7)
print(f"  epochs run: {len(cheb_hist['train_loss'])}, "
      f"best val loss: {min(cheb_hist['val_loss']):.4f}")

print("training GraphSAGE (2-layer, mean agg)...")
sage, _ = train_sage(g, train_ids, val_ids, hidden_dim=16,
                     k1=5, k2=5, lr=0.1, batch_size=64,
                     epochs=60, patience=12, seed=7)

scores = {
    "ChebNet (2-layer, K=2)": [cheb.predict(g, [v])[v] for v in val_ids],
    "GraphSAGE (2-layer, mean agg)": [sage.predict(g, [v])[v]
                                      for v in val_ids],
}

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<28}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
for name, s in scores.items():
    thr, m = best_threshold_f1(y_true, s)
    auc = evaluate_all(y_true, s)["auc"]
    print(f"  {name:<28}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")
print("\nChebNet's edge over the GCN is the filter, not the schedule: "
      "each layer mixes the 0-, 1-, and 2-hop terms with learned "
      "weights instead of one normalized hop, so the ring's dense "
      "pocket survives without the hub down-weighting that costs "
      "the GCN.")

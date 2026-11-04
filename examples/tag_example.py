"""TAGCN vs the plain GCN and GraphSAGE on the same fraud-ring graph.

Honest headline: on the dense ring, TAGCN (two layers, K = 2, so up
to 4 hops with a separate weight matrix per hop) is measured against
the GCN it generalizes and the SAGE mean model. Like every other
headline in this repo, this is one seed (7) until a sweep says
otherwise.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_gcn,
    train_sage,
    train_tag,
    train_val_split,
)

gen = GraphGenerator(seed=7)
g = gen.generate(n=200, ring_size=18)
ring = set(gen.ring_nodes(g))
print(f"generated {g.n} accounts, {g.edge_count()} edges, "
      f"{len(ring)} planted ring members")

train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
y_true = [g.labels[v] for v in val_ids]

print("\ntraining TAGCN (2-layer, K=2)...")
tag, tag_hist = train_tag(g, train_ids, val_ids, hidden_dim=16, K=2,
                          lr=0.5, batch_size=64, epochs=100,
                          patience=25, seed=7)
print(f"  epochs run: {len(tag_hist['train_loss'])}, "
      f"best val loss: {min(tag_hist['val_loss']):.4f}")

print("training GCN (2-layer)...")
gcn, _ = train_gcn(g, train_ids, val_ids, hidden_dim=16,
                   lr=0.5, batch_size=64, epochs=100,
                   patience=25, seed=7)

print("training GraphSAGE (2-layer, mean agg)...")
sage, _ = train_sage(g, train_ids, val_ids, hidden_dim=16,
                     k1=5, k2=5, lr=0.1, batch_size=64,
                     epochs=60, patience=12, seed=7)

scores = {
    "TAGCN (2-layer, K=2)": [tag.predict(g, [v])[v] for v in val_ids],
    "GCN (2-layer)": [gcn.predict(g, [v])[v] for v in val_ids],
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
print("\nTAGCN generalizes the GCN: one layer already mixes the 0-, 1-"
      ", and 2-hop neighborhoods with a weight matrix per hop, so "
      "the two-layer stack reads up to 4 hops without a second "
      "aggregation washing the pocket out. Whether the per-hop "
      "weights actually buy anything over the GCN's single hop per "
      "layer is what the table above says, not the construction.")

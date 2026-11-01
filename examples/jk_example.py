"""JK-GCN vs plain GCN vs the feature-only baseline on the same graph.

Jumping Knowledge keeps the layer-1 embedding alongside the layer-2
embedding and scores their concatenation. On the dense ring the
question is sharp: layer 1 already sees the dense pocket (1 hop), and
layer 2 sees the pocket's wider neighborhood (2 hops). If the plain
GCN's last layer already carries the pocket signal, JK should tie it.
If the second aggregation washes the pocket out, keeping layer 1
alongside should win.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_gcn,
    train_jk,
    train_mlp,
    train_val_split,
)

gen = GraphGenerator(seed=7)
g = gen.generate(n=200, ring_size=18)
ring = set(gen.ring_nodes(g))
print(f"generated {g.n} accounts, {g.edge_count()} edges, "
      f"{len(ring)} planted ring members")

train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
y_true = [g.labels[v] for v in val_ids]

print("\ntraining JK-GCN (2-layer, [layer-1 || layer-2] head)...")
jk, jk_hist = train_jk(g, train_ids, val_ids, hidden_dim=16,
                       lr=0.5, batch_size=64, epochs=100,
                       patience=25, seed=7)
print(f"  epochs run: {len(jk_hist['train_loss'])}, "
      f"best val loss: {min(jk_hist['val_loss']):.4f}")

print("training GCN (2-layer, last layer only)...")
gcn, _ = train_gcn(g, train_ids, val_ids, hidden_dim=16,
                   lr=0.5, batch_size=64, epochs=100,
                   patience=25, seed=7)

print("training MLP baseline (node features only)...")
mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                   hidden_dim=16, lr=0.3, batch_size=32,
                   epochs=150, patience=20, seed=7)

scores = {
    "JK-GCN ([layer-1 || layer-2])": [jk.predict(g, [v])[v] for v in val_ids],
    "GCN (2-layer, last layer)": [gcn.predict(g, [v])[v] for v in val_ids],
    "MLP (features only)": mlp.predict([g.features[v] for v in val_ids]),
}

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<28}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
for name, s in scores.items():
    thr, m = best_threshold_f1(y_true, s)
    auc = evaluate_all(y_true, s)["auc"]
    print(f"  {name:<28}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")
print("\nJK-GCN costs one wider output head (32 weights instead of 16). "
      "Whatever it scores, the shallow representation earned the "
      "difference over the plain GCN.")

"""LightGCN vs the feature-only baseline on the same
fraud-ring graph, plus a propagation-depth sweep.

LightGCN is this repo's structure-only control: free per-node
embeddings, propagation with no weight matrices and no
nonlinearity, a uniform average over the propagation powers,
and a linear head. It never reads a single account feature, so
whatever it ranks, it ranks from the wiring alone - and the
K = 0 row of the sweep is the control's control, where a
held-out account's embedding is never trained at all.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_lightgcn,
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

print("\ntraining LightGCN (free embeddings + averaged propagation, K=3)...")
lgcn, lgcn_hist = train_lightgcn(g, train_ids, val_ids, emb_dim=16,
                                 K=3, lr=0.5, batch_size=64,
                                 epochs=100, patience=25, seed=7)
print(f"  epochs run: {len(lgcn_hist['train_loss'])}, "
      f"best val loss: {min(lgcn_hist['val_loss']):.4f}")

print("training MLP baseline (node features only)...")
mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                   hidden_dim=16, lr=0.3, batch_size=32,
                   epochs=150, patience=20, seed=7)

scores = {
    "LightGCN (structure only, K=3)": [lgcn.predict(g, [v])[v] for v in val_ids],
    "MLP (features only)": mlp.predict([g.features[v] for v in val_ids]),
}

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<32}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
for name, s in scores.items():
    thr, m = best_threshold_f1(y_true, s)
    auc = evaluate_all(y_true, s)["auc"]
    print(f"  {name:<32}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")

print("\npropagation depth sweep (same schedule, only K changes):")
print(f"  {'K':>3}{'own-embedding weight':>22}{'AUC':>8}{'F1':>8}")
for K in (0, 2, 3):
    if K == 3:
        s = scores["LightGCN (structure only, K=3)"]
    else:
        m_k, _ = train_lightgcn(g, train_ids, val_ids, emb_dim=16,
                                K=K, lr=0.5, batch_size=64,
                                epochs=100, patience=25, seed=7)
        s = [m_k.predict(g, [v])[v] for v in val_ids]
    thr, m = best_threshold_f1(y_true, s)
    auc = evaluate_all(y_true, s)["auc"]
    print(f"  {K:>3}{1.0 / (K + 1):>22.3f}{auc:>8.3f}{m['f1']:>8.3f}")

print("\nNo features, no weight matrices, no nonlinearity - the "
      "wiring alone ranks the ring, because the generator "
      "camouflages the profiles and leaves the pocket in the "
      "edges. The price is transduction: a new account has no "
      "embedding row until the model is retrained with it.")

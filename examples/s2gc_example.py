"""S2GC vs SGC vs the feature-only baseline on the same
fraud-ring graph, plus a depth sweep over the averaging order.

S2GC averages the propagation powers, (1 / (K + 1)) sum_k
S_hat^k H, instead of keeping only the last one (SGC) or
learning a weight per power (GPR-GNN). The filter has no learned
parameters at all: a node's own encoder state stays in its mix
at weight 1 / (K + 1) at any depth, and the sweep below shows
what the depth itself buys on the dense ring.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_mlp,
    train_s2gc,
    train_sgc,
    train_val_split,
)

gen = GraphGenerator(seed=7)
g = gen.generate(n=200, ring_size=18)
ring = set(gen.ring_nodes(g))
print(f"generated {g.n} accounts, {g.edge_count()} edges, "
      f"{len(ring)} planted ring members")

train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
y_true = [g.labels[v] for v in val_ids]

print("\ntraining S2GC (MLP + averaged propagation, K=5)...")
s2gc, s2gc_hist = train_s2gc(g, train_ids, val_ids, hidden_dim=16,
                             K=5, lr=0.5, batch_size=64,
                             epochs=100, patience=25, seed=7)
print(f"  epochs run: {len(s2gc_hist['train_loss'])}, "
      f"best val loss: {min(s2gc_hist['val_loss']):.4f}")

print("training SGC (last power only, hops=2, same schedule)...")
sgc, _ = train_sgc(g, train_ids, val_ids, hops=2, lr=0.5,
                   batch_size=64, epochs=100, patience=25, seed=7)

print("training MLP baseline (node features only)...")
mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                   hidden_dim=16, lr=0.3, batch_size=32,
                   epochs=150, patience=20, seed=7)

scores = {
    "S2GC (averaged powers, K=5)": [s2gc.predict(g, [v])[v] for v in val_ids],
    "SGC (2-hop propagation)": [sgc.predict(g, [v])[v] for v in val_ids],
    "MLP (features only)": mlp.predict([g.features[v] for v in val_ids]),
}

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<32}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
for name, s in scores.items():
    thr, m = best_threshold_f1(y_true, s)
    auc = evaluate_all(y_true, s)["auc"]
    print(f"  {name:<32}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")

print("\naveraging depth sweep (same schedule, only K changes):")
print(f"  {'K':>3}{'own-state weight':>18}{'AUC':>8}{'F1':>8}")
for K in (0, 2, 5):
    if K == 5:
        s = scores["S2GC (averaged powers, K=5)"]
    else:
        m_k, _ = train_s2gc(g, train_ids, val_ids, hidden_dim=16,
                            K=K, lr=0.5, batch_size=64,
                            epochs=100, patience=25, seed=7)
        s = [m_k.predict(g, [v])[v] for v in val_ids]
    thr, m = best_threshold_f1(y_true, s)
    auc = evaluate_all(y_true, s)["auc"]
    print(f"  {K:>3}{1.0 / (K + 1):>18.3f}{auc:>8.3f}{m['f1']:>8.3f}")

print("\nAveraging keeps every depth in the mix at a fixed weight: "
      "no learned filter, and the node's own state never washes "
      "out the way the last-power-only model lets it.")

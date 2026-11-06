"""GPR-GNN vs APPNP vs SGC vs the feature-only baseline on the same
fraud-ring graph.

GPR-GNN starts from APPNP's coefficients -- the personalized PageRank
weights with teleport alpha = 0.1 over 10 propagation steps -- and
then *learns* them. If the learned weights stay near the PPR init,
APPNP's hand-picked polynomial was already right for this graph; if
they move, especially negative, the data wanted a different filter
shape than any fixed teleport schedule provides. Either way the
comparison is against the same protocol, seed, and split.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    ppr_gamma_init,
    train_appnp,
    train_gpr,
    train_mlp,
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

print("\ntraining GPR-GNN (MLP + 10 propagation steps, learned gamma)...")
gpr, gpr_hist = train_gpr(g, train_ids, val_ids, hidden_dim=16,
                          steps=10, alpha=0.1, lr=0.5,
                          batch_size=64, epochs=100,
                          patience=25, seed=7)
print(f"  epochs run: {len(gpr_hist['train_loss'])}, "
      f"best val loss: {min(gpr_hist['val_loss']):.4f}")

print("training APPNP (MLP + 10 personalized PageRank steps, alpha=0.1)...")
appnp, _ = train_appnp(g, train_ids, val_ids, hidden_dim=16,
                       steps=10, alpha=0.1, lr=0.5,
                       batch_size=64, epochs=100,
                       patience=25, seed=7)

print("training SGC (2-hop propagation + logistic regression)...")
sgc, _ = train_sgc(g, train_ids, val_ids, hops=2,
                   lr=0.5, batch_size=64, epochs=100,
                   patience=25, seed=7)

print("training MLP baseline (node features only)...")
mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                   hidden_dim=16, lr=0.3, batch_size=32,
                   epochs=150, patience=20, seed=7)

scores = {
    "GPR-GNN (learned gamma)": [gpr.predict(g, [v])[v] for v in val_ids],
    "APPNP (predict, then propagate)": [appnp.predict(g, [v])[v]
                                         for v in val_ids],
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

print("\nlearned gamma vs its PPR init (alpha=0.1, steps=10):")
init = ppr_gamma_init(10, 0.1)
print(f"  {'k':>3}{'init':>9}{'learned':>9}")
for k in range(11):
    print(f"  {k:>3}{init[k]:>9.3f}{gpr.gamma[k]:>9.3f}")
print(f"  sum of learned gamma: {sum(gpr.gamma):.3f}")
print("\nGPR-GNN's edge over APPNP, if any, is the learned polynomial: "
      "the fixed PPR weights are one point in a space the data may "
      "prefer to leave.")

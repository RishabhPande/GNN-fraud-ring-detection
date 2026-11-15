"""JacobiConv vs BernNet vs GPR-GNN vs the feature-only baseline
on the same fraud-ring graph.

JacobiConv learns the same kind of propagation polynomial as
GPR-GNN and BernNet, but in the Jacobi basis P_k^(a, b) of the
normalized adjacency, where the two basis parameters tilt the
basis toward one end of the spectrum. The basis choice turns out
to be a conditioning choice: at the default corner a = b = -0.5
each term is a shrinking multiple of a Chebyshev polynomial
(1/2, 3/8, 5/16, ...), the term norms stay under 1, and the
filter trains at lr 0.5 like BernNet's. This example also trains
the Legendre corner (a = b = 0) at the smaller step it needs, so
the gap between the corners is measured, not asserted.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    jacobi_theta_init,
    train_bern,
    train_gpr,
    train_jacobi,
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

print("\ntraining JacobiConv (MLP + Jacobi filter, K=4, a=b=-0.5)...")
jac, jac_hist = train_jacobi(g, train_ids, val_ids, hidden_dim=16,
                             K=4, lr=0.5, batch_size=64,
                             epochs=100, patience=25, seed=7)
print(f"  epochs run: {len(jac_hist['train_loss'])}, "
      f"best val loss: {min(jac_hist['val_loss']):.4f}")

print("training JacobiConv at the Legendre corner (a=b=0, lr=0.1)...")
leg, _ = train_jacobi(g, train_ids, val_ids, hidden_dim=16,
                      K=4, a=0.0, b=0.0, lr=0.1, batch_size=64,
                      epochs=100, patience=25, seed=7)

print("training BernNet (MLP + Bernstein filter, K=4)...")
bern, _ = train_bern(g, train_ids, val_ids, hidden_dim=16,
                     K=4, lr=0.5, batch_size=64,
                     epochs=100, patience=25, seed=7)

print("training GPR-GNN (MLP + 10 propagation steps, learned gamma)...")
gpr, _ = train_gpr(g, train_ids, val_ids, hidden_dim=16,
                   steps=10, alpha=0.1, lr=0.5,
                   batch_size=64, epochs=100,
                   patience=25, seed=7)

print("training MLP baseline (node features only)...")
mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                   hidden_dim=16, lr=0.3, batch_size=32,
                   epochs=150, patience=20, seed=7)

scores = {
    "JacobiConv (a=b=-0.5 filter)": [jac.predict(g, [v])[v] for v in val_ids],
    "JacobiConv (a=b=0, Legendre)": [leg.predict(g, [v])[v] for v in val_ids],
    "BernNet (Bernstein filter)": [bern.predict(g, [v])[v] for v in val_ids],
    "GPR-GNN (learned gamma)": [gpr.predict(g, [v])[v] for v in val_ids],
    "MLP (features only)": mlp.predict([g.features[v] for v in val_ids]),
}

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<32}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
for name, s in scores.items():
    thr, m = best_threshold_f1(y_true, s)
    auc = evaluate_all(y_true, s)["auc"]
    print(f"  {name:<32}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")

print("\nlearned theta vs its (I + S) / 2 init, default corner "
      "(theta_k weights the degree-k Jacobi term):")
init = jacobi_theta_init(4)
print(f"  {'k':>3}{'init':>9}{'learned':>9}")
for k in range(5):
    print(f"  {k:>3}{init[k]:>9.3f}{jac.theta[k]:>9.3f}")
print("\nThe polynomial space is the same at every (a, b) corner -- "
      "degree 4 in S either way. What changes is the conditioning "
      "of the basis the coefficients are learned in, and on this "
      "graph that is the difference between matching BernNet and "
      "landing near the GCN.")

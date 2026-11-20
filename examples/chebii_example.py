"""ChebNetII vs BernNet vs JacobiConv vs the feature-only baseline
on the same fraud-ring graph.

ChebNetII learns the same kind of propagation polynomial as
BernNet and JacobiConv, but its learned parameters are not basis
coefficients at all: they are the filter's values at the Chebyshev
nodes x_j = cos((j + 1/2) pi / (K + 1)), turned into Chebyshev
coefficients by a fixed interpolation map (a discrete cosine
transform). Every gamma is therefore readable directly - it is
the filter's gain at one frequency - and the parameterization
avoids the ill conditioning that direct coefficient learning
suffers at higher orders. This example prints the learned value
table against its (I + S) / 2 init so the filter shape can be
read off without any basis conversion.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    chebii_gamma_init,
    chebii_nodes,
    evaluate_all,
    train_bern,
    train_chebii,
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

print("\ntraining ChebNetII (MLP + interpolated filter, K=4)...")
cii, cii_hist = train_chebii(g, train_ids, val_ids, hidden_dim=16,
                             K=4, lr=0.5, batch_size=64,
                             epochs=100, patience=25, seed=7)
print(f"  epochs run: {len(cii_hist['train_loss'])}, "
      f"best val loss: {min(cii_hist['val_loss']):.4f}")

print("training BernNet (MLP + Bernstein filter, K=4)...")
bern, _ = train_bern(g, train_ids, val_ids, hidden_dim=16,
                     K=4, lr=0.5, batch_size=64,
                     epochs=100, patience=25, seed=7)

print("training JacobiConv (MLP + Jacobi filter, K=4, a=b=-0.5)...")
jac, _ = train_jacobi(g, train_ids, val_ids, hidden_dim=16,
                      K=4, lr=0.5, batch_size=64,
                      epochs=100, patience=25, seed=7)

print("training MLP baseline (node features only)...")
mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                   hidden_dim=16, lr=0.3, batch_size=32,
                   epochs=150, patience=20, seed=7)

scores = {
    "ChebNetII (value filter)": [cii.predict(g, [v])[v] for v in val_ids],
    "BernNet (Bernstein filter)": [bern.predict(g, [v])[v] for v in val_ids],
    "JacobiConv (a=b=-0.5 filter)": [jac.predict(g, [v])[v] for v in val_ids],
    "MLP (features only)": mlp.predict([g.features[v] for v in val_ids]),
}

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<32}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
for name, s in scores.items():
    thr, m = best_threshold_f1(y_true, s)
    auc = evaluate_all(y_true, s)["auc"]
    print(f"  {name:<32}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")

print("\nlearned filter values vs their (I + S) / 2 init "
      "(gamma_j is the filter's gain at node x_j, low frequencies "
      "first):")
init = chebii_gamma_init(4)
nodes = chebii_nodes(4)
print(f"  {'j':>3}{'x_j':>9}{'init':>9}{'learned':>9}")
for j in range(5):
    print(f"  {j:>3}{nodes[j]:>9.3f}{init[j]:>9.3f}{cii.gamma[j]:>9.3f}")
print("\nThe degree-4 polynomial space is the same one BernNet and "
      "JacobiConv search. What changes is the coordinates the "
      "coefficients are learned in: values at frequencies, which "
      "stay on the filter's own scale, instead of basis weights "
      "whose effect on the filter is indirect.")

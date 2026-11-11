"""BernNet vs GPR-GNN vs SGC vs the feature-only baseline on the same
fraud-ring graph.

BernNet learns the same kind of propagation polynomial as GPR-GNN,
but in the Bernstein basis of the rescaled Laplacian, where each
coefficient theta_k is the filter's value at the k-th Bernstein point
of the spectrum (lambda = 2k/K): low frequencies at small k, high
frequencies at large k. The init theta_k = 1 - k/K is the filter
g(lambda) = 1 - lambda/2, the operator (I + S)/2. If the learned theta
stays near that line, one half-smoothing step was already the right
filter; where it leaves the line, the shape it moves to can be read
directly as a frequency response.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    bern_theta_init,
    best_threshold_f1,
    evaluate_all,
    train_bern,
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

print("\ntraining BernNet (MLP + Bernstein filter, K=4)...")
bern, bern_hist = train_bern(g, train_ids, val_ids, hidden_dim=16,
                             K=4, lr=0.5, batch_size=64,
                             epochs=100, patience=25, seed=7)
print(f"  epochs run: {len(bern_hist['train_loss'])}, "
      f"best val loss: {min(bern_hist['val_loss']):.4f}")

print("training GPR-GNN (MLP + 10 propagation steps, learned gamma)...")
gpr, _ = train_gpr(g, train_ids, val_ids, hidden_dim=16,
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
    "BernNet (Bernstein filter)": [bern.predict(g, [v])[v] for v in val_ids],
    "GPR-GNN (learned gamma)": [gpr.predict(g, [v])[v] for v in val_ids],
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

print("\nlearned theta vs its linear low-pass init "
      "(theta_k is the filter value at lambda = 2k/K):")
init = bern_theta_init(4)
print(f"  {'k':>3}{'lambda':>8}{'init':>9}{'learned':>9}")
for k in range(5):
    print(f"  {k:>3}{2.0 * k / 4:>8.2f}{init[k]:>9.3f}{bern.theta[k]:>9.3f}")
print("\nBernNet's coefficients are a frequency response, not step "
      "weights: read the learned column top to bottom and you are "
      "reading how much of each frequency band survives the filter.")

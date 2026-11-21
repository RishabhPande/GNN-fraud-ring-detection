"""GDC vs GPR-GNN vs the feature-only baseline on the same
fraud-ring graph.

GDC fixes the propagation filter's shape to the heat kernel -
Poisson weights theta_k = exp(-t) t^k / k! over the propagation
powers - and learns only the diffusion time t (in log space,
init t = 2). Where GPR-GNN learns a weight per step and can
shape the filter freely, GDC asks how far the signal diffuses:
one number, read back with diffusion_time(), with the learned
coefficient table printed against its t = 2 init.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    heat_kernel_coeffs,
    train_gdc,
    train_gpr,
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

print("\ntraining GDC (MLP + heat-kernel diffusion, K=6, t init 2)...")
gdc, gdc_hist = train_gdc(g, train_ids, val_ids, hidden_dim=16,
                          K=6, lr=0.5, batch_size=64,
                          epochs=100, patience=25, seed=7)
print(f"  epochs run: {len(gdc_hist['train_loss'])}, "
      f"best val loss: {min(gdc_hist['val_loss']):.4f}")

print("training GPR-GNN (MLP + learned step weights)...")
gpr, _ = train_gpr(g, train_ids, val_ids, hidden_dim=16,
                   steps=10, lr=0.5, batch_size=64,
                   epochs=100, patience=25, seed=7)

print("training MLP baseline (node features only)...")
mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                   hidden_dim=16, lr=0.3, batch_size=32,
                   epochs=150, patience=20, seed=7)

scores = {
    "GDC (learned diffusion time)": [gdc.predict(g, [v])[v] for v in val_ids],
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

t = gdc.diffusion_time()
print(f"\nlearned diffusion time: t = {t:.3f} (init 2.000)")
print("heat-kernel coefficients at the init vs the learned time:")
init = heat_kernel_coeffs(2.0, 6)
learned = heat_kernel_coeffs(t, 6)
print(f"  {'k':>3}{'init':>9}{'learned':>9}")
for k in range(7):
    print(f"  {k:>3}{init[k]:>9.3f}{learned[k]:>9.3f}")
print("\nOne learned number replaces GPR-GNN's per-step weights: "
      "the shape is the Poisson diffusion shape by construction, "
      "and training only chooses its reach.")

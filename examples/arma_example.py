"""ARMA-GNN vs GCN vs the feature-only baseline on the same
fraud-ring graph.

ARMA replaces the polynomial filters (ChebNet, GPR-GNN, BernNet)
with a rational one built as a recursion: each layer re-injects
its input transform at every iteration while the state propagates
through the normalized adjacency under a shared recurrent weight.
A rational filter can sharpen its response in fewer steps than a
polynomial of the same depth, and the re-injected input term is
the moving-average part that keeps a node's own features from
washing out. The recursion here is linear, with the ReLU at the
layer output: an activation inside the loop trained far worse in
this repo's experiments, which the module docstring records.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_arma,
    train_gcn,
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

print("\ntraining ARMA-GNN (2-layer, iters=3, linear recursion)...")
arma, arma_hist = train_arma(g, train_ids, val_ids, hidden_dim=16,
                             iters=3, lr=0.1, batch_size=64,
                             epochs=150, patience=40, seed=7)
print(f"  epochs run: {len(arma_hist['train_loss'])}, "
      f"best val loss: {min(arma_hist['val_loss']):.4f}")

print("training GCN (2-layer, normalized)...")
gcn, _ = train_gcn(g, train_ids, val_ids, hidden_dim=16,
                   lr=0.5, batch_size=64, epochs=100,
                   patience=25, seed=7)

print("training MLP baseline (node features only)...")
mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                   hidden_dim=16, lr=0.3, batch_size=32,
                   epochs=150, patience=20, seed=7)

scores = {
    "ARMA-GNN (rational filter)": [arma.predict(g, [v])[v] for v in val_ids],
    "GCN (2-layer, normalized)": [gcn.predict(g, [v])[v] for v in val_ids],
    "MLP (features only)": mlp.predict([g.features[v] for v in val_ids]),
}

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<32}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
for name, s in scores.items():
    thr, m = best_threshold_f1(y_true, s)
    auc = evaluate_all(y_true, s)["auc"]
    print(f"  {name:<32}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")

print("\nrecurrent weight magnitude, layer 1 vs layer 2")
print("(the shared W_rec is the filter: a near-zero norm would mean")
print(" the recursion never fires and the model is a plain MLP stack):")
for name, layer in (("layer 1", arma.layer1), ("layer 2", arma.layer2)):
    fro = sum(x * x for row in layer.W_rec for x in row) ** 0.5
    print(f"  {name}: ||W_rec||_F = {fro:.3f}")

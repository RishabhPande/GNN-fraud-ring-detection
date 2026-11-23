"""Virtual-node GraphSAGE vs the feature-only baseline on the
same fraud-ring graph, plus a post-hoc ablation that zeroes the
global term in the trained model.

Every other model here reads only a node's neighborhood, so
information travels one edge per layer. A virtual node - here
the mean of all node states, broadcast back through a learned
matrix at each layer - gives every account the whole graph's
average state as context and gives distant accounts a two step
path through the virtual node in a single layer.
"""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_mlp,
    train_val_split,
    train_vnode,
)

gen = GraphGenerator(seed=7)
g = gen.generate(n=200, ring_size=18)
ring = set(gen.ring_nodes(g))
print(f"generated {g.n} accounts, {g.edge_count()} edges, "
      f"{len(ring)} planted ring members")

train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
y_true = [g.labels[v] for v in val_ids]

print("\ntraining virtual-node GraphSAGE (2 layers, global broadcast)...")
model, hist = train_vnode(g, train_ids, val_ids, hidden_dim=16,
                          lr=0.1, batch_size=64,
                          epochs=60, patience=12, seed=7)
print(f"  epochs run: {len(hist['train_loss'])}, "
      f"best val loss: {min(hist['val_loss']):.4f}")
for name, layer in (("layer 1", model.layer1), ("layer 2", model.layer2)):
    norm = sum(x * x for row in layer.W_glob for x in row) ** 0.5
    print(f"  {name} W_glob Frobenius norm after training: {norm:.3f}")

print("training MLP baseline (node features only)...")
mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                   hidden_dim=16, lr=0.3, batch_size=32,
                   epochs=150, patience=20, seed=7)

ablated = copy.deepcopy(model)
for layer in (ablated.layer1, ablated.layer2):
    layer.W_glob = [[0.0] * len(row) for row in layer.W_glob]

scores = {
    "VirtualNodeSAGE (global)": [model.predict(g, [v])[v] for v in val_ids],
    "Trained model, W_glob zeroed": [ablated.predict(g, [v])[v]
                                     for v in val_ids],
    "MLP (features only)": mlp.predict([g.features[v] for v in val_ids]),
}

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<32}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
for name, s in scores.items():
    thr, m = best_threshold_f1(y_true, s)
    auc = evaluate_all(y_true, s)["auc"]
    print(f"  {name:<32}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")

print("\nThe global vector is the same for every account, so it "
      "cannot rank anyone by itself: it is context, a graph-wide "
      "baseline each account's own neighborhood is read against, "
      "plus a two step path between accounts too far apart for "
      "two neighborhood layers to connect.")

"""SIGN vs SGC vs GCN vs the feature-only baseline on the same
fraud-ring graph.

SIGN keeps every hop's propagated features and lets a small MLP decide
which ones matter; SGC keeps only the last hop under a single linear
head. If SIGN beats SGC, hop identity matters. If it ties, the last
hop already carried everything, and the extra parameters bought
nothing. The GCN shows what the full nonlinear message-passing model
does on the same graph.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_gcn,
    train_mlp,
    train_sgc,
    train_sign,
    train_val_split,
)

gen = GraphGenerator(seed=7)
g = gen.generate(n=200, ring_size=18)
ring = set(gen.ring_nodes(g))
print(f"generated {g.n} accounts, {g.edge_count()} edges, "
      f"{len(ring)} planted ring members")

train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
y_true = [g.labels[v] for v in val_ids]

print("\ntraining SIGN (0/1/2-hop concat + MLP head)...")
sign, sign_hist = train_sign(g, train_ids, val_ids, hops=2,
                             hidden_dim=16, lr=0.5, batch_size=64,
                             epochs=100, patience=25, seed=7)
print(f"  epochs run: {len(sign_hist['train_loss'])}, "
      f"best val loss: {min(sign_hist['val_loss']):.4f}")

print("training SGC (2-hop propagation + logistic regression)...")
sgc, _ = train_sgc(g, train_ids, val_ids, hops=2,
                   lr=0.5, batch_size=64, epochs=100,
                   patience=25, seed=7)

print("training GCN (2-layer, symmetric normalization)...")
gcn, _ = train_gcn(g, train_ids, val_ids, hidden_dim=16,
                   lr=0.5, batch_size=64, epochs=100,
                   patience=25, seed=7)

print("training MLP baseline (node features only)...")
mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                   hidden_dim=16, lr=0.3, batch_size=32,
                   epochs=150, patience=20, seed=7)

scores = {
    "SIGN (0/1/2-hop concat + MLP)": [sign.predict(g, [v])[v] for v in val_ids],
    "SGC (2-hop propagation)": [sgc.predict(g, [v])[v] for v in val_ids],
    "GCN (2-layer, normalized)": [gcn.predict(g, [v])[v] for v in val_ids],
    "MLP (features only)": mlp.predict([g.features[v] for v in val_ids]),
}

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<28}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
for name, s in scores.items():
    thr, m = best_threshold_f1(y_true, s)
    auc = evaluate_all(y_true, s)["auc"]
    print(f"  {name:<28}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")

# Which hop block does the head actually lean on? The W1 column norms
# per block say how much each hop's features can move the hidden layer.
in_dim = g.feature_dim
print("\nSIGN head: W1 column-norm per hop block "
      "(how much each hop can move the hidden layer):")
for k in range(3):
    cols = range(k * in_dim, (k + 1) * in_dim)
    norm = sum(
        sum(model_w * model_w for model_w in (sign.W1[i][j] for j in cols))
        for i in range(sign.hidden_dim)
    ) ** 0.5
    print(f"  hop {k}: {norm:.3f}")
print("\nSIGN precomputes the propagation once and learns only the head: "
      "whatever it scores, the hop weighting earned it.")

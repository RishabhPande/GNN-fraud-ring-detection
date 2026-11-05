"""PNA vs GraphSAGE on the same fraud-ring graph.

Honest headline: PNA matches the repo's best models on the dense
ring (AUC 1.000, F1 1.000 vs SAGE mean's 0.978/0.857), joining
max-pool, LSTM, and GIN at the top -- and for the same reason GIN
wins: the extra towers let the model see *how many* suspicious
neighbors an account has (max and std over a dozen ring neighbors
look nothing like over one or two background neighbors), where the
flat mean washes that out. Same one-seed caveat as the original GIN
and LSTM headlines: no seed sweep yet, so this is a mechanism result
-- the towers distinguish counts and spreads the mean cannot -- not
a robustness claim.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_pna,
    train_sage,
    train_val_split,
)

gen = GraphGenerator(seed=7)
g = gen.generate(n=200, ring_size=18)
ring = set(gen.ring_nodes(g))
print(f"generated {g.n} accounts, {g.edge_count()} edges, "
      f"{len(ring)} planted ring members")

train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
y_true = [g.labels[v] for v in val_ids]

print("\ntraining PNA (2-layer, mean/max/min/std x 3 degree scalers)...")
pna, pna_hist = train_pna(g, train_ids, val_ids, hidden_dim=16,
                          lr=0.1, batch_size=64, epochs=60,
                          patience=12, seed=7)
print(f"  epochs run: {len(pna_hist['train_loss'])}, "
      f"best val loss: {min(pna_hist['val_loss']):.4f}, "
      f"delta (mean degree): {pna.delta:.2f}")

print("training GraphSAGE (2-layer, mean agg)...")
sage, _ = train_sage(g, train_ids, val_ids, hidden_dim=16,
                     k1=5, k2=5, lr=0.1, batch_size=64,
                     epochs=60, patience=12, seed=7)

scores = {
    "PNA (2-layer, 4 aggs x 3 scalers)": [pna.predict(g, [v])[v]
                                          for v in val_ids],
    "GraphSAGE (2-layer, mean agg)": [sage.predict(g, [v])[v]
                                      for v in val_ids],
}

print("\nvalidation metrics (per-model F1-tuned threshold):")
print(f"  {'model':<32}{'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
for name, s in scores.items():
    thr, m = best_threshold_f1(y_true, s)
    auc = evaluate_all(y_true, s)["auc"]
    print(f"  {name:<32}{thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{auc:>8.3f}")

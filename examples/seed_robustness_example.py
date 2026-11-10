"""Seed robustness: do the headline models survive fresh graphs?

The GIN and LSTM results in this repo were single-seed (seed 7) claims,
and the README said so plainly. The mean aggregator is known to be
fragile across seeds: it collapses to a degenerate checkpoint on two
of seeds 7-9 (validation BCE looks healthy, ranking broken). This
sweep asks whether GIN's sum and the LSTM aggregator share that
fragility, or whether they hold up where mean falls over.

Protocol: same as the per-model examples -- 200 accounts, 18 ring
members, 25% validation split -- with the graph, split, and training
seeds all set to the sweep seed, so each seed is a genuinely fresh
graph, split, and initialization. Max-pool is included as the
already-swept reference: it held AUC 1.000 on all three seeds in the
earlier check.

The second panel extends the same question to the thirteen models
added since: the GCN family and its descendants (GCN, SGC, APPNP,
GATv2, JK-GCN, ChebNet, GCNII, TAGCN, PNA, GPR-GNN, MixHop, H2GCN,
SIGN). Each runs its own per-model example schedule (100 epochs,
patience 25, lr 0.5 for the full-graph family; 60 epochs, patience
12 for GATv2 and PNA, which use the sampled/aggregator schedules),
so the numbers are the headline protocols re-seeded, not a new
tuning. The full sweep takes about eight minutes in pure Python.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_appnp,
    train_cheb,
    train_gatv2,
    train_gcn,
    train_gcnii,
    train_gin,
    train_gpr,
    train_h2gcn,
    train_jk,
    train_mixhop,
    train_pna,
    train_sage,
    train_sgc,
    train_sign,
    train_tag,
    train_val_split,
)

SEEDS = (7, 8, 9)

# Panel 1: the aggregators. 60 epochs, patience 12, lr 0.1, hidden 16.
PANEL1 = [
    ("GraphSAGE (mean agg)", "mean"),
    ("GraphSAGE (max-pool agg)", "pool"),
    ("GraphSAGE (LSTM agg)", "lstm"),
    ("GIN (sum agg)", "gin"),
]

# Panel 2: the later models, each on its per-model example schedule.
# (name, train fn, kwargs without graph/train_ids/val_ids/seed)
PANEL2 = [
    ("GCN", train_gcn,
     dict(hidden_dim=16, lr=0.5, batch_size=64, epochs=100,
          patience=25)),
    ("SGC", train_sgc,
     dict(hops=2, lr=0.5, batch_size=64, epochs=100, patience=25)),
    ("APPNP", train_appnp,
     dict(hidden_dim=16, steps=10, alpha=0.1, lr=0.5, batch_size=64,
          epochs=100, patience=25)),
    ("GATv2", train_gatv2,
     dict(hidden_dim=16, k1=5, k2=5, lr=0.5, batch_size=64,
          epochs=60, patience=12)),
    ("JK-GCN", train_jk,
     dict(hidden_dim=16, lr=0.5, batch_size=64, epochs=100,
          patience=25)),
    ("ChebNet", train_cheb,
     dict(hidden_dim=16, K=2, lr=0.5, batch_size=64, epochs=100,
          patience=25)),
    ("GCNII-4", train_gcnii,
     dict(hidden_dim=16, n_layers=4, lr=0.5, batch_size=64,
          epochs=100, patience=25)),
    ("TAGCN", train_tag,
     dict(hidden_dim=16, K=2, lr=0.5, batch_size=64, epochs=100,
          patience=25)),
    ("PNA", train_pna,
     dict(hidden_dim=16, lr=0.1, batch_size=64, epochs=60,
          patience=12)),
    ("GPR-GNN", train_gpr,
     dict(hidden_dim=16, steps=10, alpha=0.1, lr=0.5, batch_size=64,
          epochs=100, patience=25)),
    ("MixHop", train_mixhop,
     dict(hidden_dim=16, K=2, lr=0.5, batch_size=64, epochs=100,
          patience=25)),
    ("H2GCN", train_h2gcn,
     dict(hidden_dim=16, lr=0.5, batch_size=64, epochs=100,
          patience=25)),
    ("SIGN", train_sign,
     dict(hidden_dim=16, hops=2, lr=0.5, batch_size=64, epochs=100,
          patience=25)),
]


def _score(model, g, val_ids):
    y_true = [g.labels[v] for v in val_ids]
    scores = [model.predict(g, [v])[v] for v in val_ids]
    _, met = best_threshold_f1(y_true, scores)
    return evaluate_all(y_true, scores)["auc"], met["f1"]


def _print_row(name, row):
    aucs = [a for a, _ in row]
    print(f"  {name:<26}" + "".join(f"{a:.3f}/{f:.3f}".rjust(16)
                                    for a, f in row)
          + f"     {sum(aucs) / len(aucs):.3f}    {min(aucs):.3f}")


def _header():
    print(f"  {'model':<26}" + "".join(f"seed {s} AUC/F1".rjust(16)
                                       for s in SEEDS)
          + "     mean AUC   min AUC")


print("seed-robustness sweep: dense ring, seeds 7-9, per-model "
      "F1-tuned threshold")
print("\nPanel 1: aggregators (60 epochs, patience 12, lr 0.1)")
_header()
for name, kind in PANEL1:
    row = []
    for seed in SEEDS:
        gen = GraphGenerator(seed=seed)
        g = gen.generate(n=200, ring_size=18)
        train_ids, val_ids = train_val_split(g.labels, val_frac=0.25,
                                             seed=seed)
        if kind == "gin":
            model, _ = train_gin(g, train_ids, val_ids, hidden_dim=16,
                                 lr=0.1, batch_size=64, epochs=60,
                                 patience=12, seed=seed)
        else:
            model, _ = train_sage(g, train_ids, val_ids, hidden_dim=16,
                                  k1=5, k2=5, lr=0.1, batch_size=64,
                                  epochs=60, patience=12, seed=seed,
                                  aggregator=kind)
        row.append(_score(model, g, val_ids))
    _print_row(name, row)

print("\nPanel 2: GCN family and descendants (each on its example "
      "schedule)")
_header()
panel2 = {}
for name, fn, kwargs in PANEL2:
    row = []
    for seed in SEEDS:
        gen = GraphGenerator(seed=seed)
        g = gen.generate(n=200, ring_size=18)
        train_ids, val_ids = train_val_split(g.labels, val_frac=0.25,
                                             seed=seed)
        model, _ = fn(g, train_ids, val_ids, seed=seed, **kwargs)
        row.append(_score(model, g, val_ids))
    panel2[name] = row
    _print_row(name, row)

print("""
Mean aggregation is fragile, but it is no longer the only fragile
one. Across all three seeds only three panel-2 models never drop
below AUC 0.90: PNA at 1.000 on every seed, ChebNet (min 0.944),
and H2GCN (min 0.944); JK-GCN comes close (min 0.889 on seed 8).
The collapses are specific, not noise: SGC falls to 0.411 and
GCNII-4 to 0.506 on seed 9 after holding 0.88-0.94 on seeds 7-8,
GATv2 collapses on seed 8 (0.467) and degrades on seed 9 (0.689),
and GCN, APPNP, TAGCN, SIGN, GPR-GNN, and MixHop all sit just
under the 0.90 bar on seeds 8-9 (0.811-0.894). The pattern is not
depth or size: the survivors all keep a channel the normalized
neighborhood mix cannot wash out (PNA's count-aware towers and
degree scalers, H2GCN's ego block, ChebNet's per-term weights
including the identity term, JK's layer-1 concatenation), while
the models that only ever see a normalized mix of their neighbors
inherit the same fragility mean has. Three seeds is still a small
sweep, so this is an observation about a pattern, not a ranking.""")

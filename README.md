# gnn-fraud

Graph neural networks for fraud detection — **GraphSAGE from scratch in
pure Python, zero dependencies.**

One-hop features miss fraud rings. A ring of mule accounts can look
perfectly ordinary one account at a time: normal ages, normal balances,
normal transaction sizes. What gives them away is the *wiring* — the dense
pocket of accounts transacting with each other far more than chance
allows. `gnn-fraud` generates that pattern synthetically and learns it
with a two-layer GraphSAGE (mean aggregation), implemented end to end —
forward pass, backprop through both message-passing layers, neighbor
sampling, mini-batch SGD — with nothing but the standard library.

> **Disclaimer:** every account in this repo is synthetic. There are no
> real people, no real transactions, and no real PII anywhere here. The
> generator exists so GNN detectors can be built, tested, and benchmarked
> without touching sensitive data.

## Quickstart

```bash
python examples/quickstart.py
```

Or as a library:

```python
from gnn_fraud import GraphGenerator, train_sage, train_val_split

g = GraphGenerator(seed=7).generate(n=200, ring_size=18)
train_ids, val_ids = train_val_split(g.labels, seed=7)
model, history = train_sage(g, train_ids, val_ids)
probs = model.predict(g, val_ids)   # full-neighborhood inference
```

## How it works

1. **Generate** (`gnn_fraud/generator.py`) — N account nodes with six
   normalized features (account age, avg amount, txn count, balance,
   night-txn ratio, counterparties). Benign accounts wire up sparsely at
   random; a planted ring forms a dense subgraph. Ring members
   *near-camouflage*: most features match benign, with only subtle
   behavioral tells (slightly higher night activity, burst counts, more
   counterparties) — weak individually, damning once pooled.
2. **Message-pass** (`gnn_fraud/sage.py`) — two GraphSAGE layers,
   `h_v = relu(W_self·x_v + W_agg·mean(x_neighbors) + b)`, sigmoid output,
   binary cross-entropy with a positive-class weight (fraud data is
   imbalanced; without it the model collapses to the base rate — the same
   fix production systems use). Mini-batches sample up to k neighbors per
   node per layer, so training is O(batch·k²), not O(full 2-hop).
3. **Train** (`gnn_fraud/train.py`) — mini-batch SGD, stratified
   train/val split, early stopping on validation loss with best-checkpoint
   restore.
4. **Baseline** (`gnn_fraud/baseline.py`) — an MLP on node features only,
   same protocol. It sees each account's weak tells one noisy row at a
   time; the GNN denoises them over the ring's dense neighborhood.
5. **Score** (`gnn_fraud/evaluate.py`) — precision/recall/F1 with a
   per-model F1-tuned operating threshold (0.5 is the wrong cutoff under
   class imbalance) plus ROC-AUC, all from scratch.

Typical quickstart output (seed 7, 200 accounts, 18 ring members):

```
model                          thr precision  recall      F1     AUC
GraphSAGE (2-layer, mean agg)  0.28     1.000   0.750   0.857   0.978
MLP (features only)            0.41     1.000   0.250   0.400   0.683
```

## Roadmap

- [x] Two-layer GraphSAGE with mean aggregation, from scratch
- [x] Neighbor-sampled mini-batch training + early stopping
- [x] Feature-only MLP baseline showing the graph signal matters
- [ ] GAT attention layer (learn *which* neighbors matter, not just the mean)
- [ ] Heterogeneous graphs (accounts, devices, merchants as distinct node types)
- [ ] Inductive evaluation on unseen nodes (GraphSAGE's real strength)
- [ ] Temporal edges: rings form fast — add edge timestamps and recency weighting

## Tests

```bash
python3 -m pytest tests/ -q
```

Covers hand-computed forward/backward passes on a tiny graph, loss
decrease during training, ring planting + feature camouflage in the
generator, metric correctness on known cases, neighbor-sampling bounds,
and the headline claim: the GNN's AUC beats the feature-only baseline.

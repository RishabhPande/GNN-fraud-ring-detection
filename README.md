# GNN Fraud Ring Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RishabhPande/GNN-fraud-ring-detection/blob/main/examples/quickstart.py)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen)
![Tests](https://img.shields.io/badge/tests-100%20passing-brightgreen)

**GraphSAGE and GAT from scratch in pure Python. Zero dependencies. Built to catch fraud rings.**

One-hop features miss fraud rings. A ring of mule accounts can look perfectly ordinary one account at a time: normal ages, normal balances, normal transaction sizes. What gives them away is the *wiring*: a dense pocket of accounts transacting with each other far more than chance allows.

This repo generates that pattern synthetically and learns it with graph neural networks implemented end to end: forward pass, backprop through message-passing layers, neighbor sampling, mini-batch SGD. Nothing but the standard library.

> **Disclaimer:** every account in this repo is synthetic. No real people, no real transactions, no real PII anywhere. The generator exists so GNN detectors can be built, tested, and benchmarked without touching sensitive data.

## Try it in 30 seconds

Click **Open in Colab** above, or run locally:

```bash
python examples/quickstart.py
```

Or as a library:

```python
from gnn_fraud import GraphGenerator, train_sage, train_val_split

g = GraphGenerator(seed=7).generate(n=200, ring_size=18)
train_ids, val_ids = train_val_split(g.labels, seed=7)
model, history = train_sage(g, train_ids, val_ids)
probs = model.predict(g, val_ids)
```

## Results

Seed 7, 200 accounts, 18 ring members. The graph signal is the whole story:

```
model                          thr precision  recall      F1     AUC
GraphSAGE (2-layer, mean agg)  0.28     1.000   0.750   0.857   0.978
MLP (features only)            0.23     0.286   0.500   0.364   0.656
```

Same protocol with attention (`examples/gat_example.py`) and on a heterogeneous graph of accounts, devices, and merchants (`examples/hetero_example.py`):

```
GAT (2-layer, attention)            0.29     0.667   1.000   0.800   0.967
Relational GraphSAGE (hetero)       0.31     0.800   1.000   0.889   0.994
```

Multi-head GAT (`examples/multihead_gat_example.py`; longer schedule --
150 epochs, patience 40, lr 1.0 -- because the averaged final layer
splits each head's gradient K ways):

```
GAT (1-head)                        0.73     1.000   1.000   1.000   1.000
GAT (2-head)                        0.19     0.800   1.000   0.889   0.994
GAT (4-head)                        0.79     1.000   1.000   1.000   1.000
```

More heads match single-head ranking but buy nothing here: the four
heads learn nearly identical, near-uniform attention (mean pairwise
head L1/2 distance 0.022 on the top-scored ring member), so they
re-learn the same neighborhood mean four times instead of specializing.

With edge timestamps -- the ring's edges land in a 3-day burst at the end of a 30-day timeline -- recency-weighted aggregation (`examples/temporal_example.py`):

```
TemporalGraphSAGE (recency)         0.27     0.571   1.000   0.727   0.950
GraphSAGE (2-layer, mean agg)       0.28     1.000   0.750   0.857   0.978
MLP (features only)                 0.23     0.286   0.500   0.364   0.656
```

Inductive check: train on graph A, freeze the threshold, score a fresh graph B with no retraining. Ranking transfers (AUC 0.84 vs 0.98 transductive); calibration does not, so re-calibrate the threshold per deployment. See `examples/inductive_example.py`.

Edge classification on the same graph: freeze the GraphSAGE encoder, score each *flow* with an MLP head over endpoint-pair features (`examples/edge_example.py`):

```
EdgeHead (frozen SAGE embeddings)       0.48     0.857   1.000   0.923   0.951
```

A second fraud pattern -- money-mule chains (`examples/mule_chain_example.py`): originator -> 4 parallel mule chains -> consolidator -> exit. Thin paths, ~zero triangles, hop timestamps monotone downstream inside a late burst:

```
GraphSAGE (2-layer, mean agg)          0.40     0.800   1.000   0.889   0.978
TemporalGraphSAGE (recency)            0.39     0.800   1.000   0.889   0.983
MLP (features only)                    0.60     1.000   0.500   0.667   0.856
```

## What's inside

* **Generate** (`generator.py`): N account nodes, six normalized features. Benign accounts wire sparsely at random; a planted ring forms a dense subgraph with near-camouflaged features. Weak tells individually, damning once pooled. A second pattern, `generate_mule_chain()`, plants a laundering network instead: originator -> parallel mule chains -> consolidator -> exit -- thin paths, ~zero triangles, hop timestamps monotone downstream in a late burst -- with per-role feature tells (pass-through mules, orchestration-volume organizer, emptied exit).
* **Message-pass** (`sage.py`): two GraphSAGE layers with mean aggregation, sigmoid output, positive-class-weighted BCE (fraud data is imbalanced; without it the model collapses to the base rate). Neighbor-sampled mini-batches keep training O(batch*k^2).
* **Attend** (`gat.py`): two-layer multi-head GAT, full backprop through attention (numeric-gradient-checked per head). Layer 1 concatenates head outputs, layer 2 averages them (the GAT paper's convention); `n_heads=1` reproduces the classic single-head model bit-for-bit, and legacy single-head checkpoints still load. `attention_weights(v)` shows which counterparties drove an account's score (head-averaged); `attention_weights_per_head(v)` shows each head separately.
* **Relate** (`rsage.py`, `hetero.py`): R-GCN-lite over a heterogeneous graph. Accounts, devices, merchants as distinct node types; each relation gets its own message matrix, because a shared *device* means something different from a shared *counterparty*.
* **Time** (`temporal.py`): edges carry timestamps; ring edges land in a fast late burst while benign edges spread over the timeline. Each layer weights neighbors by recency (`softmax(-Δt/τ)`) with a learned, log-parametrized decay timescale, so aggregation concentrates on burst-coordinated neighbors.
* **Generalize** (`inductive.py`): inductive evaluation on unseen nodes. The honest protocol: frozen threshold from the training graph, no label leakage.
* **Classify edges** (`edge_cls.py`): score individual flows, not just accounts. An MLP head over `[h_u, h_v, h_u⊙h_v, |h_u−h_v|]` on *frozen* encoder embeddings, trained on edge-level mini-batches with full backprop through the head (numeric-gradient-checked). A ring-internal edge is one whose endpoints are both ring members.
* **Baseline** (`baseline.py`): feature-only MLP under the same protocol, so the graph signal is measured, not assumed.
* **Score** (`evaluate.py`): precision/recall/F1 with F1-tuned operating threshold plus ROC-AUC, all from scratch.

## Honest findings

* The GAT matches GraphSAGE, but its learned attention stays near-uniform on camouflaged rings: the signal lives in the neighborhood *mean*, not any single neighbor. Attention earns its keep where neighbors differ, which is what the heterogeneous graph provides.
* Multi-head attention does not rescue this: on camouflaged rings the heads do not specialize -- four heads learn the same diffuse distribution (pairwise L1/2 distance 0.022) -- so 4-head AUC ties single-head (1.000 vs 1.000) while training far slower. The averaged final layer dilutes each head's gradient K-fold, so under vanilla SGD the heads sit in a flat uniform-attention region for dozens of epochs; short-patience early stopping can then lock in a degenerate checkpoint that looks fine on validation BCE (0.35) but ranks backwards (AUC 0.256). Budget a long schedule (150 epochs, patience 40 here) and verify ranking, not just loss. Heads would earn their keep where different neighbor *kinds* matter -- e.g. one head for the dense pocket, another for feature outliers -- which the current generator does not provide.
* The hetero gap is wide because the generator plants a loud device-sharing pattern. Real rings launder through shared devices more subtly. The point is the mechanism, not the magnitude: per-relation weights learn *which kind* of neighbor carries the signal.
* Inductive ranking transfers across fresh graphs (AUC 0.84); score calibration does not (frozen-threshold F1 collapses 0.86 to 0.08). Ship the aggregation weights freely, re-calibrate the threshold per deployment.
* Edge classification is an easier task than node classification here, not a harder one: the head is a probe of frozen embeddings, so it mostly re-discovers the dense pocket (AUC 0.951 on held-out edges). The genuinely hard negatives are the bridge edges -- ring member to benign account -- which score halfway between (0.479 vs 0.511 ring-internal, 0.421 benign): one suspicious endpoint leaks signal. A production edge model would add flow features (amount, direction, timing) rather than leaning on structure alone.
* Recency weighting is not free: it concentrates aggregation on recent edges, which sharpens the burst but thins each node's effective neighborhood and adds variance. On this generator the dense structure is already a near-perfect signal, so the plain mean wins slightly (AUC 0.978 vs 0.950) while both crush the feature baseline (0.656). Recency earns its keep where old edges mislead -- dormant accounts, or histories that predate a compromise -- which is the next experiment, not this one. The learned timescale barely moves from its 5-day init: the loss is nearly flat in τ once the burst/background separation is strong, and its gradient (numeric-checked) is orders of magnitude smaller than the weight gradients.
* Money-mule chains rank about as well as dense pockets (AUC 0.98 both), but the graph signal's marginal value is smaller: the MLP climbs to 0.856 from 0.656 because the organizer/consolidator/exit roles carry stronger individual tells than camouflaged ring members. A production story about "structure over features" should pick its pattern honestly -- dense pockets are where the GNN earns the biggest premium. Thin chains also need patience: the default 30-epoch schedule only reached AUC 0.78, while 120 epochs reached 0.98. Recency weighting adds nothing on the chains (0.983 vs 0.978): on an undirected graph the sequential burst is already separated by structure alone.

## Roadmap

* [x] Two-layer GraphSAGE with mean aggregation, from scratch
* [x] Neighbor-sampled mini-batch training + early stopping
* [x] Feature-only MLP baseline showing the graph signal matters
* [x] GAT attention layer
* [x] Heterogeneous graphs (accounts, devices, merchants)
* [x] Inductive evaluation on unseen nodes
* [x] Temporal edges: rings form fast -- edge timestamps with recency-weighted aggregation and a learned decay timescale
* [x] Edge classification: score flows (ring-internal vs background edges) with an MLP head over frozen GNN embeddings
* [x] Multi-head GAT: independent attention heads (concat in layer 1, average in layer 2 per the GAT paper), per-head backprop with numeric gradient checks, per-head attention introspection, legacy checkpoint compatibility
* [x] Money-mule chains: a second fraud pattern -- originator -> parallel mule chains -> consolidator -> exit (thin paths, ~zero triangles, monotone burst hop timestamps), per-role feature tells, and a head-to-head with the dense pocket showing where the graph signal's premium actually lives

## Tests

```bash
python3 -m pytest tests/ -q
```

100 tests cover hand-computed forward/backward passes, gradient checks (including per-head multi-head attention in both concat and average modes), single-head bit-for-bit equivalence, legacy checkpoint loading, generator camouflage, metric correctness, sampling bounds, mule-chain topology/timestamp ordering/determinism, and the headline claims: GNN AUC beats the feature-only baseline on both the dense ring and the mule-chain patterns, inductive AUC stays above 0.80 on a fresh graph, and the edge head separates ring-internal flows (AUC 0.85+) on held-out edges.

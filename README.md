# GNN Fraud Ring Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RishabhPande/GNN-fraud-ring-detection/blob/main/examples/quickstart.py)
[![DOI](https://zenodo.org/badge/23749421.svg)](https://doi.org/10.5281/zenodo.23749421)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen)
![Tests](https://img.shields.io/badge/tests-168%20passing-brightgreen)
[![Tests](https://github.com/RishabhPande/GNN-fraud-ring-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/RishabhPande/GNN-fraud-ring-detection/actions/workflows/tests.yml)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/RishabhPande/GNN-fraud-ring-detection)

**GraphSAGE and GAT from scratch in pure Python. Zero dependencies. Built to catch fraud rings.**

One-hop features miss fraud rings. A ring of mule accounts can look perfectly ordinary one account at a time: normal ages, normal balances, normal transaction sizes. What gives them away is the *wiring*: a dense pocket of accounts transacting with each other far more than chance allows.

This repo generates that pattern synthetically and learns it with graph neural networks implemented end to end: forward pass, backprop through message-passing layers, neighbor sampling, mini-batch SGD. Nothing but the standard library.

> **Disclaimer:** every account in this repo is synthetic. No real people, no real transactions, no real PII anywhere. The generator exists so GNN detectors can be built, tested, and benchmarked without touching sensitive data.

## Seed robustness at a glance

| Seed-robustness sweep | Result |
|---|---|
| GIN + LSTM, AUC/F1, seeds 7-9 | 1.000 |
| Mean aggregation, AUC, seed 8 | 0.728 |
| Mean aggregation, AUC, seed 9 | 0.539 |

Test suite: 168 tests. Full per-seed table in Results below.

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
model                              thr precision  recall      F1     AUC
GraphSAGE (2-layer, mean agg)      0.28     1.000   0.750   0.857   0.978
GraphSAGE (2-layer, max-pool agg)  0.41     1.000   1.000   1.000   1.000
MLP (features only)                0.23     0.286   0.500   0.364   0.656
```

Same protocol with max-pool aggregation (`examples/pool_example.py`): each
neighbor is transformed through `relu(W_pool @ x_u + b_pool)` and the layer
keeps the elementwise max -- the single loudest tell per dimension, not the
neighborhood mean (the GraphSAGE paper's pooling aggregator, selected with
`aggregator="pool"`).

Same protocol with attention (`examples/gat_example.py`) and on a heterogeneous graph of accounts, devices, and merchants (`examples/hetero_example.py`):

```
GAT (2-layer, attention)            0.29     0.667   1.000   0.800   0.967
Relational GraphSAGE (hetero)       0.31     0.800   1.000   0.889   0.994
```

Same protocol with a GCN (`examples/gcn_example.py`; longer schedule --
100 epochs, patience 25, lr 0.5 -- because the normalized aggregation
shrinks gradients and SAGE's lr 0.1 underfits badly, AUC ~0.48-0.71):

```
GCN (2-layer, normalized)           0.31     0.500   1.000   0.667   0.944
```

GCN lands just under SAGE's mean model: the symmetric normalization
that defines it also dilutes the ring's own signal, since ring members
are the highest-degree nodes in the graph and their messages get
down-weighted the most. Still far above the feature-only baseline -- a
real architectural difference, not a tuning artifact, since both models
train at their own best learning rate.

Same protocol with an SGC (`examples/sgc_example.py`): delete the GCN's
hidden layer and ReLU, propagate the features twice through the
normalized adjacency, then logistic regression:

```
SGC (2-hop propagation)             0.28     0.500   1.000   0.667   0.928
GCN (2-layer, normalized)           0.31     0.500   1.000   0.667   0.944
MLP (features only)                 0.23     0.286   0.500   0.364   0.656
```

SGC lands within 0.016 AUC of the GCN it ablates, with the same F1:
on this graph the hidden layer buys almost nothing, and the propagated
features alone carry the signal. The cheaper, linear, easier-to-audit
model is the one to ship here.

Same protocol with a GIN (`examples/gin_example.py`):

```
GIN (2-layer, sum agg)              0.50     1.000   1.000   1.000   1.000
GraphSAGE (2-layer, mean agg)       0.28     1.000   0.750   0.857   0.978
```

GIN beats mean-SAGE outright here: sum aggregation keeps the
neighborhood *count* -- a ring member has a dozen ring neighbors,
a background account one or two -- which mean erases and GCN's
normalization dilutes. Learned eps stays near zero at lr 0.1
(layer1 -0.066, layer2 -0.029); lr 0.5 still ranks perfectly but
drives layer-2 eps to -2.19, so 0.1 is the default.

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
GraphSAGE (2-layer, max-pool agg)      0.50     0.800   1.000   0.889   0.978
TemporalGraphSAGE (recency)            0.39     0.800   1.000   0.889   0.983
MLP (features only)                    0.60     1.000   0.500   0.667   0.856
```

A third fraud pattern -- smurfing / structuring (`examples/smurfing_example.py`): one source splits a large sum into small sub-threshold amounts routed through 16 smurf intermediaries to a single collector. Fan-out/fan-in, not density and not a path: the source and the sink are hubs, every smurf hangs off exactly those two, no triangles at all, hop pairs fast and late in a burst:

```
GraphSAGE (2-layer, mean agg)          0.30     0.750   0.750   0.750   0.933
TemporalGraphSAGE (recency)            0.57     1.000   0.750   0.857   0.939
MLP (features only)                    0.43     0.400   0.500   0.444   0.672
```

Seed robustness (`examples/seed_robustness_example.py`): the same dense-ring
protocol, fresh graph + split + initialization per seed, AUC/F1 per seed:

```
model                          seed 7     seed 8     seed 9     mean  min
GraphSAGE (mean agg)           0.978/0.857  0.728/0.333  0.539/0.240  0.748  0.539
GraphSAGE (max-pool agg)       1.000/1.000  1.000/1.000  1.000/1.000  1.000  1.000
GraphSAGE (LSTM agg)           1.000/1.000  1.000/1.000  1.000/1.000  1.000  1.000
GIN (sum agg)                  1.000/1.000  1.000/1.000  1.000/1.000  1.000  1.000
```

Mean is the only fragile aggregator; the GIN and LSTM headline numbers
are no longer single-seed claims.

## What's inside

* **Generate** (`generator.py`): N account nodes, six normalized features. Benign accounts wire sparsely at random; a planted ring forms a dense subgraph with near-camouflaged features. Weak tells individually, damning once pooled. A second pattern, `generate_mule_chain()`, plants a laundering network instead: originator -> parallel mule chains -> consolidator -> exit -- thin paths, ~zero triangles, hop timestamps monotone downstream in a late burst -- with per-role feature tells (pass-through mules, orchestration-volume organizer, emptied exit). A third pattern, `generate_smurfing()`, plants a structuring network: one source -> N smurf intermediaries -> one collector -- fan-out/fan-in hubs and spokes, zero triangles, in-flow-before-out-flow hop pairs inside a late burst -- with per-role tells (small-amount pass-through smurfs, fan-out source, emptied collector).
* **Message-pass** (`sage.py`): two GraphSAGE layers with mean, max-pool, or LSTM aggregation, sigmoid output, positive-class-weighted BCE (fraud data is imbalanced; without it the model collapses to the base rate). Neighbor-sampled mini-batches keep training O(batch*k^2). The max-pool aggregator (the GraphSAGE paper's pooling aggregator) transforms each neighbor through `relu(W_pool @ x_u + b_pool)` -- He-initialized, since it feeds a ReLU -- then takes the elementwise max; backprop routes each dimension's gradient to its argmax neighbor. The LSTM aggregator (the paper's third aggregator) feeds neighbors through an LSTM in a pseudo-random order -- seeded from the neighbor vectors themselves, so the same neighborhood always takes the same order -- and takes the final hidden state as the aggregation, with full backprop through time (numeric-gradient-checked over all input, recurrent, and bias parameters) and a forget-gate bias initialized to 1.0. Mean-mode training is bit-for-bit unchanged (verified against the pre-pool code: identical loss trajectories and AUC).
* **Convolve** (`gcn.py`): two-layer GCN (Kipf & Welling) with symmetric degree normalization (`D_hat^{-1/2} A_hat D_hat^{-1/2}`, self-loops included), full-graph forward -- no sampling, since the normalization needs true degrees -- and full backprop through both layers (numeric-gradient-checked, layer and end-to-end). Needs a larger step than SAGE (lr 0.5 vs 0.1): the normalized aggregation shrinks gradients. Lands just under SAGE's mean model on the dense ring (AUC 0.944 vs 0.978) because ring members are the graph's hubs and normalization down-weights their messages most.
* **Simplify** (`sgc.py`): Simplified Graph Convolution (Wu et al.) -- K-hop symmetric-normalized propagation (`propagate_features`, `hops=0` degrades exactly to logistic regression on raw features), then a single linear head trained with weighted BCE. The GCN ablation: no hidden layer, no nonlinearity between propagations, full backprop replaced by logistic-regression gradients (numeric-gradient-checked).
* **Isomorph** (`gin.py`): two-layer GIN (Xu et al.) -- sum aggregation, a learnable `eps` weighting the node's own features, then a two-linear-layer MLP per layer (the paper's injectivity recipe: sum + MLP distinguishes neighbor multisets where mean and normalized sums cannot). Full-graph forward, full backprop (numeric-gradient-checked, layer and end-to-end, including `eps`). Beats mean-SAGE on the dense ring (AUC 1.000 vs 0.978, F1 1.000 vs 0.857) because the ring's signal is partly a count, and sum is the only aggregation here that keeps it.
* **Attend** (`gat.py`): two-layer multi-head GAT, full backprop through attention (numeric-gradient-checked per head). Layer 1 concatenates head outputs, layer 2 averages them (the GAT paper's convention); `n_heads=1` reproduces the classic single-head model bit-for-bit, and legacy single-head checkpoints still load. `attention_weights(v)` shows which counterparties drove an account's score (head-averaged); `attention_weights_per_head(v)` shows each head separately.
* **Attend, dynamically** (`gatv2.py`): two-layer multi-head GATv2 (Brody et al.) -- the score is `a . LeakyReLU(W_att @ [h_v || h_u])`, so the neighbor ranking can depend on the query node, where GAT's ranking cannot (its self term is constant across neighbors). Full backprop through the pair map (numeric-gradient-checked in concat and average modes, including a hand-computed ranking flip GAT cannot express). Same head-combination convention and introspection API as GAT.
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
* Max-pool aggregation matches or beats mean on both patterns (dense ring: AUC 1.000 vs 0.978, F1 1.000 vs 0.857; chains: tied at 0.978/0.889) -- the learned per-neighbor transform amplifies the tell direction before the max, so one strongly-suspicious neighbor suffices where mean needs the whole neighborhood to agree. It is also the stabler optimizer on the dense ring: across generator seeds 7-9 the mean model twice collapses to a degenerate checkpoint (validation BCE looks healthy, ranking broken: AUC 0.73 and 0.54 -- the same failure mode the multi-head GAT showed), while max-pool holds AUC 1.000 on all three seeds. The seed fragility is pre-existing, not introduced here: mean-mode training is bit-for-bit identical to the pre-pool code (verified: identical loss trajectories and AUC on both seeds). Two implementation details mattered: the pool transform needs its own bias (the paper has one; without it the transform starts half-dead) and He -- not Xavier -- initialization, since it feeds a ReLU.
* LSTM aggregation beats mean on the seed-7 dense ring (AUC 1.000 vs 0.978, F1 1.000 vs 0.857) -- the recurrence lets the summary weigh neighbors by position in the feed order instead of flat-averaging them, and every neighbor gets gradient (unlike max-pool's argmax routing). One seed only, so this is a mechanism result, not a robustness claim: unlike max-pool it has not been run across seeds 7-9, and the LSTM trains noticeably slower in pure Python (a per-neighbor cell vs a single transform), so it is the expensive way to buy the same headline number on this generator.
* GIN beats mean-SAGE on the seed-7 dense ring (AUC 1.000 vs 0.978, F1 1.000 vs 0.857) for the simplest reason in the repo: the ring's tell is partly how *many* suspicious neighbors an account has, and sum is the only aggregation here that preserves count. Same one-seed caveat as the LSTM result -- no seed sweep yet -- and one tuning note: lr 0.5 still ranks perfectly but drives layer-2 eps to -2.19, i.e. the model starts down-weighting its own features to compensate for the sum's scale, so the default stays 0.1 where eps stays near zero (-0.066, -0.029).
* Smurfing ranks about as well as the dense pocket and the chains (AUC 0.933 vs 0.978 both), but the graph premium is back to ring levels, not chain levels: the MLP stalls at 0.672 (vs 0.856 on chains) because the hubs' degree tell -- a source touching 16 smurfs, a sink collecting from 16 -- is invisible to a feature-only model, which sees only the smurfs' weak per-account tells (small amounts, pass-through balances). Density is the wrong lens for this pattern (0.209, between the ring's ~0.7 and the chains' thin paths): what identifies it is the hub/spoke degree split plus zero triangles. Recency adds almost nothing on top of mean (0.939 vs 0.933): the fan structure already separates the burst from background wiring, so timing is corroboration, not the signal. Like the chains, smurfing needs the long schedule -- the 30-epoch schedule only reaches AUC 0.850.
* SGC nearly matches the GCN it ablates (AUC 0.928 vs 0.944, same F1 0.667) on the seed-7 dense ring, so on this graph the GCN's hidden layer buys almost nothing: the signal is in the propagated features, not in a learned nonlinear transform of them. That is a one-seed result, like the original GIN and LSTM headlines before the seed sweep, so it is a shipping argument for this generator, not a general claim that SGC matches GCN everywhere. Where SGC does match, it is strictly cheaper to train, audit, and explain, since the whole model is one weight vector over propagated features.
* Seed robustness closes the GIN/LSTM one-seed caveat, and sharpens the mean-fragility story: across seeds 7-9, max-pool, LSTM, and GIN all hold AUC 1.000 / F1 1.000 on every seed, while mean collapses on two of three (AUC 0.728 on seed 8, 0.539 on seed 9 -- worse than the feature baseline's 0.656 on seed 7, and a genuinely backwards ranking, not noise). So the fragility is specific to mean aggregation, not to the training protocol: every model here trains on the same protocol, and only the flat average falls over. Three seeds is still a small sweep -- the claim is "holds where mean demonstrably breaks", not "holds everywhere".
* GATv2 edges out GAT on the seed-7 dense ring (AUC 0.978 vs 0.967, F1 0.889 vs 0.800) but not for the reason its name suggests: its trained layer-2 attention is perfectly diffuse (entropy gap 0.000 nats, every counterparty at 0.084), exactly like GAT's. The small gain comes from the different parameterization, not from selective listening, so dynamic attention joins static attention as a mechanism this generator does not reward: the ring signal is neighborhood-wide, and no ranking over neighbors can beat the mean at finding it. One seed only, like the original GAT headline.

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
* [x] Max-pool aggregator: the GraphSAGE paper's pooling aggregator (`aggregator="pool"` on `SAGELayer`/`GraphSAGE`/`train_sage`) -- per-neighbor `relu(W_pool @ x + b_pool)` transform with He init, elementwise max, argmax-routed backprop (numeric-gradient-checked); head-to-head vs mean on both patterns plus a seed-robustness check showing it avoids the mean model's degenerate checkpoints
* [x] LSTM aggregator: the GraphSAGE paper's third aggregator (`aggregator="lstm"` on `SAGELayer`/`GraphSAGE`/`train_sage`) -- neighbors in pseudo-random order (seeded from the neighbor vectors, so training and inference are reproducible) through a from-scratch LSTM cell, final hidden state as the aggregation, full BPTT (numeric-gradient-checked over every gate parameter), forget-bias 1.0 init, legacy checkpoint compatibility; on the seed-7 dense ring it beats mean (AUC 1.000 vs 0.978, F1 1.000 vs 0.857)
* [x] GCN: two-layer Graph Convolutional Network (Kipf & Welling, `gcn.py`/`train_gcn`) -- symmetric degree normalization (`D_hat^{-1/2} A_hat D_hat^{-1/2}` with self-loops), full-graph forward (no sampling: the normalization needs true degrees), full backprop (numeric-gradient-checked at the layer and end-to-end); needs lr 0.5 (SAGE's 0.1 underfits, AUC ~0.48-0.71); on the seed-7 dense ring it lands just under SAGE mean (AUC 0.944 vs 0.978, F1 0.667 vs 0.857) because ring members are the graph's hubs and normalization down-weights their messages most
* [x] GIN: two-layer Graph Isomorphism Network (Xu et al., `gin.py`/`train_gin`) -- sum aggregation, learnable `eps`, two-linear-layer MLP per layer, full-graph forward, full backprop (numeric-gradient-checked at the layer and end-to-end, including `eps`); on the seed-7 dense ring it beats SAGE mean (AUC 1.000 vs 0.978, F1 1.000 vs 0.857) because sum preserves the neighborhood count the ring's signal partly is; lr 0.1 (0.5 ranks the same but drives layer-2 eps to -2.19)
* [x] Smurfing / structuring: a third fraud pattern (`generate_smurfing()`) -- one source -> N smurf intermediaries -> one collector, fan-out/fan-in hubs and spokes with zero triangles, in-flow-before-out-flow hop pairs inside a late burst, per-role feature tells (small-amount pass-through smurfs, fan-out source, emptied collector); on the seed-7 graph SAGE mean reaches AUC 0.933 vs the MLP's 0.672, with recency adding almost nothing (0.939) -- the degree tell, invisible to features alone, is the signal
* [x] Seed-robustness sweep (`examples/seed_robustness_example.py`): the dense-ring protocol rerun on fresh graphs for seeds 7-9 (graph, split, and init all re-seeded) -- max-pool, LSTM, and GIN hold AUC 1.000 / F1 1.000 on all three seeds while mean collapses on seeds 8-9 (AUC 0.728, 0.539), retiring the GIN/LSTM one-seed caveats and showing the mean fragility is aggregator-specific, not protocol-wide
* [x] SGC: Simplified Graph Convolution (Wu et al., `sgc.py`/`train_sgc`) -- K-hop symmetric-normalized propagation then logistic regression, the GCN with its hidden layer and nonlinearity removed; on the seed-7 dense ring it lands within 0.016 AUC of the GCN (0.928 vs 0.944, same F1 0.667), showing the ring's signal lives in the propagated features themselves
* [x] GATv2: dynamic attention (Brody et al., `gatv2.py`/`train_gatv2`) -- score from `a . LeakyReLU(W_att @ [h_v || h_u])` so neighbor ranking can depend on the query node, full backprop numeric-gradient-checked in concat and average modes, hand-computed ranking flip that GAT cannot express; on the seed-7 dense ring it edges out GAT (AUC 0.978 vs 0.967, F1 0.889 vs 0.800) while its trained attention stays perfectly diffuse (entropy gap 0.000), so the gain is parameterization, not selective listening

## Tests

```bash
python3 -m pytest tests/ -q
```

168 tests cover hand-computed forward/backward passes, gradient checks (including per-head multi-head attention in both concat and average modes, the GATv2 pair-map backprop in concat and average modes plus its query-dependent ranking flip, the pool aggregator's argmax-routed backprop, the LSTM aggregator's BPTT over every gate parameter, the GCN layer plus the full two-layer GCN end to end, the GIN layer's sum aggregation, MLP, and learnable eps plus the full two-layer GIN end to end, and the SGC logistic head over propagated features), single-head bit-for-bit equivalence, mean-mode bit-for-bit equivalence after the pool refactor, legacy checkpoint loading, generator camouflage, metric correctness, sampling bounds, mule-chain topology/timestamp ordering/determinism, smurfing fan-out/fan-in topology (hub degrees, zero triangles, hop-pair timestamp ordering)/determinism, SGC propagation hand-computed on two-node, star, and isolated-node graphs plus its hops=0 logistic-regression equivalence, and the headline claims: GNN AUC beats the feature-only baseline on the dense ring, the mule-chain, and the smurfing patterns, the pool aggregator trains to AUC 0.95+ on the dense ring, the LSTM aggregator trains to AUC 0.90+ on the dense ring, the GCN trains to AUC 0.90+ on the dense ring, the GIN trains to AUC 0.90+ on the dense ring, the SGC trains to AUC 0.90+ on the dense ring, the GATv2 trains to AUC 0.90+ on the dense ring, inductive AUC stays above 0.80 on a fresh graph, the edge head separates ring-internal flows (AUC 0.85+) on held-out edges, and GIN plus LSTM hold AUC/F1 0.90+ on seed 9, the seed where mean aggregation collapses hardest (AUC 0.539).

## Contributing

Contributions welcome: see CONTRIBUTING.md.

## Part of a fraud-ML series

- [Agentic Fraud Investigator](https://github.com/RishabhPande/agentic-fraud-investigator) - A multi-agent fraud investigation system in pure Python, zero dependencies.
- [ID-document-forgery-detection-CV-](https://github.com/RishabhPande/ID-document-forgery-detection-CV-) - Classic image forensics for ID-document forgery detection, written from scratch in pure Python, zero dependencies.
- [Time-series-anomaly-detection](https://github.com/RishabhPande/Time-series-anomaly-detection) - Time-series anomaly detection, including collective anomalies
- [fraud-eval](https://github.com/RishabhPande/fraud-eval) - Open benchmark for evaluating LLM-based fraud and abuse detectors (20-case smoke test, cost-aware metrics, counterfactual evaluation)
- [fraud-rings](https://github.com/RishabhPande/fraud-rings) - Synthetic identity-fraud ring generator plus graph-based detection, all in pure Python, no dependencies.

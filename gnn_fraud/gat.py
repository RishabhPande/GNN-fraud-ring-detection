"""Graph Attention Network (multi-head) from scratch in pure Python.

Layer update for node ``v`` over its candidate set
``C(v) = {v} ∪ N(v)`` (self-attention keeps the node's own signal).
Each head ``k`` runs its own attention::

    Wh^k_u      = W^k @ h_u                    for u in C(v)
    e^k_vu      = LeakyReLU(a_self^k · Wh^k_v + a_neigh^k · Wh^k_u)
    α^k_vu      = softmax_u(e^k_vu)
    agg^k_v     = Σ_u α^k_vu · Wh^k_u

Heads are combined the way the original GAT paper prescribes: the
hidden layer concatenates head outputs, the final layer averages them::

    concat:  h'_v = relu([agg^1_v + b^1 || ... || agg^K_v + b^K])
    average: h'_v = relu((1/K) Σ_k (agg^k_v + b^k))

With ``n_heads=1`` (the default) the "concat" math is exactly the classic
single-head update, so existing checkpoints and behavior are preserved.
Why multiple heads: on camouflaged rings a single head converges to the
neighborhood mean (documented in ``examples/gat_example.py``). Heads are
independent attention functions, so different heads can specialize --
one concentrating on the dense ring pocket, another on feature outliers --
instead of one head averaging everything.

Unlike GraphSAGE's uniform mean, GAT *learns* how much each neighbor
matters. For fraud rings that means the layer can downweight the dozens
of noisy benign edges and concentrate on the dense ring pocket -- and
because the α_vu are probabilities, the learned weights double as an
explanation: "this account was flagged mostly because of *these*
counterparties".

All linear algebra is hand-rolled lists-of-floats -- no numpy, no torch.
"""

from __future__ import annotations

import copy
import math
import random


# ---------------------------------------------------------------------------
# tiny linalg helpers (lists of floats)
# ---------------------------------------------------------------------------


def _matvec(W: list[list[float]], x: list[float]) -> list[float]:
    return [sum(w * xi for w, xi in zip(row, x)) for row in W]


def _mat_t_vec(W: list[list[float]], v: list[float]) -> list[float]:
    n_cols = len(W[0])
    return [sum(W[r][c] * v[r] for r in range(len(W))) for c in range(n_cols)]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _vec_add(a: list[float], b: list[float]) -> list[float]:
    return [ai + bi for ai, bi in zip(a, b)]


def _scale(v: list[float], s: float) -> list[float]:
    return [vi * s for vi in v]


def _relu(v: list[float]) -> list[float]:
    return [x if x > 0.0 else 0.0 for x in v]


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _leaky_relu(z: float, slope: float) -> float:
    return z if z > 0.0 else slope * z


def _bce(p: float, y: int) -> float:
    p = min(max(p, 1e-7), 1.0 - 1e-7)
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


# ---------------------------------------------------------------------------
# one message-passing layer (K attention heads)
# ---------------------------------------------------------------------------


class GATLayer:
    """Multi-head graph attention layer with ReLU output.

    Args:
        n_heads: number of independent attention heads. Each head gets
            ``out_dim // n_heads`` output dimensions.
        out_mode: how head outputs combine -- ``"concat"`` (hidden-layer
            style, output dimension = out_dim) or ``"average"``
            (final-layer style, output dimension = out_dim // n_heads,
            gradients split evenly across heads).
    """

    def __init__(self, in_dim: int, out_dim: int, rng: random.Random,
                 negative_slope: float = 0.2, n_heads: int = 1,
                 out_mode: str = "concat"):
        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}")
        if out_mode not in ("concat", "average"):
            raise ValueError(f"out_mode must be 'concat' or 'average', "
                             f"got {out_mode!r}")
        if out_dim % n_heads != 0:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by "
                             f"n_heads ({n_heads})")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.out_mode = out_mode
        self.negative_slope = negative_slope
        self.heads: list[dict] = []
        for _ in range(n_heads):
            # Attention is scored on the *transformed* features, as in the
            # GAT paper: a^T [Wh_v || Wh_u], split here into a_self and
            # a_neigh.
            scale = math.sqrt(2.0 / (in_dim + self.head_dim))  # Xavier-ish
            W = [[rng.uniform(-scale, scale) for _ in range(in_dim)]
                 for _ in range(self.head_dim)]
            attn_scale = math.sqrt(1.0 / self.head_dim)
            head = {
                "W": W,
                "a_self": [rng.uniform(-attn_scale, attn_scale)
                           for _ in range(self.head_dim)],
                "a_neigh": [rng.uniform(-attn_scale, attn_scale)
                            for _ in range(self.head_dim)],
                "b": [0.0] * self.head_dim,
            }
            self.heads.append(head)
        self.zero_grad()

    def zero_grad(self) -> None:
        self.grads = []
        for _ in range(self.n_heads):
            self.grads.append({
                "W": [[0.0] * self.in_dim for _ in range(self.head_dim)],
                "a_self": [0.0] * self.head_dim,
                "a_neigh": [0.0] * self.head_dim,
                "b": [0.0] * self.head_dim,
            })

    # -- legacy parameter access (kept for n_heads=1 ergonomics) -------------

    @property
    def W(self):
        return self.heads[0]["W"]

    @W.setter
    def W(self, value):
        self.heads[0]["W"] = value

    @property
    def a_self(self):
        return self.heads[0]["a_self"]

    @a_self.setter
    def a_self(self, value):
        self.heads[0]["a_self"] = value

    @property
    def a_neigh(self):
        return self.heads[0]["a_neigh"]

    @a_neigh.setter
    def a_neigh(self, value):
        self.heads[0]["a_neigh"] = value

    @property
    def b(self):
        return self.heads[0]["b"]

    @b.setter
    def b(self, value):
        self.heads[0]["b"] = value

    @property
    def gW(self):
        return self.grads[0]["W"]

    @property
    def ga_self(self):
        return self.grads[0]["a_self"]

    @property
    def ga_neigh(self):
        return self.grads[0]["a_neigh"]

    @property
    def gb(self):
        return self.grads[0]["b"]

    # -- forward ------------------------------------------------------------

    def _head_forward(self, head: dict, v: int,
                      h: dict[int, list[float]],
                      cand: list[int]) -> dict:
        W, a_self, a_neigh = head["W"], head["a_self"], head["a_neigh"]
        Wh = {u: _matvec(W, h[u]) for u in cand}
        self_score = _dot(a_self, Wh[v])
        logits = {u: _leaky_relu(self_score + _dot(a_neigh, Wh[u]),
                                 self.negative_slope)
                  for u in cand}
        # Softmax over the candidate set.
        m = max(logits.values())
        exp = {u: math.exp(z - m) for u, z in logits.items()}
        total = sum(exp.values())
        alphas = {u: e / total for u, e in exp.items()}

        agg = [0.0] * self.head_dim
        for u in cand:
            w = alphas[u]
            wh = Wh[u]
            for i in range(self.head_dim):
                agg[i] += w * wh[i]
        pre = _vec_add(agg, head["b"])
        return {"Wh": Wh, "logits": logits, "alphas": alphas, "pre": pre}

    def forward(self, v: int, h: dict[int, list[float]],
                neighbors: list[int]):
        """Returns (output, cache) where cache holds everything backprop
        needs. ``h`` maps candidate nodes to their input embeddings;
        ``neighbors`` are the sampled neighbors of ``v`` (self-attention
        is added automatically)."""
        cand = sorted({v} | set(neighbors))
        head_caches = [self._head_forward(head, v, h, cand)
                       for head in self.heads]
        if self.out_mode == "concat":
            pre: list[float] = []
            for hc in head_caches:
                pre.extend(hc["pre"])
        else:  # average
            pre = [0.0] * self.head_dim
            for hc in head_caches:
                for i in range(self.head_dim):
                    pre[i] += hc["pre"][i] / self.n_heads
        out = _relu(pre)
        cache = {"v": v, "cand": cand, "h": h,
                 "head_caches": head_caches, "pre": pre}
        return out, cache

    # -- backward -----------------------------------------------------------

    def _head_backward(self, head: dict, grad: dict, d_pre: list[float],
                       v: int, cand: list[int],
                       h: dict[int, list[float]], hc: dict) -> dict:
        """Backprop through one head. Returns {node: d_h contribution}."""
        W, a_self, a_neigh = head["W"], head["a_self"], head["a_neigh"]
        Wh, logits, alphas = hc["Wh"], hc["logits"], hc["alphas"]
        gW, ga_self, ga_neigh, gb = (grad["W"], grad["a_self"],
                                    grad["a_neigh"], grad["b"])
        for i in range(self.head_dim):
            gb[i] += d_pre[i]

        # Gradient of the loss w.r.t. each alpha via the weighted sum.
        d_alpha = {u: _dot(d_pre, Wh[u]) for u in cand}
        # Softmax jacobian: dL/dz_u = α_u (dL/dα_u − Σ_k α_k dL/dα_k).
        s = sum(alphas[u] * d_alpha[u] for u in cand)
        d_z = {u: alphas[u] * (d_alpha[u] - s) for u in cand}

        dWh = {u: [0.0] * self.head_dim for u in cand}
        for u in cand:
            # Weighted-sum contribution: out += α_u · Wh_u.
            for i in range(self.head_dim):
                dWh[u][i] += alphas[u] * d_pre[i]
            # Attention-path contribution.
            z = logits[u]
            d_logit = d_z[u] * (1.0 if z > 0.0 else self.negative_slope)
            for i in range(self.head_dim):
                ga_self[i] += d_logit * Wh[v][i]
                ga_neigh[i] += d_logit * Wh[u][i]
                dWh[v][i] += d_logit * a_self[i]
                dWh[u][i] += d_logit * a_neigh[i]

        d_h = {u: _mat_t_vec(W, dWh[u]) for u in cand}
        # Accumulate parameter gradients for W: dL/dW[i][j] = Σ_u dWh[u][i]·h[u][j].
        for i in range(self.head_dim):
            row = gW[i]
            for j in range(self.in_dim):
                row[j] += sum(dWh[u][i] * h[u][j] for u in cand)
        return d_h

    def backward(self, d_out: list[float], cache) -> dict[int, list[float]]:
        """Backprop one step. Returns {node: gradient w.r.t. input embedding}."""
        v, cand = cache["v"], cache["cand"]
        h, head_caches, pre = (cache["h"], cache["head_caches"],
                               cache["pre"])
        d_pre = [d * (1.0 if p > 0.0 else 0.0) for d, p in zip(d_out, pre)]

        d_h: dict[int, list[float]] = {u: [0.0] * self.in_dim for u in cand}
        for k in range(self.n_heads):
            if self.out_mode == "concat":
                d_pre_k = d_pre[k * self.head_dim:(k + 1) * self.head_dim]
            else:  # average: each head saw pre/K, so its share is d_pre/K
                d_pre_k = [d / self.n_heads for d in d_pre]
            d_h_k = self._head_backward(self.heads[k], self.grads[k],
                                        d_pre_k, v, cand, h, head_caches[k])
            for u, g in d_h_k.items():
                d_h[u] = _vec_add(d_h[u], g)
        return d_h

    def step(self, lr: float) -> None:
        for head, grad in zip(self.heads, self.grads):
            for i in range(self.head_dim):
                for j in range(self.in_dim):
                    head["W"][i][j] -= lr * grad["W"][i][j]
                head["b"][i] -= lr * grad["b"][i]
                head["a_self"][i] -= lr * grad["a_self"][i]
                head["a_neigh"][i] -= lr * grad["a_neigh"][i]

    def state_dict(self):
        return copy.deepcopy({"n_heads": self.n_heads,
                              "out_mode": self.out_mode,
                              "heads": self.heads,
                              "negative_slope": self.negative_slope})

    def load_state_dict(self, state) -> None:
        if "heads" not in state and self.n_heads == 1:
            # Legacy single-head checkpoint format: W/a_self/a_neigh/b
            # stored flat on the layer.
            self.heads[0] = {
                "W": copy.deepcopy(state["W"]),
                "a_self": copy.deepcopy(state["a_self"]),
                "a_neigh": copy.deepcopy(state["a_neigh"]),
                "b": copy.deepcopy(state["b"]),
            }
            return
        self.heads = copy.deepcopy(state["heads"])


# ---------------------------------------------------------------------------
# two-layer model with neighbor sampling
# ---------------------------------------------------------------------------


class GraphAttention:
    """Two-layer multi-head GAT for binary node classification.

    Layer 1 concatenates head outputs (hidden-layer style); layer 2
    averages them (final-layer style, as in the original GAT paper).
    ``n_heads=1`` reproduces the classic single-head behavior exactly.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 8,
        lr: float = 0.1,
        seed: int = 0,
        negative_slope: float = 0.2,
        n_heads: int = 1,
    ):
        self.rng = random.Random(seed)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.negative_slope = negative_slope
        self.n_heads = n_heads
        self.layer1 = GATLayer(in_dim, hidden_dim, self.rng,
                              negative_slope=negative_slope,
                              n_heads=n_heads, out_mode="concat")
        # Single-head keeps the legacy concat+relu tail so old checkpoints
        # and exact single-head behavior are preserved; multi-head averages
        # head outputs in the final layer, per the GAT paper.
        self.layer2 = GATLayer(hidden_dim, hidden_dim, self.rng,
                               negative_slope=negative_slope,
                               n_heads=n_heads,
                               out_mode="concat" if n_heads == 1 else "average")
        # The final layer averages heads (per the GAT paper), so its
        # output dimension is hidden_dim // n_heads.
        self.out_feat = hidden_dim if n_heads == 1 else hidden_dim // n_heads
        scale = math.sqrt(2.0 / (self.out_feat + 1))
        self.w_out = [self.rng.uniform(-scale, scale)
                      for _ in range(self.out_feat)]
        self.b_out = 0.0
        self.gw_out = [0.0] * self.out_feat
        self.gb_out = 0.0

    # -- sampling --------------------------------------------------------

    def _sample(self, candidates: list[int], k: int | None) -> list[int]:
        if k is None or len(candidates) <= k:
            return list(candidates)
        return self.rng.sample(candidates, k)

    def _expand_batch(self, graph, targets: list[int], k1: int | None,
                      k2: int | None):
        """Sample the 2-hop computation graph for a mini-batch.

        Returns (s2, s1): s2[v] = layer-2 sampled neighbors of target v;
        s1[x] = layer-1 sampled neighbors for every node x whose layer-1
        embedding is needed (targets + their s2 nodes).
        """
        s2: dict[int, list[int]] = {}
        layer1_nodes: set[int] = set(targets)
        for v in targets:
            s2[v] = self._sample(graph.neighbors(v), k2)
            layer1_nodes.update(s2[v])
        s1: dict[int, list[int]] = {}
        for x in layer1_nodes:
            s1[x] = self._sample(graph.neighbors(x), k1)
        return s2, s1

    # -- forward ----------------------------------------------------------

    def forward(self, graph, targets: list[int],
                k1: int | None = 3, k2: int | None = 3):
        """Forward pass. Returns (probs dict, cache)."""
        s2, s1 = self._expand_batch(graph, targets, k1, k2)

        h1: dict[int, list[float]] = {}
        cache1: dict[int, tuple] = {}
        for x in s1:
            h_x = {u: graph.features[u] for u in ({x} | set(s1[x]))}
            h1[x], cache1[x] = self.layer1.forward(x, h_x, s1[x])

        h2: dict[int, list[float]] = {}
        cache2: dict[int, tuple] = {}
        for v in targets:
            h_v = {u: h1[u] for u in ({v} | set(s2[v]))}
            h2[v], cache2[v] = self.layer2.forward(v, h_v, s2[v])

        probs: dict[int, float] = {}
        for v in targets:
            z = sum(w * h for w, h in zip(self.w_out, h2[v])) + self.b_out
            probs[v] = _sigmoid(z)

        cache = {"s2": s2, "s1": s1, "h1": h1, "cache1": cache1,
                 "h2": h2, "cache2": cache2, "targets": list(targets)}
        return probs, cache

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        """Inference with full (unsampled) neighborhoods -- deterministic."""
        probs, _ = self.forward(graph, targets, k1=None, k2=None)
        return probs

    def _attention_per_head(self, graph, v: int):
        """Per-layer, per-head attention distributions over v's full
        neighborhood. Returns [[head dicts...] (layer 1),
        [head dicts...] (layer 2)]; each dict sums to 1. Deterministic."""
        neighs = graph.neighbors(v)

        def run_layer(layer, h_in, center, neigh):
            cand_map = {u: h_in[u] for u in ({center} | set(neigh))}
            _, cache = layer.forward(center, cand_map, neigh)
            return [dict(hc["alphas"]) for hc in cache["head_caches"]]

        h_x = {u: graph.features[u] for u in ({v} | set(neighs))}
        l1 = run_layer(self.layer1, h_x, v, neighs)
        h1 = {}
        for x in ({v} | set(neighs)):
            hx = {u: graph.features[u] for u in ({x} | set(graph.neighbors(x)))}
            h1[x], _ = self.layer1.forward(x, hx, graph.neighbors(x))
        h_v = {u: h1[u] for u in ({v} | set(neighs))}
        l2 = run_layer(self.layer2, h_v, v, neighs)
        return [l1, l2]

    def attention_weights_per_head(self, graph, v: int):
        """Per-layer, per-head attention distributions over v's full
        neighborhood -- which counterparties each head listens to.
        Deterministic. See :meth:`attention_weights` for the
        head-averaged summary."""
        return self._attention_per_head(graph, v)

    def attention_weights(self, graph, v: int) -> list[dict[int, float]]:
        """Per-layer attention distributions over v's full neighborhood.

        For multi-head models this averages across heads, so the result is
        still a probability distribution per layer (sums to 1). Returns
        [layer1_alphas, layer2_alphas]. Deterministic; use
        :meth:`attention_weights_per_head` for per-head detail.
        """
        per_head = self._attention_per_head(graph, v)
        averaged = []
        for layer_heads in per_head:
            agg: dict[int, float] = {}
            for head_alphas in layer_heads:
                for u, a in head_alphas.items():
                    agg[u] = agg.get(u, 0.0) + a / len(layer_heads)
            averaged.append(agg)
        return averaged

    # -- training step -----------------------------------------------------

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    k1: int | None = 3, k2: int | None = 3,
                    pos_weight: float = 1.0) -> float:
        """One SGD step on a mini-batch. Returns mean weighted BCE loss.

        Args:
            pos_weight: multiplier on the loss of positive (fraud) examples.
                Fraud data is heavily imbalanced; without upweighting,
                the model collapses to predicting the base rate for
                everything. Standard practice in production fraud models.
        """
        self.layer1.zero_grad()
        self.layer2.zero_grad()
        self.gw_out = [0.0] * self.out_feat
        self.gb_out = 0.0

        probs, cache = self.forward(graph, targets, k1, k2)
        s2, s1 = cache["s2"], cache["s1"]
        h1, cache1 = cache["h1"], cache["cache1"]
        h2, cache2 = cache["h2"], cache["cache2"]

        B = len(targets)
        weights = {v: (pos_weight if labels[v] == 1 else 1.0) for v in targets}
        loss = sum(weights[v] * _bce(probs[v], labels[v])
                   for v in targets) / B

        # Output layer grads.
        dh2: dict[int, list[float]] = {}
        for v in targets:
            d_logit = weights[v] * (probs[v] - labels[v]) / B
            for j in range(self.out_feat):
                self.gw_out[j] += d_logit * h2[v][j]
            self.gb_out += d_logit
            dh2[v] = _scale(self.w_out, d_logit)

        # Backprop through layer 2; accumulate dh1 over layer-1 nodes.
        dh1: dict[int, list[float]] = {x: [0.0] * self.hidden_dim for x in s1}
        for v in targets:
            d_h2 = self.layer2.backward(dh2[v], cache2[v])
            for x, g in d_h2.items():
                dh1[x] = _vec_add(dh1[x], g)

        # Backprop through layer 1 (node features are leaves -- no need to
        # propagate into them).
        for x in s1:
            self.layer1.backward(dh1[x], cache1[x])

        # SGD update.
        self.layer1.step(self.lr)
        self.layer2.step(self.lr)
        for j in range(self.out_feat):
            self.w_out[j] -= self.lr * self.gw_out[j]
        self.b_out -= self.lr * self.gb_out
        return loss

    # -- checkpointing ------------------------------------------------------

    def state_dict(self):
        return {
            "layer1": self.layer1.state_dict(),
            "layer2": self.layer2.state_dict(),
            "w_out": copy.deepcopy(self.w_out),
            "b_out": self.b_out,
        }

    def load_state_dict(self, state) -> None:
        self.layer1.load_state_dict(state["layer1"])
        self.layer2.load_state_dict(state["layer2"])
        self.w_out = copy.deepcopy(state["w_out"])
        self.b_out = state["b_out"]

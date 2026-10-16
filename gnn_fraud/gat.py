"""Graph Attention Network (single-head) from scratch in pure Python.

Layer update for node ``v`` over its candidate set
``C(v) = {v} ∪ N(v)`` (self-attention keeps the node's own signal)::

    Wh_u      = W @ h_u                       for u in C(v)
    e_vu      = LeakyReLU(a_self · Wh_v + a_neigh · Wh_u)
    α_vu      = softmax_u(e_vu)
    h'_v      = relu(Σ_u α_vu · Wh_u + b)

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
# one message-passing layer
# ---------------------------------------------------------------------------


class GATLayer:
    """Single-head graph attention layer with ReLU output."""

    def __init__(self, in_dim: int, out_dim: int, rng: random.Random,
                 negative_slope: float = 0.2):
        scale = math.sqrt(2.0 / (in_dim + out_dim))  # Xavier-ish init
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.negative_slope = negative_slope
        self.W = [
            [rng.uniform(-scale, scale) for _ in range(in_dim)]
            for _ in range(out_dim)
        ]
        # Attention is scored on the *transformed* features, as in the GAT
        # paper: a^T [Wh_v || Wh_u], split here into a_self and a_neigh.
        attn_scale = math.sqrt(1.0 / out_dim)
        self.a_self = [rng.uniform(-attn_scale, attn_scale)
                       for _ in range(out_dim)]
        self.a_neigh = [rng.uniform(-attn_scale, attn_scale)
                        for _ in range(out_dim)]
        self.b = [0.0] * out_dim
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW = [[0.0] * self.in_dim for _ in range(self.out_dim)]
        self.ga_self = [0.0] * self.out_dim
        self.ga_neigh = [0.0] * self.out_dim
        self.gb = [0.0] * self.out_dim

    def forward(self, v: int, h: dict[int, list[float]],
                neighbors: list[int]):
        """Returns (output, cache) where cache holds everything backprop
        needs. ``h`` maps candidate nodes to their input embeddings;
        ``neighbors`` are the sampled neighbors of ``v`` (self-attention
        is added automatically)."""
        cand = sorted({v} | set(neighbors))
        Wh = {u: _matvec(self.W, h[u]) for u in cand}

        self_score = _dot(self.a_self, Wh[v])
        logits = {u: _leaky_relu(self_score + _dot(self.a_neigh, Wh[u]),
                                 self.negative_slope)
                  for u in cand}
        # Softmax over the candidate set.
        m = max(logits.values())
        exp = {u: math.exp(z - m) for u, z in logits.items()}
        total = sum(exp.values())
        alphas = {u: e / total for u, e in exp.items()}

        agg = [0.0] * self.out_dim
        for u in cand:
            w = alphas[u]
            wh = Wh[u]
            for i in range(self.out_dim):
                agg[i] += w * wh[i]
        pre = _vec_add(agg, self.b)
        out = _relu(pre)
        cache = {"v": v, "cand": cand, "h": h, "Wh": Wh, "logits": logits,
                 "alphas": alphas, "pre": pre}
        return out, cache

    def backward(self, d_out: list[float], cache) -> dict[int, list[float]]:
        """Backprop one step. Returns {node: gradient w.r.t. input embedding}."""
        v, cand = cache["v"], cache["cand"]
        h, Wh, logits, alphas, pre = (cache["h"], cache["Wh"], cache["logits"],
                                     cache["alphas"], cache["pre"])
        d_pre = [d * (1.0 if p > 0.0 else 0.0) for d, p in zip(d_out, pre)]
        for i in range(self.out_dim):
            self.gb[i] += d_pre[i]

        # Gradient of the loss w.r.t. each alpha via the weighted sum.
        d_alpha = {u: _dot(d_pre, Wh[u]) for u in cand}
        # Softmax jacobian: dL/dz_u = α_u (dL/dα_u − Σ_k α_k dL/dα_k).
        s = sum(alphas[u] * d_alpha[u] for u in cand)
        d_z = {u: alphas[u] * (d_alpha[u] - s) for u in cand}

        dWh = {u: [0.0] * self.out_dim for u in cand}
        for u in cand:
            # Weighted-sum contribution: out += α_u · Wh_u.
            for i in range(self.out_dim):
                dWh[u][i] += alphas[u] * d_pre[i]
            # Attention-path contribution.
            z = logits[u]
            d_logit = d_z[u] * (1.0 if z > 0.0 else self.negative_slope)
            for i in range(self.out_dim):
                self.ga_self[i] += d_logit * Wh[v][i]
                self.ga_neigh[i] += d_logit * Wh[u][i]
                dWh[v][i] += d_logit * self.a_self[i]
                dWh[u][i] += d_logit * self.a_neigh[i]

        d_h = {u: _mat_t_vec(self.W, dWh[u]) for u in cand}
        # Accumulate parameter gradients for W: dL/dW[i][j] = Σ_u dWh[u][i]·h[u][j].
        for i in range(self.out_dim):
            row = self.gW[i]
            for j in range(self.in_dim):
                row[j] += sum(dWh[u][i] * h[u][j] for u in cand)
        return d_h

    def step(self, lr: float) -> None:
        for i in range(self.out_dim):
            for j in range(self.in_dim):
                self.W[i][j] -= lr * self.gW[i][j]
            self.b[i] -= lr * self.gb[i]
            self.a_self[i] -= lr * self.ga_self[i]
            self.a_neigh[i] -= lr * self.ga_neigh[i]

    def state_dict(self):
        return copy.deepcopy({"W": self.W, "a_self": self.a_self,
                              "a_neigh": self.a_neigh, "b": self.b,
                              "negative_slope": self.negative_slope})

    def load_state_dict(self, state) -> None:
        self.W = copy.deepcopy(state["W"])
        self.a_self = copy.deepcopy(state["a_self"])
        self.a_neigh = copy.deepcopy(state["a_neigh"])
        self.b = copy.deepcopy(state["b"])


# ---------------------------------------------------------------------------
# two-layer model with neighbor sampling
# ---------------------------------------------------------------------------


class GraphAttention:
    """Two-layer GAT (single head) for binary node classification."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 8,
        lr: float = 0.1,
        seed: int = 0,
        negative_slope: float = 0.2,
    ):
        self.rng = random.Random(seed)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.negative_slope = negative_slope
        self.layer1 = GATLayer(in_dim, hidden_dim, self.rng,
                              negative_slope=negative_slope)
        self.layer2 = GATLayer(hidden_dim, hidden_dim, self.rng,
                               negative_slope=negative_slope)
        scale = math.sqrt(2.0 / (hidden_dim + 1))
        self.w_out = [self.rng.uniform(-scale, scale) for _ in range(hidden_dim)]
        self.b_out = 0.0
        self.gw_out = [0.0] * hidden_dim
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

    def attention_weights(self, graph, v: int) -> list[dict[int, float]]:
        """Per-layer attention distributions over v's full neighborhood.

        Returns [layer1_alphas, layer2_alphas]; each maps node -> α,
        summing to 1. A readable explanation of *which counterparties*
        drove v's fraud score. No sampling: deterministic.
        """
        neighs = graph.neighbors(v)
        h_x = {u: graph.features[u] for u in ({v} | set(neighs))}
        _, c1 = self.layer1.forward(v, h_x, neighs)
        h1 = {}
        for x in ({v} | set(neighs)):
            hx = {u: graph.features[u] for u in ({x} | set(graph.neighbors(x)))}
            h1[x], _ = self.layer1.forward(x, hx, graph.neighbors(x))
        h_v = {u: h1[u] for u in ({v} | set(neighs))}
        _, c2 = self.layer2.forward(v, h_v, neighs)
        return [dict(c1["alphas"]), dict(c2["alphas"])]

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
        self.gw_out = [0.0] * self.hidden_dim
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
            for j in range(self.hidden_dim):
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
        for j in range(self.hidden_dim):
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

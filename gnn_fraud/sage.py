"""GraphSAGE with mean aggregation, 2 layers, from scratch in pure Python.

Layer update for node ``v``::

    h_v = relu(W_self @ x_v + W_agg @ mean({x_u : u in sampled_neighbors(v)}) + b)

A final linear layer + sigmoid maps the layer-2 embedding to a fraud
probability, trained with binary cross-entropy and full backprop through
both message-passing layers. Mini-batches use neighbor sampling: each
target node samples up to ``k`` neighbors per layer, so training cost is
bounded by ``O(batch * k^2)`` instead of the full 2-hop neighborhood.

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


def _outer(a: list[float], b: list[float]) -> list[list[float]]:
    return [[ai * bj for bj in b] for ai in a]


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


def _mean_vec(vecs: list[list[float]], dim: int) -> list[float]:
    if not vecs:
        return [0.0] * dim
    m = len(vecs)
    return [sum(col) / m for col in zip(*vecs)]


def _bce(p: float, y: int) -> float:
    p = min(max(p, 1e-7), 1.0 - 1e-7)
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


# ---------------------------------------------------------------------------
# one message-passing layer
# ---------------------------------------------------------------------------


class SAGELayer:
    """Single GraphSAGE mean-aggregation layer with ReLU."""

    def __init__(self, in_dim: int, out_dim: int, rng: random.Random):
        scale = math.sqrt(2.0 / (in_dim + out_dim))  # Xavier-ish init
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W_self = [
            [rng.uniform(-scale, scale) for _ in range(in_dim)]
            for _ in range(out_dim)
        ]
        self.W_agg = [
            [rng.uniform(-scale, scale) for _ in range(in_dim)]
            for _ in range(out_dim)
        ]
        self.b = [0.0] * out_dim
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW_self = [[0.0] * self.in_dim for _ in range(self.out_dim)]
        self.gW_agg = [[0.0] * self.in_dim for _ in range(self.out_dim)]
        self.gb = [0.0] * self.out_dim

    def forward(self, h_self: list[float], h_neighs: list[list[float]]):
        """Returns (output, cache) where cache holds everything backprop needs."""
        agg = _mean_vec(h_neighs, self.in_dim)
        pre = _vec_add(
            _vec_add(_matvec(self.W_self, h_self), _matvec(self.W_agg, agg)),
            self.b,
        )
        out = _relu(pre)
        return out, (pre, h_self, agg)

    def backward(self, d_out: list[float], cache) -> tuple[list[float], list[float]]:
        """Backprop one step.

        Returns (d_h_self, d_agg) -- gradients w.r.t. the node's own input
        embedding and w.r.t. the *mean* neighbor embedding. The caller splits
        d_agg across the sampled neighbors.
        """
        pre, h_self, agg = cache
        d_pre = [d * (1.0 if p > 0.0 else 0.0) for d, p in zip(d_out, pre)]
        for i in range(self.out_dim):
            di = d_pre[i]
            if di == 0.0:
                continue
            row_s = self.gW_self[i]
            row_a = self.gW_agg[i]
            for j in range(self.in_dim):
                row_s[j] += di * h_self[j]
                row_a[j] += di * agg[j]
            self.gb[i] += di
        d_h_self = _mat_t_vec(self.W_self, d_pre)
        d_agg = _mat_t_vec(self.W_agg, d_pre)
        return d_h_self, d_agg

    def step(self, lr: float) -> None:
        for i in range(self.out_dim):
            for j in range(self.in_dim):
                self.W_self[i][j] -= lr * self.gW_self[i][j]
                self.W_agg[i][j] -= lr * self.gW_agg[i][j]
            self.b[i] -= lr * self.gb[i]

    def state_dict(self):
        return copy.deepcopy(
            {"W_self": self.W_self, "W_agg": self.W_agg, "b": self.b}
        )

    def load_state_dict(self, state) -> None:
        self.W_self = copy.deepcopy(state["W_self"])
        self.W_agg = copy.deepcopy(state["W_agg"])
        self.b = copy.deepcopy(state["b"])


# ---------------------------------------------------------------------------
# two-layer model with neighbor sampling
# ---------------------------------------------------------------------------


class GraphSAGE:
    """Two-layer GraphSAGE for binary node classification."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 8,
        lr: float = 0.1,
        seed: int = 0,
    ):
        self.rng = random.Random(seed)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.layer1 = SAGELayer(in_dim, hidden_dim, self.rng)
        self.layer2 = SAGELayer(hidden_dim, hidden_dim, self.rng)
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
            neighs = [graph.features[n] for n in s1[x]]
            h1[x], cache1[x] = self.layer1.forward(graph.features[x], neighs)

        h2: dict[int, list[float]] = {}
        cache2: dict[int, tuple] = {}
        for v in targets:
            neighs = [h1[u] for u in s2[v]]
            h2[v], cache2[v] = self.layer2.forward(h1[v], neighs)

        probs: dict[int, float] = {}
        logits: dict[int, float] = {}
        for v in targets:
            z = sum(w * h for w, h in zip(self.w_out, h2[v])) + self.b_out
            logits[v] = z
            probs[v] = _sigmoid(z)

        cache = {"s2": s2, "s1": s1, "h1": h1, "cache1": cache1,
                 "h2": h2, "cache2": cache2, "targets": list(targets)}
        return probs, cache

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        """Inference with full (unsampled) neighborhoods -- deterministic."""
        probs, _ = self.forward(graph, targets, k1=None, k2=None)
        return probs

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
            d_h_self, d_agg = self.layer2.backward(dh2[v], cache2[v])
            dh1[v] = _vec_add(dh1[v], d_h_self)
            m = len(s2[v])
            if m:
                share = _scale(d_agg, 1.0 / m)
                for u in s2[v]:
                    dh1[u] = _vec_add(dh1[u], share)

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

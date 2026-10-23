"""GraphSAGE with mean or max-pool aggregation, 2 layers, from scratch.

Layer update for node ``v`` (mean mode)::

    h_v = relu(W_self @ x_v + W_agg @ mean({x_u : u in sampled_neighbors(v)}) + b)

Max-pool mode (the GraphSAGE paper's pooling aggregator) transforms each
neighbor first and keeps the loudest signal per dimension::

    h_v = relu(W_self @ x_v + W_agg @ max_u relu(W_pool @ x_u + b_pool) + b)

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
    """Single GraphSAGE layer with ReLU: ``aggregator`` is ``"mean"`` or
    ``"pool"`` (the GraphSAGE paper's max-pooling aggregator)."""

    def __init__(self, in_dim: int, out_dim: int, rng: random.Random,
                 aggregator: str = "mean"):
        if aggregator not in ("mean", "pool"):
            raise ValueError(
                f"aggregator must be 'mean' or 'pool', got {aggregator!r}")
        scale = math.sqrt(2.0 / (in_dim + out_dim))  # Xavier-ish init
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggregator = aggregator
        self.W_self = [
            [rng.uniform(-scale, scale) for _ in range(in_dim)]
            for _ in range(out_dim)
        ]
        self.W_agg = [
            [rng.uniform(-scale, scale) for _ in range(in_dim)]
            for _ in range(out_dim)
        ]
        # Max-pool mode (the GraphSAGE paper's pooling aggregator): each
        # neighbor is transformed through relu(W_pool @ x_u + b_pool)
        # before the elementwise max, so the layer can keep a single loud
        # tell instead of the neighborhood mean. He init on W_pool: it
        # feeds a ReLU, and a symmetric Xavier init leaves the transform
        # half-dead from the first forward pass.
        pool_scale = math.sqrt(2.0 / in_dim)
        self.W_pool = (
            [[rng.uniform(-pool_scale, pool_scale) for _ in range(in_dim)]
             for _ in range(in_dim)]
            if aggregator == "pool" else None
        )
        self.b_pool = [0.0] * in_dim if aggregator == "pool" else None
        self.b = [0.0] * out_dim
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW_self = [[0.0] * self.in_dim for _ in range(self.out_dim)]
        self.gW_agg = [[0.0] * self.in_dim for _ in range(self.out_dim)]
        self.gW_pool = (
            [[0.0] * self.in_dim for _ in range(self.in_dim)]
            if self.aggregator == "pool" else None
        )
        self.gb_pool = (
            [0.0] * self.in_dim if self.aggregator == "pool" else None
        )
        self.gb = [0.0] * self.out_dim

    def _pre(self, h_self: list[float], agg: list[float]) -> list[float]:
        return _vec_add(
            _vec_add(_matvec(self.W_self, h_self), _matvec(self.W_agg, agg)),
            self.b,
        )

    def _pool_agg(self, h_neighs: list[list[float]]
                  ) -> tuple[list[float], list[list[float]], list[int]]:
        """Max-pool aggregation: relu(W_pool @ x_u + b_pool) per neighbor,
        then an elementwise max. Returns (agg, transformed, argmax)."""
        t = [_relu(_vec_add(_matvec(self.W_pool, x), self.b_pool))
             for x in h_neighs]
        if not t:
            return [0.0] * self.in_dim, [], [-1] * self.in_dim
        agg, argmax = [], []
        for i in range(self.in_dim):
            best, bi = t[0][i], 0
            for j in range(1, len(t)):
                if t[j][i] > best:
                    best, bi = t[j][i], j
            agg.append(best)
            argmax.append(bi)
        return agg, t, argmax

    def forward(self, h_self: list[float], h_neighs: list[list[float]]):
        """Returns (output, cache) where cache holds everything backprop needs."""
        if self.aggregator == "pool":
            agg, t, argmax = self._pool_agg(h_neighs)
            cache = (self._pre(h_self, agg), h_self, agg, t, argmax, h_neighs)
        else:
            agg = _mean_vec(h_neighs, self.in_dim)
            cache = (self._pre(h_self, agg), h_self, agg, len(h_neighs))
        return _relu(cache[0]), cache

    def backward(self, d_out: list[float], cache
                 ) -> tuple[list[float], list[list[float]]]:
        """Backprop one step.

        Returns (d_h_self, d_neighs) -- the gradient w.r.t. the node's own
        input embedding, and per-neighbor gradients aligned with the
        forward pass's neighbor list. Mean mode splits the aggregation
        gradient evenly (bit-for-bit what the old (d_h_self, d_agg)
        contract produced); pool mode routes each dimension's gradient to
        its argmax neighbor, skipping dead ReLUs. Same contract as
        :meth:`TemporalSAGELayer.backward_temporal`.
        """
        pre, h_self, agg = cache[0], cache[1], cache[2]
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
        if self.aggregator == "pool":
            t, argmax, h_neighs = cache[3], cache[4], cache[5]
            d_neighs = [[0.0] * self.in_dim for _ in h_neighs]
            for i in range(self.in_dim):
                bi = argmax[i]
                if bi < 0 or t[bi][i] <= 0.0:
                    continue  # no winner, or a dead ReLU: no gradient flows
                g = d_agg[i]
                row_p = self.gW_pool[i]
                x = h_neighs[bi]
                self.gb_pool[i] += g
                for j in range(self.in_dim):
                    row_p[j] += g * x[j]
                    d_neighs[bi][j] += g * self.W_pool[i][j]
        else:
            m = cache[3]
            if m:
                share = _scale(d_agg, 1.0 / m)
                d_neighs = [list(share) for _ in range(m)]
            else:
                d_neighs = []
        return d_h_self, d_neighs

    def step(self, lr: float) -> None:
        for i in range(self.out_dim):
            for j in range(self.in_dim):
                self.W_self[i][j] -= lr * self.gW_self[i][j]
                self.W_agg[i][j] -= lr * self.gW_agg[i][j]
            self.b[i] -= lr * self.gb[i]
        if self.aggregator == "pool":
            for i in range(self.in_dim):
                for j in range(self.in_dim):
                    self.W_pool[i][j] -= lr * self.gW_pool[i][j]
                self.b_pool[i] -= lr * self.gb_pool[i]

    def state_dict(self):
        state = {
            "W_self": self.W_self,
            "W_agg": self.W_agg,
            "b": self.b,
            "aggregator": self.aggregator,
        }
        if self.aggregator == "pool":
            state["W_pool"] = self.W_pool
            state["b_pool"] = self.b_pool
        return copy.deepcopy(state)

    def load_state_dict(self, state) -> None:
        # The checkpoint defines the model: a mean checkpoint (old or new
        # format) restores a mean layer, a pool checkpoint a pool layer.
        self.aggregator = state.get("aggregator", "mean")
        self.W_self = copy.deepcopy(state["W_self"])
        self.W_agg = copy.deepcopy(state["W_agg"])
        self.b = copy.deepcopy(state["b"])
        if self.aggregator == "pool":
            self.W_pool = copy.deepcopy(state["W_pool"])
            self.b_pool = copy.deepcopy(state["b_pool"])
        else:
            self.W_pool = None
            self.b_pool = None
        self.zero_grad()


# ---------------------------------------------------------------------------
# two-layer model with neighbor sampling
# ---------------------------------------------------------------------------


class GraphSAGE:
    """Two-layer GraphSAGE for binary node classification.

    Args:
        aggregator: ``"mean"`` or ``"pool"`` (max-pooling aggregator from
            the GraphSAGE paper), shared by both layers.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 8,
        lr: float = 0.1,
        seed: int = 0,
        aggregator: str = "mean",
    ):
        self.rng = random.Random(seed)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.aggregator = aggregator
        self.layer1 = SAGELayer(in_dim, hidden_dim, self.rng,
                                aggregator=aggregator)
        self.layer2 = SAGELayer(hidden_dim, hidden_dim, self.rng,
                                aggregator=aggregator)
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
            d_h_self, d_neighs = self.layer2.backward(dh2[v], cache2[v])
            dh1[v] = _vec_add(dh1[v], d_h_self)
            for u, d_h_u in zip(s2[v], d_neighs):
                dh1[u] = _vec_add(dh1[u], d_h_u)

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

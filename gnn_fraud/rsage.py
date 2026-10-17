"""Relational GraphSAGE (R-GCN-lite) for the heterogeneous fraud graph.

Layer update for node ``v`` of type ``t``::

    h_v = relu(W_self[t] @ x_v + sum_r W_rel[r] @ mean({x_u : u in N_r(v)}) + b)

Each relation gets its *own* message matrix (an account→device edge means
something different from an account→account edge), and each node type gets
its own self matrix, so types with different feature dimensions project
into one shared hidden space. Two layers, sigmoid output over accounts
only, full backprop through both layers. Pure stdlib, like the rest of
this repo.
"""

from __future__ import annotations

import copy
import math
import random

from gnn_fraud.hetero import (
    ACCOUNT,
    DEVICE,
    MERCHANT,
    RELATION_NBR_TYPE,
    RELATIONS,
)

# ---------------------------------------------------------------------------
# tiny linalg helpers (lists of floats)
# ---------------------------------------------------------------------------


def _matvec(W: list[list[float]], x: list[float]) -> list[float]:
    return [sum(w * xi for w, xi in zip(row, x)) for row in W]


def _mat_t_vec(W: list[list[float]], v: list[float]) -> list[float]:
    n_cols = len(W[0])
    return [sum(W[r][c] * v[r] for r in range(len(W))) for c in range(n_cols)]


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


def _init_matrix(out_dim: int, in_dim: int,
                 rng: random.Random) -> list[list[float]]:
    scale = math.sqrt(2.0 / (in_dim + out_dim))  # Xavier-ish init
    return [
        [rng.uniform(-scale, scale) for _ in range(in_dim)]
        for _ in range(out_dim)
    ]


def _zero_matrix(out_dim: int, in_dim: int) -> list[list[float]]:
    return [[0.0] * in_dim for _ in range(out_dim)]


# ---------------------------------------------------------------------------
# one relational message-passing layer
# ---------------------------------------------------------------------------


class RelSAGELayer:
    """Single R-GCN-lite layer: per-type self matrices + per-relation
    message matrices over mean-aggregated neighbors."""

    def __init__(self, in_dims: dict[int, int], out_dim: int,
                 rng: random.Random):
        self.in_dims = dict(in_dims)
        self.out_dim = out_dim
        self.W_self = {
            t: _init_matrix(out_dim, in_dim, rng)
            for t, in_dim in in_dims.items()
        }
        self.W_rel = {
            r: _init_matrix(out_dim, in_dims[RELATION_NBR_TYPE[r]], rng)
            for r in RELATIONS
        }
        self.b = [0.0] * out_dim
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW_self = {
            t: _zero_matrix(self.out_dim, in_dim)
            for t, in_dim in self.in_dims.items()
        }
        self.gW_rel = {
            r: _zero_matrix(self.out_dim, self.in_dims[RELATION_NBR_TYPE[r]])
            for r in RELATIONS
        }
        self.gb = [0.0] * self.out_dim

    def forward(self, node_type: int, h_self: list[float],
                h_by_rel: dict[int, list[list[float]]]):
        """Returns (output, cache).

        Args:
            node_type: type of the center node.
            h_self: the center node's input embedding.
            h_by_rel: per-relation lists of neighbor input embeddings.
        """
        agg = {
            r: _mean_vec(h_by_rel.get(r, []), self.in_dims[RELATION_NBR_TYPE[r]])
            for r in RELATIONS
        }
        pre = _matvec(self.W_self[node_type], h_self)
        for r in RELATIONS:
            pre = _vec_add(pre, _matvec(self.W_rel[r], agg[r]))
        pre = _vec_add(pre, self.b)
        out = _relu(pre)
        return out, (node_type, pre, h_self, agg)

    def backward(self, d_out: list[float], cache):
        """Backprop one step.

        Returns (d_h_self, d_agg_by_rel): gradient w.r.t. the node's own
        input embedding, and per-relation gradients w.r.t. the *mean*
        neighbor embedding. The caller splits each d_agg across that
        relation's sampled neighbors.
        """
        node_type, pre, h_self, agg = cache
        d_pre = [d * (1.0 if p > 0.0 else 0.0) for d, p in zip(d_out, pre)]

        # Parameter grads: self path.
        g_self = self.gW_self[node_type]
        in_dim = self.in_dims[node_type]
        for i in range(self.out_dim):
            di = d_pre[i]
            if di == 0.0:
                continue
            row = g_self[i]
            for j in range(in_dim):
                row[j] += di * h_self[j]
            self.gb[i] += di
        d_h_self = _mat_t_vec(self.W_self[node_type], d_pre)

        # Parameter grads: per-relation paths.
        d_agg_by_rel: dict[int, list[float]] = {}
        for r in RELATIONS:
            nbr_dim = self.in_dims[RELATION_NBR_TYPE[r]]
            g_rel = self.gW_rel[r]
            agg_r = agg[r]
            for i in range(self.out_dim):
                di = d_pre[i]
                if di == 0.0:
                    continue
                row = g_rel[i]
                for j in range(nbr_dim):
                    row[j] += di * agg_r[j]
            d_agg_by_rel[r] = _mat_t_vec(self.W_rel[r], d_pre)

        return d_h_self, d_agg_by_rel

    def step(self, lr: float) -> None:
        for t, W in self.W_self.items():
            gW = self.gW_self[t]
            for i in range(self.out_dim):
                for j in range(self.in_dims[t]):
                    W[i][j] -= lr * gW[i][j]
        for r, W in self.W_rel.items():
            gW = self.gW_rel[r]
            nbr_dim = self.in_dims[RELATION_NBR_TYPE[r]]
            for i in range(self.out_dim):
                for j in range(nbr_dim):
                    W[i][j] -= lr * gW[i][j]
        for i in range(self.out_dim):
            self.b[i] -= lr * self.gb[i]

    def state_dict(self):
        return {
            "W_self": copy.deepcopy(self.W_self),
            "W_rel": copy.deepcopy(self.W_rel),
            "b": copy.deepcopy(self.b),
        }

    def load_state_dict(self, state) -> None:
        self.W_self = copy.deepcopy(state["W_self"])
        self.W_rel = copy.deepcopy(state["W_rel"])
        self.b = copy.deepcopy(state["b"])


# ---------------------------------------------------------------------------
# two-layer relational model over the heterogeneous graph
# ---------------------------------------------------------------------------


class RelationalGraphSAGE:
    """Two-layer R-GCN-lite for binary classification of account nodes.

    Devices and merchants participate in message passing (their embeddings
    carry the shared-device / shared-merchant signal to their accounts)
    but are never scored: labels exist only for accounts.
    """

    def __init__(
        self,
        type_dims: dict[int, int],
        hidden_dim: int = 8,
        lr: float = 0.1,
        seed: int = 0,
    ):
        self.rng = random.Random(seed)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.layer1 = RelSAGELayer(type_dims, hidden_dim, self.rng)
        hidden_dims = {ACCOUNT: hidden_dim, DEVICE: hidden_dim,
                       MERCHANT: hidden_dim}
        self.layer2 = RelSAGELayer(hidden_dims, hidden_dim, self.rng)
        scale = math.sqrt(2.0 / (hidden_dim + 1))
        self.w_out = [self.rng.uniform(-scale, scale) for _ in range(hidden_dim)]
        self.b_out = 0.0
        self.gw_out = [0.0] * hidden_dim
        self.gb_out = 0.0

    # -- sampling ----------------------------------------------------------

    def _sample(self, candidates: list[int], k: int | None) -> list[int]:
        if k is None or len(candidates) <= k:
            return list(candidates)
        return self.rng.sample(candidates, k)

    def _expand_batch(self, graph, targets: list[int],
                      k1: int | None, k2: int | None):
        """Sample the 2-hop typed computation graph for a mini-batch.

        Returns (s2, s1): s2[v][r] = layer-2 sampled neighbors of target v
        under relation r; s1[x][r] likewise for every layer-1 node x.
        """
        s2: dict[int, dict[int, list[int]]] = {}
        layer1_nodes: set[int] = set(targets)
        for v in targets:
            s2[v] = {r: self._sample(graph.rel_neighbors(v, r), k2)
                     for r in RELATIONS}
            for r in RELATIONS:
                layer1_nodes.update(s2[v][r])
        s1: dict[int, dict[int, list[int]]] = {}
        for x in layer1_nodes:
            s1[x] = {r: self._sample(graph.rel_neighbors(x, r), k1)
                     for r in RELATIONS}
        return s2, s1

    # -- forward ------------------------------------------------------------

    def forward(self, graph, targets: list[int],
                k1: int | None = 3, k2: int | None = 3):
        """Forward pass. Returns (probs dict, cache)."""
        s2, s1 = self._expand_batch(graph, targets, k1, k2)

        h1: dict[int, list[float]] = {}
        cache1: dict[int, tuple] = {}
        for x in s1:
            h_by_rel = {
                r: [graph.features[u] for u in s1[x][r]] for r in RELATIONS
            }
            h1[x], cache1[x] = self.layer1.forward(
                graph.node_type[x], graph.features[x], h_by_rel)

        h2: dict[int, list[float]] = {}
        cache2: dict[int, tuple] = {}
        for v in targets:
            h_by_rel = {r: [h1[u] for u in s2[v][r]] for r in RELATIONS}
            h2[v], cache2[v] = self.layer2.forward(
                graph.node_type[v], h1[v], h_by_rel)

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

    # -- training step -------------------------------------------------------

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    k1: int | None = 3, k2: int | None = 3,
                    pos_weight: float = 1.0) -> float:
        """One SGD step on a mini-batch of account nodes."""
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
            d_h_self, d_agg_by_rel = self.layer2.backward(dh2[v], cache2[v])
            dh1[v] = _vec_add(dh1[v], d_h_self)
            for r in RELATIONS:
                m = len(s2[v][r])
                if m:
                    share = _scale(d_agg_by_rel[r], 1.0 / m)
                    for u in s2[v][r]:
                        dh1[u] = _vec_add(dh1[u], share)

        # Backprop through layer 1 (input features are leaves).
        for x in s1:
            self.layer1.backward(dh1[x], cache1[x])

        # SGD update.
        self.layer1.step(self.lr)
        self.layer2.step(self.lr)
        for j in range(self.hidden_dim):
            self.w_out[j] -= self.lr * self.gw_out[j]
        self.b_out -= self.lr * self.gb_out
        return loss

    # -- checkpointing --------------------------------------------------------

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

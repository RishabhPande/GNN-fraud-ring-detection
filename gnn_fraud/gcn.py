"""Graph Convolutional Network (Kipf & Welling, 2017), from scratch.

Layer update (one layer, all nodes at once)::

    H' = relu(D_hat^{-1/2} A_hat D_hat^{-1/2} H W),   A_hat = A + I

Per node ``v`` that is::

    h'_v = relu(sum_{u in N(v) u {v}}  W @ h_u / sqrt(d_hat_v * d_hat_u))

where ``d_hat`` is the degree *including* the self-loop. Two things make
this different from the GraphSAGE layers in :mod:`gnn_fraud.sage`:

* **No self/neighbor split.** SAGE learns separate ``W_self`` and
  ``W_agg`` matrices and concatenates self and neighborhood; GCN folds
  the self-loop into the normalized sum, so one matrix ``W`` does both
  jobs and the self contribution is just another term with weight
  ``1 / d_hat_v``.
* **Degree normalization, not a flat mean.** A hub neighbor's message is
  down-weighted by ``1 / sqrt(d_hat_v * d_hat_u)`` -- a high-degree
  account spreads its signal thin, and a node with many neighbors is
  not swamped by any single one. On fraud graphs that matters: ring
  members are dense (high degree), so a flat mean lets the ring dominate
  its own neighborhood, while GCN's normalization keeps background hubs
  from shouting down the tells.

Unlike SAGE there is no neighbor sampling here: the normalization needs
the true degrees, so training runs full-batch over the whole graph
(``train_batch`` still scores only the target subset, but the forward
pass computes every node's embedding). The graphs in this repo are small
enough that this is cheap, and it keeps the model exactly the paper's
-- no sampling noise in the normalization.

A final linear layer + sigmoid maps the layer-2 embedding to a fraud
probability, trained with binary cross-entropy and full backprop through
both convolutional layers. All linear algebra is hand-rolled
lists-of-floats -- no numpy, no torch.
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


def _bce(p: float, y: int) -> float:
    p = min(max(p, 1e-7), 1.0 - 1e-7)
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


def _norm_weights(graph) -> list[list[tuple[int, float]]]:
    """Symmetric normalization weights: for node ``v``, the list of
    ``(u, c_vu)`` over ``u in N(v) u {v}`` with
    ``c_vu = 1 / sqrt(d_hat_v * d_hat_u)``, ``d_hat = degree + 1``.

    The matrix is symmetric (``c_vu == c_uv``) because the graph is
    undirected and the self-loop is added to both endpoints' degrees --
    which is also why the backward pass can reuse these same weights.
    """
    d_hat = [graph.degree(v) + 1 for v in range(graph.n)]
    weights: list[list[tuple[int, float]]] = []
    for v in range(graph.n):
        row = []
        for u in sorted(set(graph.neighbors(v)) | {v}):
            row.append((u, 1.0 / math.sqrt(d_hat[v] * d_hat[u])))
        weights.append(row)
    return weights


# ---------------------------------------------------------------------------
# one convolutional layer
# ---------------------------------------------------------------------------


class GCNLayer:
    """Single GCN layer with ReLU: ``H' = relu(D_hat^{-1/2} A_hat
    D_hat^{-1/2} H W)``. Operates on the whole graph at once (the
    normalization needs true degrees), so ``forward`` takes the full
    embedding list indexed by node id, not one node's neighborhood."""

    def __init__(self, in_dim: int, out_dim: int, rng: random.Random):
        scale = math.sqrt(2.0 / (in_dim + out_dim))  # Xavier-ish init
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = [
            [rng.uniform(-scale, scale) for _ in range(in_dim)]
            for _ in range(out_dim)
        ]
        self.b = [0.0] * out_dim
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW = [[0.0] * self.in_dim for _ in range(self.out_dim)]
        self.gb = [0.0] * self.out_dim

    def forward(self, graph, H: list[list[float]]):
        """Returns (H_out, cache) where cache holds everything backprop
        needs: (norm weights, H_in, pre-activations)."""
        weights = _norm_weights(graph)
        agg = []
        for v in range(graph.n):
            acc = [0.0] * self.in_dim
            for u, c in weights[v]:
                hu = H[u]
                for j in range(self.in_dim):
                    acc[j] += c * hu[j]
            agg.append(acc)
        pre = [_vec_add(_matvec(self.W, a), self.b) for a in agg]
        return _relu_list(pre), (weights, H, pre, agg)

    def backward(self, d_out: list[list[float]], cache
                 ) -> list[list[float]]:
        """Backprop one layer.

        Returns ``d_H`` -- the gradient w.r.t. the input embeddings,
        indexed by node id. The normalized adjacency is symmetric, so
        the message pass in reverse uses the same weights: each node's
        outgoing gradient is its ``d_pre`` spread to its normalized
        neighbors (including itself) through ``W^T``.
        """
        weights, H, pre, agg = cache
        n = len(weights)
        d_pre = [
            [d * (1.0 if p > 0.0 else 0.0) for d, p in zip(d_out[v], pre[v])]
            for v in range(n)
        ]
        for v in range(n):
            dv = d_pre[v]
            if not any(dv):
                continue
            row = self.gW
            for i in range(self.out_dim):
                di = dv[i]
                if di == 0.0:
                    continue
                for j in range(self.in_dim):
                    row[i][j] += di * agg[v][j]
                self.gb[i] += di
        d_agg: list[list[float]] = []
        for v in range(n):
            d_agg.append(_mat_t_vec(self.W, d_pre[v]))
        d_H = [[0.0] * self.in_dim for _ in range(n)]
        for v in range(n):
            for u, c in weights[v]:
                du = d_agg[v]
                row = d_H[u]
                for j in range(self.in_dim):
                    row[j] += c * du[j]
        return d_H

    def step(self, lr: float) -> None:
        for i in range(self.out_dim):
            for j in range(self.in_dim):
                self.W[i][j] -= lr * self.gW[i][j]
            self.b[i] -= lr * self.gb[i]

    def state_dict(self):
        return {"W": copy.deepcopy(self.W), "b": copy.deepcopy(self.b)}

    def load_state_dict(self, state) -> None:
        self.W = copy.deepcopy(state["W"])
        self.b = copy.deepcopy(state["b"])
        self.zero_grad()


def _relu_list(pre: list[list[float]]) -> list[list[float]]:
    return [_relu(p) for p in pre]


# ---------------------------------------------------------------------------
# two-layer model
# ---------------------------------------------------------------------------


class GraphConvolution:
    """Two-layer GCN for binary node classification.

    Full-graph forward (no sampling -- see the module docstring), linear
    + sigmoid output head, weighted BCE loss like :class:`GraphSAGE`.
    """

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
        self.layer1 = GCNLayer(in_dim, hidden_dim, self.rng)
        self.layer2 = GCNLayer(hidden_dim, hidden_dim, self.rng)
        scale = math.sqrt(2.0 / (hidden_dim + 1))
        self.w_out = [self.rng.uniform(-scale, scale) for _ in range(hidden_dim)]
        self.b_out = 0.0
        self.gw_out = [0.0] * hidden_dim
        self.gb_out = 0.0

    # -- forward ----------------------------------------------------------

    def forward(self, graph, targets: list[int]):
        """Forward pass over the whole graph. Returns (probs dict, cache)
        with probs only for the requested targets."""
        h1, cache1 = self.layer1.forward(graph, graph.features)
        h2, cache2 = self.layer2.forward(graph, h1)

        probs: dict[int, float] = {}
        for v in targets:
            z = sum(w * h for w, h in zip(self.w_out, h2[v])) + self.b_out
            probs[v] = _sigmoid(z)

        cache = {"h1": h1, "cache1": cache1,
                 "h2": h2, "cache2": cache2, "targets": list(targets)}
        return probs, cache

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        """Inference -- deterministic, no sampling anywhere in GCN."""
        probs, _ = self.forward(graph, targets)
        return probs

    # -- training step -----------------------------------------------------

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    pos_weight: float = 1.0) -> float:
        """One full-batch SGD step scored on ``targets``. Returns mean
        weighted BCE loss. The forward pass still embeds every node:
        targets' embeddings depend on their 2-hop neighborhoods."""
        self.layer1.zero_grad()
        self.layer2.zero_grad()
        self.gw_out = [0.0] * self.hidden_dim
        self.gb_out = 0.0

        probs, cache = self.forward(graph, targets)
        h2, cache2 = cache["h2"], cache["cache2"]

        B = len(targets)
        weights = {v: (pos_weight if labels[v] == 1 else 1.0) for v in targets}
        loss = sum(weights[v] * _bce(probs[v], labels[v])
                   for v in targets) / B

        # Output layer grads.
        dh2 = [[0.0] * self.hidden_dim for _ in range(graph.n)]
        for v in targets:
            d_logit = weights[v] * (probs[v] - labels[v]) / B
            for j in range(self.hidden_dim):
                self.gw_out[j] += d_logit * h2[v][j]
            self.gb_out += d_logit
            dh2[v] = _scale(self.w_out, d_logit)

        # Backprop through layer 2, then layer 1 (features are leaves).
        dh1 = self.layer2.backward(dh2, cache2)
        self.layer1.backward(dh1, cache["cache1"])

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

"""Chebyshev spectral convolution (ChebNet, Defferrard et al., 2016),
from scratch.

Layer update (one layer, all nodes at once)::

    H' = relu(sum_{k=0}^{K} T_k(L_tilde) H W_k + b)

where ``T_k`` is the Chebyshev polynomial of the first kind and
``L_tilde`` is the rescaled normalized Laplacian. With the usual
``lambda_max ~= 2`` approximation, the rescaling collapses to::

    L_tilde = 2L / lambda_max - I = -D^{-1/2} A D^{-1/2}

so applying ``L_tilde`` to embeddings is a plain neighbor sum with
symmetric degree weights and a minus sign -- no self-loop, unlike the
GCN in :mod:`gnn_fraud.gcn`, which folds a self-loop into its
normalization. The polynomials are applied by recurrence, never by
forming a matrix power::

    Z_0 = H,  Z_1 = L_tilde H,  Z_k = 2 L_tilde Z_{k-1} - Z_{k-2}

so one Cheb layer with ``K = 2`` already reads a 2-hop neighborhood,
where a GCN layer reads only one hop. ``K = 0`` degrades exactly to a
linear layer over raw features (the feature-only baseline's layer).

Backprop runs the recurrence backwards: the gradient w.r.t. ``Z_k``
flows to ``Z_{k-1}`` through ``2 L_tilde`` (symmetric, so the reverse
pass uses the same weights) and to ``Z_{k-2}`` with a minus sign, and
``Z_1``'s gradient flows to ``Z_0`` through ``L_tilde``. Every
``W_k`` and the bias are numeric-gradient-checked, as is the full
two-layer model end to end.

Like the GCN, there is no neighbor sampling: the degree normalization
needs true degrees, so training runs full-batch over the whole graph.
All linear algebra is hand-rolled lists-of-floats -- no numpy, no
torch.
"""

from __future__ import annotations

import copy
import math
import random


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


def _laplacian_weights(graph) -> list[list[tuple[int, float]]]:
    """Rescaled-Laplacian weights: for node ``v``, the list of ``(u, c)``
    over ``u in N(v)`` with ``c = -1 / sqrt(d_v * d_u)``, i.e.
    ``L_tilde = -D^{-1/2} A D^{-1/2}`` under the ``lambda_max ~= 2``
    approximation. No self-loop term: the ``k = 0`` polynomial term
    (``T_0 = I``) is what carries a node's own features.

    Isolated nodes (degree 0) get an empty row instead of a division by
    zero: ``L_tilde`` maps their embeddings to zero, as it should.
    """
    degrees = [graph.degree(v) for v in range(graph.n)]
    weights: list[list[tuple[int, float]]] = []
    for v in range(graph.n):
        row = []
        if degrees[v] > 0:
            for u in sorted(set(graph.neighbors(v))):
                if degrees[u] > 0:
                    row.append((u, -1.0 / math.sqrt(degrees[v] * degrees[u])))
        weights.append(row)
    return weights


def _apply_operator(weights, X: list[list[float]]) -> list[list[float]]:
    """``Y[v] = sum_{(u, c)} c * X[u]`` for one rescaled-Laplacian step."""
    if not X:
        return []
    dim = len(X[0])
    out = []
    for v in range(len(weights)):
        acc = [0.0] * dim
        for u, c in weights[v]:
            xu = X[u]
            for j in range(dim):
                acc[j] += c * xu[j]
        out.append(acc)
    return out


# ---------------------------------------------------------------------------
# one Chebyshev layer
# ---------------------------------------------------------------------------


class ChebLayer:
    """Single ChebNet layer: ``relu(sum_k T_k(L_tilde) H W_k + b)``.
    Operates on the whole graph at once (the normalization needs true
    degrees), so ``forward`` takes the full embedding list indexed by
    node id, not one node's neighborhood."""

    def __init__(self, in_dim: int, out_dim: int, K: int = 2,
                 rng: random.Random | None = None):
        if K < 0:
            raise ValueError("K must be >= 0")
        rng = rng if rng is not None else random.Random(0)
        scale = math.sqrt(2.0 / (in_dim + out_dim))  # Xavier-ish init
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = K
        self.W = [
            [
                [rng.uniform(-scale, scale) for _ in range(in_dim)]
                for _ in range(out_dim)
            ]
            for _ in range(K + 1)
        ]
        self.b = [0.0] * out_dim
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW = [
            [[0.0] * self.in_dim for _ in range(self.out_dim)]
            for _ in range(self.K + 1)
        ]
        self.gb = [0.0] * self.out_dim

    def forward(self, graph, H: list[list[float]]):
        """Returns (H_out, cache) where cache holds everything backprop
        needs: (laplacian weights, Chebyshev terms ``Z_k``, pre-activations)."""
        weights = _laplacian_weights(graph)
        Z: list[list[list[float]]] = [H]
        if self.K >= 1:
            Z.append(_apply_operator(weights, H))
        for k in range(2, self.K + 1):
            prev = _apply_operator(weights, Z[k - 1])
            Z.append([
                [2.0 * p - q for p, q in zip(prev[v], Z[k - 2][v])]
                for v in range(graph.n)
            ])
        pre = []
        for v in range(graph.n):
            acc = list(self.b)
            for k in range(self.K + 1):
                wk = _matvec(self.W[k], Z[k][v])
                for i in range(self.out_dim):
                    acc[i] += wk[i]
            pre.append(acc)
        return [_relu(p) for p in pre], (weights, Z, pre)

    def backward(self, d_out: list[list[float]], cache
                 ) -> list[list[float]]:
        """Backprop one layer. Returns ``d_H`` -- the gradient w.r.t.
        the input embeddings, indexed by node id.

        The reverse recurrence walks ``k`` from ``K`` down to 1: each
        term's gradient flows to the previous term through ``2
        L_tilde`` and to the term before that with a minus sign, and
        term 1's gradient flows to term 0 through ``L_tilde``. The
        operator is symmetric, so the reverse pass reuses the forward
        weights.
        """
        weights, Z, pre = cache
        n = len(Z[0])
        d_pre = [
            [d * (1.0 if p > 0.0 else 0.0) for d, p in zip(d_out[v], pre[v])]
            for v in range(n)
        ]
        # Gradients w.r.t. each W_k and the bias, plus dZ_k. The bias is
        # shared across all K + 1 terms, so its gradient is summed once
        # over nodes -- not once per term.
        for v in range(n):
            dv = d_pre[v]
            for i in range(self.out_dim):
                self.gb[i] += dv[i]
        dZ: list[list[list[float]]] = []
        for k in range(self.K + 1):
            dZ.append([_mat_t_vec(self.W[k], d_pre[v]) for v in range(n)])
            for v in range(n):
                dv = d_pre[v]
                if not any(dv):
                    continue
                row = self.gW[k]
                for i in range(self.out_dim):
                    di = dv[i]
                    if di == 0.0:
                        continue
                    for j in range(self.in_dim):
                        row[i][j] += di * Z[k][v][j]
        # Reverse the Chebyshev recurrence to get dH = adj[0].
        adj = [[list(row) for row in dZk] for dZk in dZ]
        for k in range(self.K, 1, -1):
            stepped = _apply_operator(weights, adj[k])
            for v in range(n):
                for j in range(self.in_dim):
                    adj[k - 1][v][j] += 2.0 * stepped[v][j]
                    adj[k - 2][v][j] -= adj[k][v][j]
        if self.K >= 1:
            stepped = _apply_operator(weights, adj[1])
            for v in range(n):
                for j in range(self.in_dim):
                    adj[0][v][j] += stepped[v][j]
        return adj[0]

    def step(self, lr: float) -> None:
        for k in range(self.K + 1):
            for i in range(self.out_dim):
                for j in range(self.in_dim):
                    self.W[k][i][j] -= lr * self.gW[k][i][j]
            # bias stepped once below, not per k
        for i in range(self.out_dim):
            self.b[i] -= lr * self.gb[i]

    def state_dict(self):
        return {"W": copy.deepcopy(self.W), "b": copy.deepcopy(self.b),
                "K": self.K}

    def load_state_dict(self, state) -> None:
        self.W = copy.deepcopy(state["W"])
        self.b = copy.deepcopy(state["b"])
        self.K = state["K"]
        self.zero_grad()


# ---------------------------------------------------------------------------
# two-layer model
# ---------------------------------------------------------------------------


class GraphChebyshev:
    """Two-layer ChebNet for binary node classification.

    Full-graph forward (no sampling -- see the module docstring), linear
    + sigmoid output head, weighted BCE loss like :class:`GraphSAGE`.
    Each layer uses ``K = 2`` by default, so the model reads up to 4
    hops out -- wider than the 2-hop GCN it is compared against, which
    is part of what the example measures.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 8,
        K: int = 2,
        lr: float = 0.1,
        seed: int = 0,
    ):
        self.rng = random.Random(seed)
        self.hidden_dim = hidden_dim
        self.K = K
        self.lr = lr
        self.layer1 = ChebLayer(in_dim, hidden_dim, K=K, rng=self.rng)
        self.layer2 = ChebLayer(hidden_dim, hidden_dim, K=K, rng=self.rng)
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
        """Inference -- deterministic, no sampling anywhere in ChebNet."""
        probs, _ = self.forward(graph, targets)
        return probs

    # -- training step -----------------------------------------------------

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    pos_weight: float = 1.0) -> float:
        """One full-batch SGD step scored on ``targets``. Returns mean
        weighted BCE loss. The forward pass still embeds every node:
        targets' embeddings depend on their neighborhoods."""
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

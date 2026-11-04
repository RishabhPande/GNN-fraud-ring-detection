"""Topology Adaptive Graph Convolutional Network (TAGCN, Du et al.,
2018), from scratch.

A GCN layer reads exactly one normalized hop per layer: two layers see
two hops, no more, no less, and every hop shares one weight matrix.
TAGCN instead lets a single layer mix several hop sizes at once, each
with its own learned weight matrix. With ``A_hat`` the GCN's symmetric
normalization (``D_hat^{-1/2} A_hat D_hat^{-1/2}``, self-loops
included) and ``K`` the largest hop, one layer is::

    H' = relu(sum_{k=0..K} A_hat^k H W_k + b)

So ``k = 0`` keeps the node's own features, ``k = 1`` reads the
1-hop neighborhood (the GCN's whole layer), and ``k = 2`` reads the
2-hop neighborhood, all in one layer, with a separate ``W_k`` per hop
so the model can weight its own features, its pocket, and its wider
neighborhood differently instead of fixing those weights by
construction. That is the difference from ChebNet in this repo: ChebNet
mixes Chebyshev polynomials of the Laplacian, TAGCN mixes plain powers
of the normalized adjacency, no spectral approximation involved.

Two TAG layers with ``K = 2`` therefore read up to 4 hops, the same
receptive field as the ChebNet's two ``K = 2`` layers, but through a
simpler mechanism: repeated multiplication by ``A_hat``. Backprop
reverses each power the same way, since ``A_hat`` is symmetric: the
gradient through ``A_hat^k`` is ``k`` more multiplications by
``A_hat``. Both the layer and the full two-layer model are
numeric-gradient-checked in ``tests/test_tag.py``.

Like the GCN, there is no neighbor sampling: the normalization needs
true degrees, so training runs full-batch over the whole graph. All
linear algebra is hand-rolled lists-of-floats -- no numpy, no torch.
"""

from __future__ import annotations

import copy
import math
import random

from gnn_fraud.gcn import _bce, _norm_weights, _relu, _sigmoid


def _apply_norm(weights, H):
    """One multiplication by the symmetric normalized adjacency:
    ``(A_hat H)[v] = sum_{u, c in weights[v]} c * H[u]``."""
    out = []
    for v in range(len(weights)):
        acc = [0.0] * len(H[0])
        for u, c in weights[v]:
            hu = H[u]
            for j in range(len(hu)):
                acc[j] += c * hu[j]
        out.append(acc)
    return out


# ---------------------------------------------------------------------------
# one TAG layer
# ---------------------------------------------------------------------------


class TAGLayer:
    """Single TAG layer: ``H' = relu(sum_{k=0..K} A_hat^k H W_k + b)``.

    ``W`` is a list of ``K + 1`` matrices, each ``out_dim x in_dim``,
    indexed by hop. ``K = 0`` degrades exactly to a linear layer over
    raw features (plus ReLU), and ``K = 1`` with ``W_0`` and ``W_1``
    tied would be a GCN layer with a separate self weight.
    """

    def __init__(self, in_dim: int, out_dim: int, K: int,
                 rng: random.Random):
        scale = math.sqrt(2.0 / (in_dim + out_dim))  # Xavier-ish init
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = K
        self.W = [
            [[rng.uniform(-scale, scale) for _ in range(in_dim)]
             for _ in range(out_dim)]
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
        """Returns (H_out, cache). Cache holds everything backprop
        needs: (norm weights, hop powers, pre-activations)."""
        weights = _norm_weights(graph)
        powers = [H]
        for _ in range(self.K):
            powers.append(_apply_norm(weights, powers[-1]))
        pre = []
        for v in range(graph.n):
            row = list(self.b)
            for k in range(self.K + 1):
                Wk = self.W[k]
                pk = powers[k][v]
                for i in range(self.out_dim):
                    wi = Wk[i]
                    s = 0.0
                    for j in range(self.in_dim):
                        s += wi[j] * pk[j]
                    row[i] += s
            pre.append(row)
        return [_relu(row) for row in pre], \
            (weights, powers, pre)

    def backward(self, d_out: list[list[float]], cache
                 ) -> list[list[float]]:
        """Backprop one layer. Returns ``d_H``, the gradient w.r.t.
        the input embeddings, indexed by node id.

        The gradient through ``A_hat^k`` is ``k`` multiplications by
        the same symmetric matrix, so each hop's contribution is the
        ``W_k^T``-projected gradient propagated back ``k`` times.
        """
        weights, powers, pre = cache
        n = len(weights)
        d_pre = [
            [d * (1.0 if p > 0.0 else 0.0) for d, p in zip(d_out[v], pre[v])]
            for v in range(n)
        ]
        for v in range(n):
            dv = d_pre[v]
            if not any(dv):
                continue
            for k in range(self.K + 1):
                gWk = self.gW[k]
                pk = powers[k][v]
                for i in range(self.out_dim):
                    di = dv[i]
                    if di == 0.0:
                        continue
                    for j in range(self.in_dim):
                        gWk[i][j] += di * pk[j]
            for i in range(self.out_dim):
                self.gb[i] += dv[i]
        d_H = [[0.0] * self.in_dim for _ in range(n)]
        for k in range(self.K + 1):
            Wk = self.W[k]
            # G[v] = W_k^T @ d_pre[v].
            cur = []
            for v in range(n):
                dv = d_pre[v]
                row = [0.0] * self.in_dim
                for i in range(self.out_dim):
                    di = dv[i]
                    if di == 0.0:
                        continue
                    wi = Wk[i]
                    for j in range(self.in_dim):
                        row[j] += wi[j] * di
                cur.append(row)
            # Propagate back through A_hat^k (symmetric, so the same
            # multiplication as forward).
            for _ in range(k):
                cur = _apply_norm(weights, cur)
            for v in range(n):
                row = d_H[v]
                add = cur[v]
                for j in range(self.in_dim):
                    row[j] += add[j]
        return d_H

    def step(self, lr: float) -> None:
        for k in range(self.K + 1):
            Wk = self.W[k]
            gWk = self.gW[k]
            for i in range(self.out_dim):
                for j in range(self.in_dim):
                    Wk[i][j] -= lr * gWk[i][j]
                self.b[i] -= lr * self.gb[i]

    def state_dict(self):
        return {"W": copy.deepcopy(self.W), "b": copy.deepcopy(self.b),
                "K": self.K}

    def load_state_dict(self, state) -> None:
        self.W = copy.deepcopy(state["W"])
        self.b = copy.deepcopy(state["b"])
        self.zero_grad()


# ---------------------------------------------------------------------------
# two-layer model
# ---------------------------------------------------------------------------


class GraphTAG:
    """Two-layer TAGCN for binary node classification.

    Layer 1 maps features to hidden dim, layer 2 stays in hidden dim,
    and a linear + sigmoid head scores each node. With ``K = 2`` each
    layer reads 0, 1, and 2 hops, so the stack reads up to 4 hops.
    Full-graph forward (no sampling -- see the module docstring),
    weighted BCE loss like :class:`gnn_fraud.gcn.GraphConvolution`.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 16,
        K: int = 2,
        lr: float = 0.5,
        seed: int = 0,
    ):
        self.rng = random.Random(seed)
        self.hidden_dim = hidden_dim
        self.K = K
        self.lr = lr
        self.layer1 = TAGLayer(in_dim, hidden_dim, K, self.rng)
        self.layer2 = TAGLayer(hidden_dim, hidden_dim, K, self.rng)
        scale = math.sqrt(2.0 / (hidden_dim + 1))
        self.w_out = [self.rng.uniform(-scale, scale)
                      for _ in range(hidden_dim)]
        self.b_out = 0.0
        self.gw_out = [0.0] * hidden_dim
        self.gb_out = 0.0

    # -- forward ----------------------------------------------------------

    def forward(self, graph, targets: list[int]):
        """Forward pass over the whole graph. Returns (probs dict,
        cache) with probs only for the requested targets."""
        h1, cache1 = self.layer1.forward(graph, graph.features)
        h2, cache2 = self.layer2.forward(graph, h1)

        probs: dict[int, float] = {}
        for v in targets:
            z = sum(w * x for w, x in zip(self.w_out, h2[v])) + self.b_out
            probs[v] = _sigmoid(z)

        cache = {"h1": h1, "cache1": cache1,
                 "h2": h2, "cache2": cache2, "targets": list(targets)}
        return probs, cache

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        """Inference -- deterministic, no sampling anywhere in TAGCN."""
        probs, _ = self.forward(graph, targets)
        return probs

    # -- training step -----------------------------------------------------

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    pos_weight: float = 1.0) -> float:
        """One full-batch SGD step scored on ``targets``. Returns mean
        weighted BCE loss. The forward pass still embeds every node:
        targets' embeddings depend on their (2 * K)-hop
        neighborhoods."""
        self.layer1.zero_grad()
        self.layer2.zero_grad()
        self.gw_out = [0.0] * self.hidden_dim
        self.gb_out = 0.0

        probs, cache = self.forward(graph, targets)
        h2, cache2 = cache["h2"], cache["cache2"]

        B = len(targets)
        weights = {v: (pos_weight if labels[v] == 1 else 1.0)
                   for v in targets}
        loss = sum(weights[v] * _bce(probs[v], labels[v])
                   for v in targets) / B

        # Output layer grads.
        dh2 = [[0.0] * self.hidden_dim for _ in range(graph.n)]
        for v in targets:
            d_logit = weights[v] * (probs[v] - labels[v]) / B
            for j in range(self.hidden_dim):
                self.gw_out[j] += d_logit * h2[v][j]
            self.gb_out += d_logit
            dh2[v] = [w * d_logit for w in self.w_out]

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

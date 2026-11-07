"""MixHop: higher-order graph convolution via adjacency powers
(Abu-El-Haija et al., 2019), from scratch.

TAGCN in this repo also mixes adjacency powers, but it *sums* the
per-hop transforms into one vector: ``relu(sum_k A_hat^k H W_k + b)``.
MixHop instead *concatenates* them::

    H' = ||_{k=0..K} relu(A^k H W_k + b_k)

where ``||`` is concatenation along the feature axis. The two
differences from TAGCN are deliberate and both matter for what the
model can say:

* **Concatenation keeps hop identity.** After a sum, the head cannot
  tell whether a signal came from a node's own features, its pocket,
  or its wider neighborhood, because the three are folded into one
  vector before the nonlinearity. After a concatenation, the 0-hop
  block, the 1-hop block, and the 2-hop block stay separate columns,
  so the next layer (and the head) can weight the pocket differently
  from the node's own tell. On a fraud graph that is the difference
  between "this account looks odd" and "this account looks odd *and*
  sits two hops from a ring hub".
* **Row-normalized adjacency, not symmetric.** ``A`` here is
  ``(A + I)`` row-normalized by degree, so each row is a plain average
  over a node and its neighbors. The paper's point is that different
  powers of a row-stochastic matrix carry different neighborhood
  scales without the hub down-weighting that the GCN's symmetric
  normalization imposes (see :mod:`gnn_fraud.gcn`). The price is that
  ``A`` is no longer symmetric, so backprop through ``A^k`` must use
  the transpose ``(A^T)^k``, built explicitly in
  :func:`_row_norm_weights_T`.

Two MixHop layers with ``K = 2`` read up to 4 hops, the same receptive
field as the TAGCN and ChebNet stacks, but through concatenation
rather than summation or a polynomial. Like the GCN family there is
no neighbor sampling: the normalization needs true degrees, so
training runs full-batch over the whole graph. All linear algebra is
hand-rolled lists-of-floats, no numpy, no torch.
"""

from __future__ import annotations

import copy
import math
import random

from gnn_fraud.gcn import _bce, _relu, _sigmoid


def _row_norm_weights(graph) -> list[list[tuple[int, float]]]:
    """Row-normalized adjacency with self-loops: for node ``v``, the
    list of ``(u, c_vu)`` over ``u in N(v) u {v}`` with
    ``c_vu = 1 / d_hat_v``, ``d_hat = degree + 1``.

    Each row sums to 1, so ``A H`` is a plain average over a node and
    its neighbors, and every power ``A^k H`` stays on the same scale
    no matter how dense the neighborhood is.
    """
    d_hat = [graph.degree(v) + 1 for v in range(graph.n)]
    weights: list[list[tuple[int, float]]] = []
    for v in range(graph.n):
        c = 1.0 / d_hat[v]
        row = [(u, c) for u in sorted(set(graph.neighbors(v)) | {v})]
        weights.append(row)
    return weights


def _row_norm_weights_T(graph) -> list[list[tuple[int, float]]]:
    """Transpose of :func:`_row_norm_weights`, as the same sparse-row
    format. ``(A^T G)[v]`` sums ``G[u] / d_hat_u`` over ``u`` in
    ``N(v) u {v}``, because the weight on the ``u`` row is ``1 /
    d_hat_u`` and the graph is undirected.
    """
    d_hat = [graph.degree(v) + 1 for v in range(graph.n)]
    weights: list[list[tuple[int, float]]] = []
    for v in range(graph.n):
        row = [(u, 1.0 / d_hat[u])
               for u in sorted(set(graph.neighbors(v)) | {v})]
        weights.append(row)
    return weights


def _apply(weights, H: list[list[float]]) -> list[list[float]]:
    """One sparse-matrix multiplication: ``(A H)[v] = sum_{u, c in
    weights[v]} c * H[u]``."""
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
# one MixHop layer
# ---------------------------------------------------------------------------


class MixHopLayer:
    """Single MixHop layer: ``H' = ||_{k=0..K} relu(A^k H W_k + b_k)``.

    ``W`` is a list of ``K + 1`` matrices, each ``out_dim x in_dim``,
    indexed by hop, and ``b`` is a list of ``K + 1`` bias vectors of
    length ``out_dim``. The output has ``(K + 1) * out_dim`` columns:
    hop ``k`` occupies columns ``k * out_dim .. (k + 1) * out_dim``.
    ``K = 0`` degrades exactly to a linear layer over raw features
    (plus ReLU), and ``K = 1`` with the 0-hop block zeroed is a
    row-normalized GCN layer whose self and neighbor terms are kept in
    separate blocks instead of summed.
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
        self.b = [[0.0] * out_dim for _ in range(K + 1)]
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW = [
            [[0.0] * self.in_dim for _ in range(self.out_dim)]
            for _ in range(self.K + 1)
        ]
        self.gb = [[0.0] * self.out_dim for _ in range(self.K + 1)]

    def forward(self, graph, H: list[list[float]]):
        """Returns (H_out, cache). Cache holds everything backprop
        needs: (norm weights, transpose weights, hop powers,
        per-hop pre-activations)."""
        weights = _row_norm_weights(graph)
        weights_T = _row_norm_weights_T(graph)
        powers = [H]
        for _ in range(self.K):
            powers.append(_apply(weights, powers[-1]))
        pre = []  # pre[k][v]: pre-activation row for hop k, node v
        for k in range(self.K + 1):
            Wk = self.W[k]
            bk = self.b[k]
            block = []
            for v in range(graph.n):
                pk = powers[k][v]
                row = list(bk)
                for i in range(self.out_dim):
                    wi = Wk[i]
                    s = 0.0
                    for j in range(self.in_dim):
                        s += wi[j] * pk[j]
                    row[i] += s
                block.append(row)
            pre.append(block)
        out = []
        for v in range(graph.n):
            row = []
            for k in range(self.K + 1):
                row.extend(_relu(pre[k][v]))
            out.append(row)
        return out, (weights, weights_T, powers, pre)

    def backward(self, d_out: list[list[float]], cache
                 ) -> list[list[float]]:
        """Backprop one layer. Returns ``d_H``, the gradient w.r.t.
        the input embeddings, indexed by node id.

        The gradient splits back into the per-hop blocks, each block
        passes through its own ``W_k`` and then back through ``A^k``
        via the transpose weights (``A`` is row-normalized, not
        symmetric, so the forward weights cannot be reused here).
        """
        weights, weights_T, powers, pre = cache
        n = len(weights)
        out_dim = self.out_dim
        d_pre = []  # d_pre[k][v]: gradient w.r.t. pre-activation
        for k in range(self.K + 1):
            block = []
            lo = k * out_dim
            for v in range(n):
                dv = d_out[v][lo:lo + out_dim]
                pv = pre[k][v]
                block.append([d * (1.0 if p > 0.0 else 0.0)
                              for d, p in zip(dv, pv)])
            d_pre.append(block)
        for v in range(n):
            for k in range(self.K + 1):
                dv = d_pre[k][v]
                if not any(dv):
                    continue
                gWk = self.gW[k]
                pk = powers[k][v]
                for i in range(out_dim):
                    di = dv[i]
                    if di == 0.0:
                        continue
                    for j in range(self.in_dim):
                        gWk[i][j] += di * pk[j]
                gbk = self.gb[k]
                for i in range(out_dim):
                    gbk[i] += dv[i]
        d_H = [[0.0] * self.in_dim for _ in range(n)]
        for k in range(self.K + 1):
            Wk = self.W[k]
            # G[v] = W_k^T @ d_pre[k][v].
            cur = []
            for v in range(n):
                dv = d_pre[k][v]
                row = [0.0] * self.in_dim
                for i in range(out_dim):
                    di = dv[i]
                    if di == 0.0:
                        continue
                    wi = Wk[i]
                    for j in range(self.in_dim):
                        row[j] += wi[j] * di
                cur.append(row)
            # Propagate back through (A^T)^k.
            for _ in range(k):
                cur = _apply(weights_T, cur)
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
            bk = self.b[k]
            gbk = self.gb[k]
            for i in range(self.out_dim):
                for j in range(self.in_dim):
                    Wk[i][j] -= lr * gWk[i][j]
                bk[i] -= lr * gbk[i]

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


class GraphMixHop:
    """Two-layer MixHop for binary node classification.

    Layer 1 maps features to ``(K + 1) * hidden_dim`` columns (one
    ``hidden_dim`` block per hop), layer 2 reads those columns and
    produces another ``(K + 1) * hidden_dim`` columns, and a linear +
    sigmoid head scores each node. With ``K = 2`` each layer reads 0,
    1, and 2 hops, so the stack reads up to 4 hops, and the head sees
    the 0-, 1-, and 2-hop blocks of layer 2 separately instead of one
    summed vector. Full-graph forward (no sampling, see the module
    docstring), weighted BCE loss like
    :class:`gnn_fraud.gcn.GraphConvolution`.
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
        self.layer1 = MixHopLayer(in_dim, hidden_dim, K, self.rng)
        self.layer2 = MixHopLayer((K + 1) * hidden_dim, hidden_dim, K,
                                  self.rng)
        out_dim = (K + 1) * hidden_dim
        scale = math.sqrt(2.0 / (out_dim + 1))
        self.w_out = [self.rng.uniform(-scale, scale)
                      for _ in range(out_dim)]
        self.b_out = 0.0
        self.gw_out = [0.0] * out_dim
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
        """Inference, deterministic, no sampling anywhere in MixHop."""
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
        self.gw_out = [0.0] * len(self.w_out)
        self.gb_out = 0.0

        probs, cache = self.forward(graph, targets)
        h2, cache2 = cache["h2"], cache["cache2"]

        B = len(targets)
        weights = {v: (pos_weight if labels[v] == 1 else 1.0)
                   for v in targets}
        loss = sum(weights[v] * _bce(probs[v], labels[v])
                   for v in targets) / B

        # Output layer grads.
        dh2 = [[0.0] * len(self.w_out) for _ in range(graph.n)]
        for v in targets:
            d_logit = weights[v] * (probs[v] - labels[v]) / B
            for j in range(len(self.w_out)):
                self.gw_out[j] += d_logit * h2[v][j]
            self.gb_out += d_logit
            dh2[v] = [w * d_logit for w in self.w_out]

        # Backprop through layer 2, then layer 1 (features are leaves).
        dh1 = self.layer2.backward(dh2, cache2)
        self.layer1.backward(dh1, cache["cache1"])

        # SGD update.
        self.layer1.step(self.lr)
        self.layer2.step(self.lr)
        for j in range(len(self.w_out)):
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

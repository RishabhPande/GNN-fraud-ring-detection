"""GCNII: deep Graph Convolutional Network with initial residual
connections and identity mapping (Chen et al., 2020), from scratch.

A plain GCN collapses when stacked deep: every layer re-averages the
neighborhood, so after a few hops every node's embedding looks alike
(oversmoothing) and the gradient through repeated normalized sums
vanishes. GCNII fixes both with two changes to the layer update, where
``A_hat`` is the GCN's symmetric normalization and ``H^(0)`` is the
first-layer representation::

    H^(l+1) = relu(((1 - alpha) A_hat H^(l) + alpha H^(0))
                    ((1 - beta_l) I + beta_l W^(l)) + b)

* **Initial residual (``alpha``).** Every layer mixes back a fraction
  ``alpha`` of the *initial* representation, not the previous layer's,
  so a node's own signal survives any depth instead of being averaged
  away hop by hop. This is what lets the model go deep without
  oversmoothing; it is also why the residual target is ``H^(0)`` and
  not ``H^(l)`` (a plain residual connection would still drift).
* **Identity mapping (``beta_l``).** The weight matrix is shrunk
  toward the identity, ``(1 - beta_l) I + beta_l W``, with
  ``beta_l = log(lambda / l + 1)`` decaying with depth ``l``. Deeper
  layers therefore stay closer to the identity transform, which keeps
  the singular values of the layer near 1 and stops the gradient from
  vanishing through the stack. With ``beta = 0`` the layer is a pure
  residual propagation (no learned transform at all); with ``beta = 1``
  it is a plain GCN layer on the residual-mixed input.

The input layer is a plain GCN layer (features -> hidden dim), since
the identity mapping needs square matrices; the GCNII layers then all
operate in hidden-dim space, and ``H^(0)`` is that input layer's
output. A final linear layer + sigmoid maps the last hidden
representation to a fraud probability, trained with binary
cross-entropy and full backprop through the whole stack, including
the residual branches (each layer's gradient w.r.t. ``H^(0)`` is
accumulated and added to the gradient flowing through the chain).

Like the GCN, there is no neighbor sampling: the normalization needs
true degrees, so training runs full-batch over the whole graph. All
linear algebra is hand-rolled lists-of-floats -- no numpy, no torch.
"""

from __future__ import annotations

import copy
import math
import random

from gnn_fraud.gcn import GCNLayer, _norm_weights


def _matvec(W: list[list[float]], x: list[float]) -> list[float]:
    return [sum(w * xi for w, xi in zip(row, x)) for row in W]


def _mat_t_vec(W: list[list[float]], v: list[float]) -> list[float]:
    n_cols = len(W[0])
    return [sum(W[r][c] * v[r] for r in range(len(W))) for c in range(n_cols)]


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


# ---------------------------------------------------------------------------
# one GCNII layer
# ---------------------------------------------------------------------------


class GCNIIConv:
    """Single GCNII layer: ``H' = relu(((1 - alpha) A_hat H + alpha H0)
    ((1 - beta) I + beta W) + b)``. Square (hidden_dim -> hidden_dim),
    since the identity mapping only makes sense for square ``W``.

    ``alpha`` is fixed per model; ``beta`` is passed per forward call
    so one layer class serves every depth (the model computes
    ``beta_l = log(lambda / l + 1)`` for layer ``l``).
    """

    def __init__(self, dim: int, alpha: float, rng: random.Random):
        scale = math.sqrt(2.0 / (dim + dim))  # Xavier-ish init
        self.dim = dim
        self.alpha = alpha
        self.W = [
            [rng.uniform(-scale, scale) for _ in range(dim)]
            for _ in range(dim)
        ]
        self.b = [0.0] * dim
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW = [[0.0] * self.dim for _ in range(self.dim)]
        self.gb = [0.0] * self.dim

    def forward(self, graph, H: list[list[float]], H0: list[list[float]],
                beta: float):
        """Returns (H_out, cache). Cache holds everything backprop
        needs: (norm weights, H0, mixed input, pre-activations, beta).
        """
        weights = _norm_weights(graph)
        agg = []
        for v in range(graph.n):
            acc = [0.0] * self.dim
            for u, c in weights[v]:
                hu = H[u]
                for j in range(self.dim):
                    acc[j] += c * hu[j]
            agg.append(acc)
        mix = [
            [(1.0 - self.alpha) * agg[v][j] + self.alpha * H0[v][j]
             for j in range(self.dim)]
            for v in range(graph.n)
        ]
        # M = (1 - beta) I + beta W, applied as M @ mix[v].
        pre = []
        for v in range(graph.n):
            row = []
            for i in range(self.dim):
                s = self.b[i] + (1.0 - beta) * mix[v][i]
                wi = self.W[i]
                for j in range(self.dim):
                    s += beta * wi[j] * mix[v][j]
                row.append(s)
            pre.append(row)
        return [_relu(p) for p in pre], (weights, H0, mix, pre, beta)

    def backward(self, d_out: list[list[float]], cache
                 ) -> tuple[list[list[float]], list[list[float]]]:
        """Backprop one layer.

        Returns ``(d_H, d_H0)`` -- gradients w.r.t. the layer input and
        w.r.t. the initial representation. The normalized adjacency is
        symmetric, so the message pass in reverse reuses the same
        weights, exactly as in :class:`gnn_fraud.gcn.GCNLayer`.
        """
        weights, _H0, mix, pre, beta = cache
        n = len(weights)
        d_pre = [
            [d * (1.0 if p > 0.0 else 0.0) for d, p in zip(d_out[v], pre[v])]
            for v in range(n)
        ]
        for v in range(n):
            dv = d_pre[v]
            if not any(dv):
                continue
            for i in range(self.dim):
                di = dv[i]
                if di == 0.0:
                    continue
                wi = self.gW[i]
                mv = mix[v]
                for j in range(self.dim):
                    wi[j] += beta * di * mv[j]
                self.gb[i] += di
        # d_mix = M^T @ d_pre, with M = (1 - beta) I + beta W.
        d_mix = []
        for v in range(n):
            dv = d_pre[v]
            row = []
            for j in range(self.dim):
                s = (1.0 - beta) * dv[j]
                for i in range(self.dim):
                    s += beta * self.W[i][j] * dv[i]
                row.append(s)
            d_mix.append(row)
        d_agg = [[(1.0 - self.alpha) * x for x in row] for row in d_mix]
        d_H0 = [[self.alpha * x for x in row] for row in d_mix]
        d_H = [[0.0] * self.dim for _ in range(n)]
        for v in range(n):
            for u, c in weights[v]:
                du = d_agg[v]
                row = d_H[u]
                for j in range(self.dim):
                    row[j] += c * du[j]
        return d_H, d_H0

    def step(self, lr: float) -> None:
        for i in range(self.dim):
            for j in range(self.dim):
                self.W[i][j] -= lr * self.gW[i][j]
            self.b[i] -= lr * self.gb[i]

    def state_dict(self):
        return {"W": copy.deepcopy(self.W), "b": copy.deepcopy(self.b)}

    def load_state_dict(self, state) -> None:
        self.W = copy.deepcopy(state["W"])
        self.b = copy.deepcopy(state["b"])
        self.zero_grad()


# ---------------------------------------------------------------------------
# full model: input GCN layer + n GCNII layers + output head
# ---------------------------------------------------------------------------


class GraphGCNII:
    """GCNII for binary node classification.

    ``n_layers`` counts the GCNII layers *after* the input GCN layer,
    so the default (4) propagates 5 hops in total. ``beta_l`` follows
    the paper, ``log(lambda / l + 1)`` with ``l`` 1-based over the
    GCNII layers, so deeper layers stay closer to the identity.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 16,
        n_layers: int = 4,
        alpha: float = 0.2,
        lam: float = 0.5,
        lr: float = 0.5,
        seed: int = 0,
    ):
        self.rng = random.Random(seed)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.alpha = alpha
        self.lr = lr
        self.betas = [math.log(lam / (l + 1) + 1.0) for l in range(n_layers)]
        self.input_layer = GCNLayer(in_dim, hidden_dim, self.rng)
        self.layers = [GCNIIConv(hidden_dim, alpha, self.rng)
                       for _ in range(n_layers)]
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
        h0, cache_in = self.input_layer.forward(graph, graph.features)
        h = h0
        caches = []
        for layer, beta in zip(self.layers, self.betas):
            h, cache = layer.forward(graph, h, h0, beta)
            caches.append(cache)

        probs: dict[int, float] = {}
        for v in targets:
            z = sum(w * x for w, x in zip(self.w_out, h[v])) + self.b_out
            probs[v] = _sigmoid(z)

        cache = {"h0": h0, "cache_in": cache_in, "h": h,
                 "caches": caches, "targets": list(targets)}
        return probs, cache

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        """Inference -- deterministic, no sampling anywhere in GCNII."""
        probs, _ = self.forward(graph, targets)
        return probs

    # -- training step -----------------------------------------------------

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    pos_weight: float = 1.0) -> float:
        """One full-batch SGD step scored on ``targets``. Returns mean
        weighted BCE loss. The forward pass still embeds every node:
        targets' embeddings depend on their (n_layers + 1)-hop
        neighborhoods."""
        self.input_layer.zero_grad()
        for layer in self.layers:
            layer.zero_grad()
        self.gw_out = [0.0] * self.hidden_dim
        self.gb_out = 0.0

        probs, cache = self.forward(graph, targets)
        h = cache["h"]

        B = len(targets)
        weights = {v: (pos_weight if labels[v] == 1 else 1.0)
                   for v in targets}
        loss = sum(weights[v] * _bce(probs[v], labels[v])
                   for v in targets) / B

        # Output layer grads.
        dh = [[0.0] * self.hidden_dim for _ in range(graph.n)]
        for v in targets:
            d_logit = weights[v] * (probs[v] - labels[v]) / B
            for j in range(self.hidden_dim):
                self.gw_out[j] += d_logit * h[v][j]
            self.gb_out += d_logit
            dh[v] = [w * d_logit for w in self.w_out]

        # Backprop through the GCNII layers in reverse. Each layer's
        # d_H0 accumulates into the initial representation's gradient;
        # the chain gradient into the first GCNII layer's input *is*
        # also a gradient w.r.t. H0, since that input is H0 itself.
        d_h0 = [[0.0] * self.hidden_dim for _ in range(graph.n)]
        for layer, layer_cache in zip(reversed(self.layers),
                                      reversed(cache["caches"])):
            dh, d_h0_layer = layer.backward(dh, layer_cache)
            for v in range(graph.n):
                row = d_h0[v]
                add = d_h0_layer[v]
                for j in range(self.hidden_dim):
                    row[j] += add[j]
        for v in range(graph.n):
            row = d_h0[v]
            add = dh[v]
            for j in range(self.hidden_dim):
                row[j] += add[j]
        self.input_layer.backward(d_h0, cache["cache_in"])

        # SGD update.
        self.input_layer.step(self.lr)
        for layer in self.layers:
            layer.step(self.lr)
        for j in range(self.hidden_dim):
            self.w_out[j] -= self.lr * self.gw_out[j]
        self.b_out -= self.lr * self.gb_out
        return loss

    # -- checkpointing ------------------------------------------------------

    def state_dict(self):
        return {
            "input_layer": self.input_layer.state_dict(),
            "layers": [layer.state_dict() for layer in self.layers],
            "w_out": copy.deepcopy(self.w_out),
            "b_out": self.b_out,
        }

    def load_state_dict(self, state) -> None:
        self.input_layer.load_state_dict(state["input_layer"])
        for layer, layer_state in zip(self.layers, state["layers"]):
            layer.load_state_dict(layer_state)
        self.w_out = copy.deepcopy(state["w_out"])
        self.b_out = state["b_out"]

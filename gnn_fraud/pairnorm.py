"""PairNorm GCN (Zhao and Akoglu, 2020), from scratch.

Deep GCN stacks oversmooth: every normalized aggregation pulls a
node toward its neighbors, so after enough layers all embeddings
converge to nearly the same vector and the head has nothing left
to separate. This repo's GCNII fights that with residual
connections back to the input layer. PairNorm fights it at the
embedding geometry instead: after each convolutional layer, the
embeddings are re-centered and rescaled so their total pairwise
distance stays exactly constant, whatever the aggregation did::

    Hc_v  = H_v - mean_u H_u                (centering)
    H'_v  = s * sqrt(n) * Hc_v / ||Hc||_F   (rescaling)

with ``s`` the target scale (``scale=1.0`` by default). Two
invariants follow directly and the tests pin both:

* the output rows sum to zero (the mean embedding is exactly
  the zero vector), and
* ``sum_v ||H'_v||^2 == s^2 * n`` exactly, so the sum of squared
  pairwise distances ``sum_{u<v} ||H'_u - H'_v||^2`` equals
  ``s^2 * n^2`` at every layer, at any depth. Embeddings can
  rotate and redistribute, but they cannot all collapse to one
  point: the collapse would need total distance zero.

The operation is scale-invariant in its input (``pairnorm(c H)``
equals ``pairnorm(H)`` for any ``c > 0``), which is the flip
side of the invariant: PairNorm keeps the *shape* of the cloud
and discards its overall size, so a layer whose signal lives in
embedding magnitude loses that channel here.

Backprop runs through both steps. With ``X`` the centered
input, ``r = ||X||_F`` and ``a = s sqrt(n) / r``, the output is
``Y = a X`` and

    dX = a * (dY - (X / r) * <dY, X / r>)

(the normalization Jacobian: the component of the gradient
along ``X`` itself is removed, since moving ``X`` radially only
changes ``r``), and the centering's transpose is centering
itself, so ``dH = dX - mean(dX)``. An all-equal input (``r = 0``,
the fully oversmoothed state) maps to zeros with zero gradient:
PairNorm can prevent collapse layer by layer, it cannot undo a
collapse that has already happened exactly.

The model is a stack of :class:`~gnn_fraud.gcn.GCNLayer`s (the
repo's symmetric-normalized convolution, reused unchanged) with
a PairNorm after each one, ``n_layers=2`` by default and deeper
stacks supported for the oversmoothing comparison, then the
usual linear + sigmoid head. All linear algebra is hand-rolled
lists-of-floats - no numpy, no torch.
"""

from __future__ import annotations

import copy
import math
import random

from gnn_fraud.gcn import GCNLayer


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _bce(p: float, y: int) -> float:
    p = min(max(p, 1e-7), 1.0 - 1e-7)
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


def pairnorm_forward(H: list[list[float]], scale: float = 1.0):
    """Center and rescale ``H`` (one row per node).

    Returns ``(Y, cache)``. Guarantees ``sum_v Y_v == 0`` per
    dimension and ``sum_v ||Y_v||^2 == scale^2 * n`` (unless the
    centered input is exactly zero, in which case ``Y`` is all
    zeros)."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    n = len(H)
    if n == 0:
        raise ValueError("H must have at least one row")
    width = len(H[0])
    if any(len(row) != width for row in H):
        raise ValueError("all rows of H must share one width")
    mean = [sum(H[v][j] for v in range(n)) / n for j in range(width)]
    centered = [[H[v][j] - mean[j] for j in range(width)]
                for v in range(n)]
    norm = math.sqrt(sum(x * x for row in centered for x in row))
    if norm == 0.0:
        Y = [[0.0] * width for _ in range(n)]
        return Y, (centered, norm, 0.0)
    alpha = scale * math.sqrt(n) / norm
    Y = [[alpha * x for x in row] for row in centered]
    return Y, (centered, norm, alpha)


def pairnorm_backward(dY: list[list[float]], cache) -> list[list[float]]:
    """Gradient of :func:`pairnorm_forward` w.r.t. its input.

    ``cache`` is the tuple ``forward`` returned. See the module
    docstring for the derivation: remove the radial component,
    scale by ``alpha``, then re-center (the centering Jacobian's
    transpose is centering)."""
    centered, norm, alpha = cache
    n = len(dY)
    width = len(dY[0])
    if norm == 0.0:
        return [[0.0] * width for _ in range(n)]
    # <dY, X / r>: the radial component of the output gradient.
    radial = sum(dY[v][j] * centered[v][j]
                 for v in range(n) for j in range(width)) / norm
    d_centered = [
        [alpha * (dY[v][j] - radial * centered[v][j] / norm)
         for j in range(width)]
        for v in range(n)
    ]
    mean_grad = [sum(d_centered[v][j] for v in range(n)) / n
                 for j in range(width)]
    return [[d_centered[v][j] - mean_grad[j] for j in range(width)]
            for v in range(n)]


def pairnorm(H: list[list[float]], scale: float = 1.0) -> list[list[float]]:
    """The PairNorm map alone (no cache): center, then rescale
    to total squared norm ``scale^2 * n``."""
    Y, _ = pairnorm_forward(H, scale)
    return Y


def total_pairwise_sq_distance(H: list[list[float]]) -> float:
    """``sum_{u<v} ||H_u - H_v||^2``, computed through the
    identity ``n * sum_v ||H_v - mean||^2`` so it costs one pass
    instead of a quadratic loop."""
    n = len(H)
    if n == 0:
        return 0.0
    width = len(H[0])
    mean = [sum(H[v][j] for v in range(n)) / n for j in range(width)]
    centered_sq = sum((H[v][j] - mean[j]) ** 2
                      for v in range(n) for j in range(width))
    return n * centered_sq


class GraphPairNormGCN:
    """A stack of GCN layers, each followed by PairNorm, with a
    linear + sigmoid head.

    ``n_layers=2`` matches the repo's other two-layer models;
    deeper stacks are where PairNorm's invariant is the point.
    ``predict`` returns ``{v: probability}`` like the other
    models, so the shared training/evaluation protocol applies
    unchanged. ``embeddings(graph)`` returns the final hidden
    rows (after the last PairNorm) for oversmoothing checks.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 16,
        n_layers: int = 2,
        scale: float = 1.0,
        lr: float = 0.5,
        seed: int = 0,
    ):
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.rng = random.Random(seed)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.scale = scale
        self.lr = lr
        self.layers: list[GCNLayer] = []
        dim = in_dim
        for _ in range(n_layers):
            self.layers.append(GCNLayer(dim, hidden_dim, self.rng))
            dim = hidden_dim
        out_scale = math.sqrt(2.0 / (hidden_dim + 1))
        self.w_out = [self.rng.uniform(-out_scale, out_scale)
                      for _ in range(hidden_dim)]
        self.b_out = 0.0
        self.gw_out = [0.0] * hidden_dim
        self.gb_out = 0.0

    # -- forward ----------------------------------------------------------

    def forward(self, graph, targets: list[int]):
        """Forward pass over the whole graph. Returns (probs dict,
        cache) with probs only for the requested targets."""
        H = [list(row) for row in graph.features]
        layer_caches = []
        pn_caches = []
        for layer in self.layers:
            H, lc = layer.forward(graph, H)
            H, pc = pairnorm_forward(H, self.scale)
            layer_caches.append(lc)
            pn_caches.append(pc)
        probs: dict[int, float] = {}
        for v in targets:
            z = sum(w * h for w, h in zip(self.w_out, H[v])) + self.b_out
            probs[v] = _sigmoid(z)
        cache = {"layer_caches": layer_caches, "pn_caches": pn_caches,
                 "H": H, "targets": list(targets)}
        return probs, cache

    def embeddings(self, graph) -> list[list[float]]:
        """Final hidden rows after the last PairNorm, one per node."""
        H = [list(row) for row in graph.features]
        for layer in self.layers:
            H, _ = layer.forward(graph, H)
            H, _ = pairnorm_forward(H, self.scale)
        return H

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        probs, _ = self.forward(graph, targets)
        return probs

    # -- training step -----------------------------------------------------

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    pos_weight: float = 1.0) -> float:
        """One full-batch SGD step scored on ``targets``. Returns
        mean weighted BCE loss. ``lr=0.0`` accumulates gradients
        without moving the weights, which the numeric gradient
        check relies on."""
        for layer in self.layers:
            layer.zero_grad()
        self.gw_out = [0.0] * self.hidden_dim
        self.gb_out = 0.0

        probs, cache = self.forward(graph, targets)
        H = cache["H"]

        B = len(targets)
        wts = {v: (pos_weight if labels[v] == 1 else 1.0)
               for v in targets}
        loss = sum(wts[v] * _bce(probs[v], labels[v])
                   for v in targets) / B

        dH = [[0.0] * self.hidden_dim for _ in range(graph.n)]
        for v in targets:
            dl = wts[v] * (probs[v] - labels[v]) / B
            if dl == 0.0:
                continue
            for j in range(self.hidden_dim):
                self.gw_out[j] += dl * H[v][j]
                dH[v][j] = dl * self.w_out[j]
            self.gb_out += dl

        for li in reversed(range(self.n_layers)):
            dH = pairnorm_backward(dH, cache["pn_caches"][li])
            dH = self.layers[li].backward(dH, cache["layer_caches"][li])

        for layer in self.layers:
            layer.step(self.lr)
        for j in range(self.hidden_dim):
            self.w_out[j] -= self.lr * self.gw_out[j]
        self.b_out -= self.lr * self.gb_out
        return loss

    # -- checkpointing ------------------------------------------------------

    def state_dict(self):
        return {
            "layers": [layer.state_dict() for layer in self.layers],
            "w_out": copy.deepcopy(self.w_out),
            "b_out": self.b_out,
        }

    def load_state_dict(self, state) -> None:
        if len(state["layers"]) != self.n_layers:
            raise ValueError("checkpoint layer count does not match")
        for layer, saved in zip(self.layers, state["layers"]):
            layer.load_state_dict(saved)
        self.w_out = copy.deepcopy(state["w_out"])
        self.b_out = state["b_out"]

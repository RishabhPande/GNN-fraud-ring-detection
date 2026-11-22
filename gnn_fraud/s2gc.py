"""Simple Spectral Graph Convolution (S2GC, Zhu et al., 2021),
from scratch.

The propagation family in this repo spans two extremes. SGC keeps
only the last propagation power, ``S_hat^K H``: cheap, but the
K-hop view is the most oversmoothed one, and every shallower view
is thrown away. SIGN keeps every power but concatenates them and
hands the head a separate weight block per hop, so the head has
to learn how to mix depths. S2GC takes the middle path the paper
proposes: average the powers, learn nothing about the mix at all::

    H^(0)_v = relu(W1 @ x_v + b1)            (MLP encoder, no graph)
    Z_v     = (1 / (K + 1)) sum_k (S_hat^k H^(0))_v
    logit   = w_out . Z_v + b_out

    S_hat = D_hat^{-1/2} (A + I) D_hat^{-1/2}

the self-looped symmetric normalization the GCN family uses (the
operator SGC propagates with), spectrum in [-1, 1], symmetric, so
the backward pass is the same averaging applied to the output
gradient. The average is a fixed low-pass filter whose frequency
response is the mean of the powers' responses: it keeps a node's
own encoder state in the mix at weight 1 / (K + 1) no matter how
deep K runs, which is exactly the term SGC loses, while adding
no learned filter parameters at all - GPR-GNN learns K + 1 of
them, GDC learns one, S2GC learns zero.

``K = 0`` degrades exactly to an MLP on raw features (the average
of one power is the power itself). All linear algebra is
hand-rolled lists-of-floats - no numpy, no torch.
"""

from __future__ import annotations

import copy
import math
import random


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _bce(p: float, y: int) -> float:
    p = min(max(p, 1e-7), 1.0 - 1e-7)
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


def _norm_weights_self_loop(graph) -> list[list[tuple[int, float]]]:
    """Self-looped symmetric normalization weights:
    ``S_hat = D_hat^{-1/2} (A + I) D_hat^{-1/2}``, the GCN
    family's operator. Every node lists itself first, so an
    isolated node's propagated state is exactly its own state
    at every power."""
    d = [graph.degree(v) + 1 for v in range(graph.n)]
    weights: list[list[tuple[int, float]]] = []
    for v in range(graph.n):
        row = [(v, 1.0 / d[v])]
        for u in sorted(set(graph.neighbors(v))):
            row.append((u, 1.0 / math.sqrt(d[v] * d[u])))
        weights.append(row)
    return weights


def _apply_S(weights, H, n: int) -> list[list[float]]:
    width = len(H[0])
    out = []
    for v in range(n):
        acc = [0.0] * width
        for u, c in weights[v]:
            hu = H[u]
            for j in range(width):
                acc[j] += c * hu[j]
        out.append(acc)
    return out


def propagation_powers(graph, h0: list[list[float]], K: int):
    """The K + 1 self-looped propagation powers
    ``[h0, S_hat h0, ..., S_hat^K h0]``, one application of
    ``S_hat`` per new power. ``K = 0`` returns ``[h0]`` (a copy)."""
    if K < 0:
        raise ValueError("K must be non-negative")
    if len(h0) != graph.n:
        raise ValueError("h0 must have one row per node")
    if any(len(row) != len(h0[0]) for row in h0):
        raise ValueError("all hidden rows must share one width")
    weights = _norm_weights_self_loop(graph)
    powers = [[list(row) for row in h0]]
    for _ in range(K):
        powers.append(_apply_S(weights, powers[-1], graph.n))
    return powers


def averaged_propagation(graph, h0: list[list[float]], K: int):
    """The S2GC filter itself: the elementwise mean of the
    propagation powers ``(1 / (K + 1)) sum_k S_hat^k h0``.
    ``K = 0`` returns a copy of ``h0``."""
    powers = propagation_powers(graph, h0, K)
    n = graph.n
    width = len(h0[0])
    scale = 1.0 / (K + 1)
    out = [[0.0] * width for _ in range(n)]
    for power in powers:
        for v in range(n):
            pv = power[v]
            ov = out[v]
            for j in range(width):
                ov[j] += scale * pv[j]
    return out


class S2GC:
    """S2GC: an MLP encoder, a parameter-free averaged
    propagation of order ``K`` over the hidden states, and a
    linear head.

    ``predict`` takes a graph and target node ids and returns
    ``{v: probability}``, matching the other models' interface,
    so the shared training/evaluation protocol applies unchanged.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 16,
        K: int = 5,
        lr: float = 0.5,
        seed: int = 0,
    ):
        if K < 0:
            raise ValueError("K must be non-negative")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        rng = random.Random(seed)
        scale = math.sqrt(2.0 / (in_dim + hidden_dim))
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.K = K
        self.lr = lr
        self.W1 = [
            [rng.uniform(-scale, scale) for _ in range(in_dim)]
            for _ in range(hidden_dim)
        ]
        self.b1 = [0.0] * hidden_dim
        out_scale = math.sqrt(2.0 / (hidden_dim + 1))
        self.w_out = [rng.uniform(-out_scale, out_scale)
                      for _ in range(hidden_dim)]
        self.b_out = 0.0
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW1 = [[0.0] * self.in_dim for _ in range(self.hidden_dim)]
        self.gb1 = [0.0] * self.hidden_dim
        self.gw_out = [0.0] * self.hidden_dim
        self.gb_out = 0.0

    # -- forward ----------------------------------------------------------

    def forward(self, graph, targets: list[int]):
        """Forward pass over the whole graph. Returns (probs dict,
        cache) with probs only for the requested targets."""
        weights = _norm_weights_self_loop(graph)
        pre1 = []
        h0 = []
        for v in range(graph.n):
            x = graph.features[v]
            pre = [
                sum(w * xi for w, xi in zip(row, x)) + b
                for row, b in zip(self.W1, self.b1)
            ]
            pre1.append(pre)
            h0.append([p if p > 0.0 else 0.0 for p in pre])
        z = averaged_propagation(graph, h0, self.K)
        probs: dict[int, float] = {}
        for v in targets:
            logit = (sum(w * zj for w, zj in zip(self.w_out, z[v]))
                     + self.b_out)
            probs[v] = _sigmoid(logit)
        cache = {
            "weights": weights,
            "pre1": pre1,
            "h0": h0,
            "z": z,
            "targets": list(targets),
        }
        return probs, cache

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        """Inference - the averaging is deterministic, so this is exact."""
        probs, _ = self.forward(graph, targets)
        return probs

    # -- training step -----------------------------------------------------

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    pos_weight: float = 1.0) -> float:
        """One SGD step on ``targets``. Returns mean weighted BCE loss.

        The filter is symmetric and parameter-free, so the encoder
        gradient is the same averaged propagation applied to the
        output gradient, then the usual ReLU/linear backward into
        the encoder. ``lr=0.0`` accumulates gradients without
        moving the weights, which the numeric gradient check
        relies on.
        """
        self.zero_grad()
        probs, cache = self.forward(graph, targets)
        weights = cache["weights"]
        pre1 = cache["pre1"]
        z = cache["z"]

        B = len(targets)
        wts = {v: (pos_weight if labels[v] == 1 else 1.0)
               for v in targets}
        loss = sum(wts[v] * _bce(probs[v], labels[v])
                   for v in targets) / B

        d_z = [[0.0] * self.hidden_dim for _ in range(graph.n)]
        for v in targets:
            dl = wts[v] * (probs[v] - labels[v]) / B
            if dl == 0.0:
                continue
            for j in range(self.hidden_dim):
                d_z[v][j] = dl * self.w_out[j]
                self.gw_out[j] += dl * z[v][j]
            self.gb_out += dl

        # Encoder backward: the averaged filter is a symmetric
        # polynomial in S_hat, so its transpose is itself.
        d_h0 = _apply_average(weights, d_z, graph.n, self.K)
        for v in range(graph.n):
            dh = d_h0[v]
            if not any(dh):
                continue
            x = graph.features[v]
            for j in range(self.hidden_dim):
                d_pre = dh[j] * (1.0 if pre1[v][j] > 0.0 else 0.0)
                if d_pre == 0.0:
                    continue
                for k in range(self.in_dim):
                    self.gW1[j][k] += d_pre * x[k]
                self.gb1[j] += d_pre

        # SGD update.
        for j in range(self.hidden_dim):
            for k in range(self.in_dim):
                self.W1[j][k] -= self.lr * self.gW1[j][k]
            self.b1[j] -= self.lr * self.gb1[j]
            self.w_out[j] -= self.lr * self.gw_out[j]
        self.b_out -= self.lr * self.gb_out
        return loss

    # -- checkpointing ------------------------------------------------------

    def state_dict(self):
        return {
            "W1": copy.deepcopy(self.W1),
            "b1": copy.deepcopy(self.b1),
            "w_out": copy.deepcopy(self.w_out),
            "b_out": self.b_out,
        }

    def load_state_dict(self, state) -> None:
        self.W1 = copy.deepcopy(state["W1"])
        self.b1 = copy.deepcopy(state["b1"])
        self.w_out = copy.deepcopy(state["w_out"])
        self.b_out = state["b_out"]
        self.zero_grad()


def _apply_average(weights, H, n: int, K: int) -> list[list[float]]:
    """Apply ``(1 / (K + 1)) sum_k S_hat^k`` to ``H`` in K
    propagation steps, accumulating the powers of S_hat."""
    width = len(H[0])
    scale = 1.0 / (K + 1)
    out = [[scale * x for x in row] for row in H]
    cur = [list(row) for row in H]
    for _ in range(K):
        cur = _apply_S(weights, cur, n)
        for v in range(n):
            for t in range(width):
                out[v][t] += scale * cur[v][t]
    return out

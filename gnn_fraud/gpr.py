"""Generalized PageRank Graph Neural Network (GPR-GNN, Chien et al.,
2021), from scratch.

GPR-GNN is APPNP with the teleport schedule learned instead of fixed.
Where :mod:`gnn_fraud.appnp` propagates scalar *logits* with frozen
personalized PageRank coefficients, GPR-GNN propagates the hidden
*representations* and learns the weight of every propagation step::

    H^(0)_v = relu(W1 @ x_v + b1)          (MLP encoder, no graph)
    H^(k)   = S H^(k-1),                   S = D_hat^{-1/2} A_hat D_hat^{-1/2}
    Z_v     = sum_k gamma_k H^(k)_v        (learned polynomial weights)
    logit   = w_out . Z_v + b_out

The coefficients ``gamma`` start at the personalized PageRank weights
(``gamma_k = alpha * (1 - alpha)^k`` for ``k < K``,
``gamma_K = (1 - alpha)^K``, so they sum to 1) and are trained jointly
with the rest of the model. That is the whole point of the paper: the
fixed PPR coefficients are a hand-picked polynomial, and the learned
ones can go anywhere the data wants -- including negative, which turns
the propagation into a high-pass filter that sharpens differences
between a node and its neighborhood instead of smoothing them away.
A ring member whose neighbors are also ring members wants the low-pass
(smoothing) end; a bridge account between the ring and the background
wants the high-pass end. One fixed polynomial cannot serve both.

Backprop runs through the weighted sum (the ``gamma`` gradient is the
dot product of the ``Z`` gradient with each step's hidden states),
then unwinds the propagation chain step by step -- each step's
gradient flows back through the symmetric ``S`` -- accumulating into
the encoder's gradient, then through the MLP. All linear algebra is
hand-rolled lists-of-floats -- no numpy, no torch.
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


def _norm_weights(graph) -> list[list[tuple[int, float]]]:
    """Symmetric normalization weights, the same ``S`` that
    :mod:`gnn_fraud.gcn`, :mod:`gnn_fraud.sgc`, and
    :mod:`gnn_fraud.appnp` use: for node ``v``, the list of
    ``(u, c_vu)`` over ``u in N(v) u {v}`` with
    ``c_vu = 1 / sqrt(d_hat_v * d_hat_u)``, ``d_hat = degree + 1``."""
    d_hat = [graph.degree(v) + 1 for v in range(graph.n)]
    weights: list[list[tuple[int, float]]] = []
    for v in range(graph.n):
        row = []
        for u in sorted(set(graph.neighbors(v)) | {v}):
            row.append((u, 1.0 / math.sqrt(d_hat[v] * d_hat[u])))
        weights.append(row)
    return weights


def ppr_gamma_init(steps: int, alpha: float) -> list[float]:
    """Personalized PageRank coefficients for ``steps`` propagation
    rounds: ``gamma_k = alpha * (1 - alpha)^k`` for ``k < steps`` and
    ``gamma_steps = (1 - alpha)^steps``. They sum to 1, so the initial
    weighted sum is a convex combination of the propagation steps --
    exactly the APPNP predictor with the same ``alpha`` and step count,
    except here the coefficients are learned from that starting point.
    ``steps=0`` gives ``[1.0]``: no propagation, the encoder alone.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    gamma = [alpha * (1.0 - alpha) ** k for k in range(steps)]
    gamma.append((1.0 - alpha) ** steps)
    return gamma


def propagate_hidden(graph, h0: list[list[float]],
                     steps: int) -> list[list[list[float]]]:
    """Propagate hidden states ``steps`` times with the symmetric
    normalized adjacency: ``H^(k) = S H^(k-1)``. Returns the trajectory
    ``[H^(0), ..., H^(steps)]``; ``steps=0`` returns ``[h0]`` (a copy,
    so mutating the result cannot touch the input).
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if any(len(row) != len(h0[0]) for row in h0):
        raise ValueError("all hidden rows must share one width")
    if len(h0) != graph.n:
        raise ValueError("h0 must have one row per node")
    weights = _norm_weights(graph)
    trajectory = [[list(row) for row in h0]]
    h = [list(row) for row in h0]
    width = len(h0[0])
    for _ in range(steps):
        nxt = []
        for v in range(graph.n):
            acc = [0.0] * width
            for u, c in weights[v]:
                hu = h[u]
                for j in range(width):
                    acc[j] += c * hu[j]
            nxt.append(acc)
        h = nxt
        trajectory.append([list(row) for row in h])
    return trajectory


class GPRGNN:
    """GPR-GNN: an MLP encoder, K rounds of normalized propagation of
    the hidden states, a learned weighted sum over the propagation
    steps, and a linear head.

    ``predict`` takes a graph and target node ids and returns
    ``{v: probability}``, matching the other models' interface, so the
    shared training/evaluation protocol applies unchanged.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 16,
        steps: int = 10,
        alpha: float = 0.1,
        lr: float = 0.5,
        seed: int = 0,
    ):
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        rng = random.Random(seed)
        scale = math.sqrt(2.0 / (in_dim + hidden_dim))
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.steps = steps
        self.alpha = alpha
        self.lr = lr
        self.W1 = [
            [rng.uniform(-scale, scale) for _ in range(in_dim)]
            for _ in range(hidden_dim)
        ]
        self.b1 = [0.0] * hidden_dim
        self.gamma = ppr_gamma_init(steps, alpha)
        out_scale = math.sqrt(2.0 / (hidden_dim + 1))
        self.w_out = [rng.uniform(-out_scale, out_scale)
                      for _ in range(hidden_dim)]
        self.b_out = 0.0
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW1 = [[0.0] * self.in_dim for _ in range(self.hidden_dim)]
        self.gb1 = [0.0] * self.hidden_dim
        self.ggamma = [0.0] * (self.steps + 1)
        self.gw_out = [0.0] * self.hidden_dim
        self.gb_out = 0.0

    # -- forward ----------------------------------------------------------

    def forward(self, graph, targets: list[int]):
        """Forward pass over the whole graph. Returns (probs dict,
        cache) with probs only for the requested targets. The cache
        holds everything backprop needs: the norm weights, the MLP
        pre-activations and encoder states, the propagation trajectory,
        and the weighted-sum representations."""
        weights = _norm_weights(graph)
        pre1 = []
        h0 = []
        for v in range(graph.n):
            x = graph.features[v]
            pre = [
                sum(w * xi for w, xi in zip(row, x)) + b
                for row, b in zip(self.W1, self.b1)
            ]
            h = [p if p > 0.0 else 0.0 for p in pre]
            pre1.append(pre)
            h0.append(h)
        trajectory = propagate_hidden(graph, h0, self.steps)
        z = []
        for v in range(graph.n):
            acc = [0.0] * self.hidden_dim
            for k in range(self.steps + 1):
                gk = self.gamma[k]
                hk = trajectory[k][v]
                for j in range(self.hidden_dim):
                    acc[j] += gk * hk[j]
            z.append(acc)
        probs: dict[int, float] = {}
        for v in targets:
            logit = (sum(w * zj for w, zj in zip(self.w_out, z[v]))
                     + self.b_out)
            probs[v] = _sigmoid(logit)
        cache = {
            "weights": weights,
            "pre1": pre1,
            "h0": h0,
            "trajectory": trajectory,
            "z": z,
            "targets": list(targets),
        }
        return probs, cache

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        """Inference -- propagation is deterministic, so this is exact."""
        probs, _ = self.forward(graph, targets)
        return probs

    # -- training step -----------------------------------------------------

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    pos_weight: float = 1.0) -> float:
        """One SGD step on ``targets``. Returns mean weighted BCE loss.

        The forward pass embeds every node (a target's propagated score
        depends on its neighbors' encoder states), but the loss and its
        seed gradient cover only ``targets``. Backprop first unwinds
        the weighted sum into a per-step gradient and a ``gamma``
        gradient, then unwinds the propagation chain step by step into
        the encoder, then the MLP. ``lr=0.0`` accumulates gradients
        without moving the weights, which the numeric gradient check
        relies on.
        """
        self.zero_grad()
        probs, cache = self.forward(graph, targets)
        weights = cache["weights"]
        pre1 = cache["pre1"]
        trajectory = cache["trajectory"]
        z = cache["z"]

        B = len(targets)
        wts = {v: (pos_weight if labels[v] == 1 else 1.0)
               for v in targets}
        loss = sum(wts[v] * _bce(probs[v], labels[v])
                   for v in targets) / B

        # Seed gradient on the final logits.
        d_logit = [0.0] * graph.n
        for v in targets:
            d_logit[v] = wts[v] * (probs[v] - labels[v]) / B

        # Head backward: gradient wrt Z, and the head's own gradients.
        d_z = [[0.0] * self.hidden_dim for _ in range(graph.n)]
        for v in targets:
            dl = d_logit[v]
            if dl == 0.0:
                continue
            for j in range(self.hidden_dim):
                d_z[v][j] = dl * self.w_out[j]
                self.gw_out[j] += dl * z[v][j]
            self.gb_out += dl

        # Weighted-sum backward: gamma_k's gradient is the dot product
        # of dZ with step k's hidden states, and step k's hidden states
        # carry gamma_k times dZ.
        d_h = []
        for k in range(self.steps + 1):
            gk = self.gamma[k]
            hk = trajectory[k]
            d_hk = [[0.0] * self.hidden_dim for _ in range(graph.n)]
            for v in range(graph.n):
                dzv = d_z[v]
                if not any(dzv):
                    continue
                gsum = 0.0
                hkv = hk[v]
                for j in range(self.hidden_dim):
                    gsum += dzv[j] * hkv[j]
                    d_hk[v][j] = gk * dzv[j]
                self.ggamma[k] += gsum
            d_h.append(d_hk)

        # Unwind the propagation chain: step k's gradient flows back
        # through S (symmetric, so the reverse pass reuses the same
        # weights) into step k-1's gradient, accumulating as it goes.
        # After the loop, d_h[0] is the full gradient wrt H^(0).
        for k in range(self.steps, 0, -1):
            d_cur = d_h[k]
            d_prev = d_h[k - 1]
            for v in range(graph.n):
                dcur_v = d_cur[v]
                if not any(dcur_v):
                    continue
                for u, c in weights[v]:
                    dpu = d_prev[u]
                    for j in range(self.hidden_dim):
                        dpu[j] += c * dcur_v[j]

        # MLP backward, over every node: propagation mixed every node's
        # encoder state into the targets' scores, so every node carries
        # gradient even though the loss covered only the targets.
        d_h0 = d_h[0]
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
        for k in range(self.steps + 1):
            self.gamma[k] -= self.lr * self.ggamma[k]
        self.b_out -= self.lr * self.gb_out
        return loss

    # -- checkpointing ------------------------------------------------------

    def state_dict(self):
        return {
            "W1": copy.deepcopy(self.W1),
            "b1": copy.deepcopy(self.b1),
            "gamma": copy.deepcopy(self.gamma),
            "w_out": copy.deepcopy(self.w_out),
            "b_out": self.b_out,
        }

    def load_state_dict(self, state) -> None:
        self.W1 = copy.deepcopy(state["W1"])
        self.b1 = copy.deepcopy(state["b1"])
        self.gamma = copy.deepcopy(state["gamma"])
        self.w_out = copy.deepcopy(state["w_out"])
        self.b_out = state["b_out"]
        self.zero_grad()

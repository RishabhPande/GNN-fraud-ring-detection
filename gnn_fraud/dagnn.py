"""Deep Adaptive Graph Neural Network (DAGNN, Liu et al., 2020),
from scratch.

GPR-GNN learns one global weight per propagation step: every node
gets the same polynomial. DAGNN asks why the hop mix should be
global at all. A ring member buried inside the pocket wants the
deep, smoothed views; a bridge account on the pocket's edge wants
its own features and its 1-hop view, before the smoothing washes
the contrast out. So DAGNN propagates the hidden states K steps
and lets *each node* choose its own mix::

    H^(0)_v = relu(W1 @ x_v + b1)          (MLP encoder, no graph)
    H^(k)   = S H^(k-1),                   S = D_hat^{-1/2} A_hat D_hat^{-1/2}
    l_{v,k} = s . H^(k)_v                  (one learned projection)
    a_{v,k} = softmax_k(l_{v,k})           (per-node hop attention)
    Z_v     = sum_k a_{v,k} H^(k)_v
    logit   = w_out . Z_v + b_out

The projection ``s`` is a single hidden-width vector, shared by
every node and every step: the model learns one direction in
hidden space along which a propagated view looks trustworthy, and
each node's attention is how strongly each of its views aligns
with that direction. ``s`` starts at zero, so at init every
attention weight is exactly ``1 / (K + 1)`` and Z is the plain
mean of the propagation steps (the S2GC-style average). Learning
moves the weights off uniform only where a node's views actually
differ along ``s``.

Backprop runs through both paths into each step's hidden state:
the direct weighted-sum path (``a_{v,k}`` times the output
gradient) and the attention path. With ``dl_{v,k} = a_{v,k} *
(dZ_v . H^(k)_v - dZ_v . Z_v)`` (the softmax Jacobian applied to
the weighted sum), step k's hidden state also carries
``dl_{v,k} * s``, and ``s`` itself accumulates
``sum_k dl_{v,k} H^(k)_v``. From there the propagation chain
unwinds step by step through the symmetric ``S`` exactly as in
GPR-GNN, then through the MLP encoder. All linear algebra is
hand-rolled lists-of-floats -- no numpy, no torch.
"""

from __future__ import annotations

import copy
import math
import random

from gnn_fraud.gpr import propagate_hidden


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
    :mod:`gnn_fraud.gpr` propagates with: for node ``v``, the list
    of ``(u, c_vu)`` over ``u in N(v) u {v}`` with
    ``c_vu = 1 / sqrt(d_hat_v * d_hat_u)``, ``d_hat = degree + 1``."""
    d_hat = [graph.degree(v) + 1 for v in range(graph.n)]
    weights: list[list[tuple[int, float]]] = []
    for v in range(graph.n):
        row = []
        for u in sorted(set(graph.neighbors(v)) | {v}):
            row.append((u, 1.0 / math.sqrt(d_hat[v] * d_hat[u])))
        weights.append(row)
    return weights


def hop_attention(h_steps: list[list[float]], s: list[float]) -> list[float]:
    """Per-node hop attention for one node: ``softmax_k(s . H^(k))``.

    ``h_steps`` is the list of that node's hidden vectors, one per
    propagation step. With ``s`` all zeros (the init) every weight
    is exactly ``1 / len(h_steps)``. The weights always sum to 1.
    """
    if not h_steps:
        raise ValueError("h_steps must contain at least one step")
    logits = [sum(sj * hj for sj, hj in zip(s, h)) for h in h_steps]
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


class DAGNN:
    """DAGNN: an MLP encoder, K rounds of normalized propagation of
    the hidden states, a per-node softmax hop attention driven by
    one learned projection vector ``s``, and a linear head.

    ``predict`` takes a graph and target node ids and returns
    ``{v: probability}``, matching the other models' interface, so
    the shared training/evaluation protocol applies unchanged.
    ``attention_weights(graph, v)`` returns node ``v``'s hop
    attention, one weight per propagation step.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 16,
        steps: int = 10,
        lr: float = 0.5,
        seed: int = 0,
    ):
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        rng = random.Random(seed)
        scale = math.sqrt(2.0 / (in_dim + hidden_dim))
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.steps = steps
        self.lr = lr
        self.W1 = [
            [rng.uniform(-scale, scale) for _ in range(in_dim)]
            for _ in range(hidden_dim)
        ]
        self.b1 = [0.0] * hidden_dim
        # Zero init: attention starts exactly uniform over the steps.
        self.s = [0.0] * hidden_dim
        out_scale = math.sqrt(2.0 / (hidden_dim + 1))
        self.w_out = [rng.uniform(-out_scale, out_scale)
                      for _ in range(hidden_dim)]
        self.b_out = 0.0
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW1 = [[0.0] * self.in_dim for _ in range(self.hidden_dim)]
        self.gb1 = [0.0] * self.hidden_dim
        self.gs = [0.0] * self.hidden_dim
        self.gw_out = [0.0] * self.hidden_dim
        self.gb_out = 0.0

    # -- forward ----------------------------------------------------------

    def _encode(self, graph):
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
        return pre1, h0

    def forward(self, graph, targets: list[int]):
        """Forward pass over the whole graph. Returns (probs dict,
        cache) with probs only for the requested targets. The cache
        holds the norm weights, the encoder pre-activations, the
        propagation trajectory, the per-node hop attention, and the
        combined representations backprop needs."""
        weights = _norm_weights(graph)
        pre1, h0 = self._encode(graph)
        trajectory = propagate_hidden(graph, h0, self.steps)
        attn = []
        z = []
        for v in range(graph.n):
            a = hop_attention([trajectory[k][v]
                               for k in range(self.steps + 1)], self.s)
            attn.append(a)
            acc = [0.0] * self.hidden_dim
            for k in range(self.steps + 1):
                ak = a[k]
                hk = trajectory[k][v]
                for j in range(self.hidden_dim):
                    acc[j] += ak * hk[j]
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
            "attn": attn,
            "z": z,
            "targets": list(targets),
        }
        return probs, cache

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        """Inference -- propagation is deterministic, so this is exact."""
        probs, _ = self.forward(graph, targets)
        return probs

    def attention_weights(self, graph, v: int) -> list[float]:
        """Node ``v``'s hop attention under the current model: one
        weight per propagation step, summing to 1. Step 0 is the
        node's own encoder state; higher steps are smoother views."""
        _, h0 = self._encode(graph)
        trajectory = propagate_hidden(graph, h0, self.steps)
        return hop_attention([trajectory[k][v]
                              for k in range(self.steps + 1)], self.s)

    # -- training step -----------------------------------------------------

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    pos_weight: float = 1.0) -> float:
        """One SGD step on ``targets``. Returns mean weighted BCE loss.

        The forward pass embeds every node (a target's propagated
        score depends on its neighbors' encoder states), but the loss
        and its seed gradient cover only ``targets``. Backprop splits
        at the adaptive sum into the direct path and the attention
        path (see the module docstring for ``dl``), unwinds the
        propagation chain into the encoder, then the MLP. ``lr=0.0``
        accumulates gradients without moving the weights, which the
        numeric gradient check relies on.
        """
        self.zero_grad()
        probs, cache = self.forward(graph, targets)
        weights = cache["weights"]
        pre1 = cache["pre1"]
        trajectory = cache["trajectory"]
        attn = cache["attn"]
        z = cache["z"]

        B = len(targets)
        wts = {v: (pos_weight if labels[v] == 1 else 1.0)
               for v in targets}
        loss = sum(wts[v] * _bce(probs[v], labels[v])
                   for v in targets) / B

        # Seed gradient on the final logits, then the head backward.
        d_z = [[0.0] * self.hidden_dim for _ in range(graph.n)]
        for v in targets:
            dl = wts[v] * (probs[v] - labels[v]) / B
            if dl == 0.0:
                continue
            for j in range(self.hidden_dim):
                d_z[v][j] = dl * self.w_out[j]
                self.gw_out[j] += dl * z[v][j]
            self.gb_out += dl

        # Adaptive-sum backward. For each node carrying gradient:
        # dl_k = a_k * (dZ . H^(k) - dZ . Z), step k's hidden state
        # gets a_k * dZ (direct) plus dl_k * s (attention path), and
        # s accumulates sum_k dl_k * H^(k).
        d_h = [[[0.0] * self.hidden_dim for _ in range(graph.n)]
               for _ in range(self.steps + 1)]
        for v in range(graph.n):
            dzv = d_z[v]
            if not any(dzv):
                continue
            dz_dot_z = sum(dzv[j] * z[v][j]
                           for j in range(self.hidden_dim))
            for k in range(self.steps + 1):
                hk = trajectory[k][v]
                dz_dot_hk = sum(dzv[j] * hk[j]
                                for j in range(self.hidden_dim))
                dl_k = attn[v][k] * (dz_dot_hk - dz_dot_z)
                dhk = d_h[k][v]
                ak = attn[v][k]
                for j in range(self.hidden_dim):
                    dhk[j] += ak * dzv[j] + dl_k * self.s[j]
                    self.gs[j] += dl_k * hk[j]

        # Unwind the propagation chain: step k's gradient flows back
        # through S (symmetric, so the reverse pass reuses the same
        # weights) into step k-1's gradient, accumulating as it goes.
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

        # MLP backward, over every node: propagation mixed every
        # node's encoder state into the targets' scores.
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
            self.s[j] -= self.lr * self.gs[j]
        self.b_out -= self.lr * self.gb_out
        return loss

    # -- checkpointing ------------------------------------------------------

    def state_dict(self):
        return {
            "W1": copy.deepcopy(self.W1),
            "b1": copy.deepcopy(self.b1),
            "s": copy.deepcopy(self.s),
            "w_out": copy.deepcopy(self.w_out),
            "b_out": self.b_out,
        }

    def load_state_dict(self, state) -> None:
        self.W1 = copy.deepcopy(state["W1"])
        self.b1 = copy.deepcopy(state["b1"])
        self.s = copy.deepcopy(state["s"])
        self.w_out = copy.deepcopy(state["w_out"])
        self.b_out = state["b_out"]
        self.zero_grad()

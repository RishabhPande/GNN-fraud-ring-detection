"""Frequency-Adaptive GCN (FAGCN, Bo et al., AAAI 2021), from scratch.

Most GCN-style propagation is a low-pass filter: it smooths a node
toward its neighbours. That helps when neighbours agree, and hurts
at a boundary, where a node differs from its neighbours on purpose.
A bridge account between a fraud ring and the background is exactly
that boundary: smoothing erases the difference that identifies it.

FAGCN keeps both frequency channels at every propagation step::

    H^(0)   = relu(W1 @ x + b1)              (MLP encoder, no graph)
    L^(k)   = S H^(k-1)                      (low-pass: neighbour mix)
    D^(k)   = H^(k-1) - L^(k)                (high-pass: disagreement)
    H^(k)   = alpha * L^(k) + beta * D^(k)
    logit   = w_out . H^(K) + b_out

where ``S = D_hat^{-1/2} A_hat D_hat^{-1/2}`` is the symmetric
normalized adjacency with self-loops, the same operator the GCN and
SGC in this repo use. ``alpha`` and ``beta`` are learned scalars,
shared across steps. The sharing is deliberate, not a shortcut:
every step's operator is a polynomial in the same ``S``, polynomials
in one matrix commute, so per-step coefficients started equal receive
identical gradients at every batch and move in exact lockstep - they
are not separately identified. One shared pair is the honest
parameterization. The pair starts at ``alpha = 1, beta = 0``, so
step 1 starts as plain low-pass propagation and the high-pass
channel is opened only if training earns it. A negative ``beta``
is allowed: the model can also learn to subtract disagreement.

Backprop: the gradients for ``alpha`` / ``beta`` are the dot
products of each step's output gradient with ``L^(k)`` / ``D^(k)``,
summed over steps.
The gradient into ``H^(k-1)`` is
``alpha * S^T d + beta * (I - S^T) d``, and ``S`` is symmetric
so ``S^T = S``. The chain is unwound step by step into the encoder,
then through the MLP. Numeric-gradient-checked end to end, including
both learned coefficients off their init. All linear algebra is
hand-rolled lists-of-floats - no numpy, no torch.
"""

from __future__ import annotations

import copy
import math
import random


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _bce(p, y):
    p = min(max(p, 1e-7), 1.0 - 1e-7)
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


def _norm_weights(graph):
    d_hat = [graph.degree(v) + 1 for v in range(graph.n)]
    weights = []
    for v in range(graph.n):
        row = []
        for u in sorted(set(graph.neighbors(v)) | {v}):
            row.append((u, 1.0 / math.sqrt(d_hat[v] * d_hat[u])))
        weights.append(row)
    return weights


def _propagate(graph, H, weights=None):
    """One symmetric-normalized propagation S H."""
    if weights is None:
        weights = _norm_weights(graph)
    width = len(H[0])
    out = []
    for v in range(graph.n):
        acc = [0.0] * width
        for u, c in weights[v]:
            hu = H[u]
            for j in range(width):
                acc[j] += c * hu[j]
        out.append(acc)
    return out


def frequency_split(graph, H):
    """Split H into (low, high): low = S H, high = H - S H.

    The two parts sum exactly to H's low plus... in fact
    low + high == H for every node, which the tests check.
    An isolated node has S row [1.0] on itself, so its high part
    is exactly zero: with no neighbours there is no disagreement.
    """
    low = _propagate(graph, H)
    high = [[h - l for h, l in zip(H[v], low[v])] for v in range(graph.n)]
    return low, high


class FAGCN:
    """FAGCN: MLP encoder, K frequency-adaptive propagation steps,
    and a linear head. See module docstring for the update rule."""

    def __init__(self, in_dim, hidden_dim=16, steps=2, lr=0.25, seed=0):
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if steps < 0:
            raise ValueError("steps must be non-negative")
        rng = random.Random(seed)
        scale = math.sqrt(2.0 / (in_dim + hidden_dim))
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.steps = steps
        self.lr = lr
        self.W1 = [[rng.uniform(-scale, scale) for _ in range(in_dim)]
                   for _ in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim
        # Start as pure low-pass: alpha=1, beta=0, shared per step.
        self.alpha = 1.0
        self.beta = 0.0
        out_scale = math.sqrt(2.0 / (hidden_dim + 1))
        self.w_out = [rng.uniform(-out_scale, out_scale)
                      for _ in range(hidden_dim)]
        self.b_out = 0.0
        self.zero_grad()

    def zero_grad(self):
        self.gW1 = [[0.0] * self.in_dim for _ in range(self.hidden_dim)]
        self.gb1 = [0.0] * self.hidden_dim
        self.galpha = 0.0
        self.gbeta = 0.0
        self.gw_out = [0.0] * self.hidden_dim
        self.gb_out = 0.0

    def forward(self, graph, targets):
        weights = _norm_weights(graph)
        pre1, h0 = [], []
        for v in range(graph.n):
            x = graph.features[v]
            pre = [sum(w * xi for w, xi in zip(row, x)) + b
                   for row, b in zip(self.W1, self.b1)]
            pre1.append(pre)
            h0.append([p if p > 0.0 else 0.0 for p in pre])
        trajectory = [[list(r) for r in h0]]
        splits = []
        H = h0
        for k in range(self.steps):
            low = _propagate(graph, H, weights)
            high = [[H[v][j] - low[v][j] for j in range(self.hidden_dim)]
                    for v in range(graph.n)]
            splits.append((low, high))
            a, b = self.alpha, self.beta
            H = [[a * low[v][j] + b * high[v][j]
                  for j in range(self.hidden_dim)] for v in range(graph.n)]
            trajectory.append([list(r) for r in H])
        probs = {}
        for v in targets:
            logit = sum(w * zj for w, zj in zip(self.w_out, H[v])) + self.b_out
            probs[v] = _sigmoid(logit)
        cache = {"weights": weights, "pre1": pre1, "h0": h0,
                 "trajectory": trajectory, "splits": splits,
                 "z": H, "targets": list(targets)}
        return probs, cache

    def predict(self, graph, targets):
        probs, _ = self.forward(graph, targets)
        return probs

    def train_batch(self, graph, targets, labels, pos_weight=1.0):
        self.zero_grad()
        probs, cache = self.forward(graph, targets)
        weights = cache["weights"]
        pre1 = cache["pre1"]
        trajectory = cache["trajectory"]
        splits = cache["splits"]
        z = cache["z"]
        B = len(targets)
        wts = {v: (pos_weight if labels[v] == 1 else 1.0) for v in targets}
        loss = sum(wts[v] * _bce(probs[v], labels[v]) for v in targets) / B

        d_z = [[0.0] * self.hidden_dim for _ in range(graph.n)]
        for v in targets:
            dl = wts[v] * (probs[v] - labels[v]) / B
            for j in range(self.hidden_dim):
                d_z[v][j] = dl * self.w_out[j]
                self.gw_out[j] += dl * z[v][j]
            self.gb_out += dl

        # Unwind propagation steps in reverse.
        d_H = d_z
        for k in range(self.steps - 1, -1, -1):
            low, high = splits[k]
            a, b = self.alpha, self.beta
            for v in range(graph.n):
                dv = d_H[v]
                if not any(dv):
                    continue
                for j in range(self.hidden_dim):
                    self.galpha += dv[j] * low[v][j]
                    self.gbeta += dv[j] * high[v][j]
            # Gradient wrt L is a*d, wrt D is b*d.
            # H_prev gradient = S^T (a*d) + (I - S^T)(b*d)
            #                  = b*d + S^T ((a-b)*d)   [S symmetric]
            coeff = [[(a - b) * x for x in row] for row in d_H]
            prop = _propagate(graph, coeff, weights)
            d_prev = [[b * d_H[v][j] + prop[v][j]
                       for j in range(self.hidden_dim)]
                      for v in range(graph.n)]
            d_H = d_prev

        d_h0 = d_H
        for v in range(graph.n):
            dh = d_h0[v]
            if not any(dh):
                continue
            x = graph.features[v]
            for j in range(self.hidden_dim):
                d_pre = dh[j] * (1.0 if pre1[v][j] > 0.0 else 0.0)
                if d_pre == 0.0:
                    continue
                for t in range(self.in_dim):
                    self.gW1[j][t] += d_pre * x[t]
                self.gb1[j] += d_pre

        for j in range(self.hidden_dim):
            for t in range(self.in_dim):
                self.W1[j][t] -= self.lr * self.gW1[j][t]
            self.b1[j] -= self.lr * self.gb1[j]
            self.w_out[j] -= self.lr * self.gw_out[j]
        self.alpha -= self.lr * self.galpha
        self.beta -= self.lr * self.gbeta
        self.b_out -= self.lr * self.gb_out
        return loss

    def state_dict(self):
        return {"W1": copy.deepcopy(self.W1), "b1": copy.deepcopy(self.b1),
                "alpha": self.alpha,
                "beta": self.beta,
                "w_out": copy.deepcopy(self.w_out), "b_out": self.b_out}

    def load_state_dict(self, state):
        self.W1 = copy.deepcopy(state["W1"])
        self.b1 = copy.deepcopy(state["b1"])
        self.alpha = state["alpha"]
        self.beta = state["beta"]
        self.w_out = copy.deepcopy(state["w_out"])
        self.b_out = state["b_out"]
        self.zero_grad()

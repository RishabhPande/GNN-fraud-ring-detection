"""LEConv-style difference convolution (Ranjan et al., 2020),
from scratch.

Every other aggregation in this repo summarises a node's
neighbours: a mean, a sum, a max, an attention-weighted mix. The
local-extrema idea behind LEConv summarises instead how the node
*differ* from its neighbours::

    diff_v = sum_{u in N(v)} (h_v - h_u) = deg(v) * h_v - sum_{u} h_u
    h'_v   = relu(W_self @ h_v + W_diff @ diff_v + b)

``diff_v`` is the (unnormalised) graph Laplacian applied to the
embeddings, ``(L H)_v``: it is exactly zero for a node whose
neighbours all look like it, and large, with a sign per dimension,
for a node sitting in a neighbourhood unlike itself. That is the
fraud-relevant direction the averaging models cannot express
directly: a bridge account between the ring and the background,
or a ring member whose camouflaged profile matches the background
while its neighbours are other ring members, is a local extremum
even when its neighbourhood mean looks ordinary. Inside a
homogeneous pocket the difference cancels, so the layer also
keeps a plain self term - the extremum signal complements the
identity signal, it does not replace it.

This repo's version keeps the paper's difference operator and
drops its per-edge learned gate, so the layer stays linear in the
embeddings (plus the ReLU) and the backprop is exact and simple:
in reverse, ``d_diff`` flows ``deg(v)`` times into the node itself
and ``-1`` times into each neighbour. An isolated node has
``diff = 0`` exactly and degrades to a linear-plus-ReLU layer over
its own features. Full backprop is numeric-gradient-checked at
the layer (including the input gradient) and end to end. All
linear algebra is hand-rolled lists-of-floats - no numpy, no
torch.
"""

from __future__ import annotations

import copy
import math
import random


def _matvec(W, x):
    return [sum(w * xi for w, xi in zip(row, x)) for row in W]


def _mat_t_vec(W, v):
    n_cols = len(W[0])
    return [sum(W[r][c] * v[r] for r in range(len(W))) for c in range(n_cols)]


def _relu(v):
    return [x if x > 0.0 else 0.0 for x in v]


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _bce(p, y):
    p = min(max(p, 1e-7), 1.0 - 1e-7)
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


def difference_sums(graph, H):
    """The difference aggregation ``deg(v) * h_v - sum_{u in N(v)}
    h_u`` for every node: the unnormalised Laplacian ``L H``.
    Zero exactly for an isolated node and for a node whose
    neighbours' embeddings sum to ``deg(v)`` times its own."""
    diffs = []
    for v in range(graph.n):
        nbrs = graph.neighbors(v)
        hv = H[v]
        acc = [len(nbrs) * x for x in hv]
        for u in nbrs:
            hu = H[u]
            for j in range(len(hv)):
                acc[j] -= hu[j]
        diffs.append(acc)
    return diffs


class LEConvLayer:
    """Single difference-convolution layer:
    ``relu(W_self @ h_v + W_diff @ diff_v + b)`` with ``diff_v``
    from :func:`difference_sums`. Operates on the whole graph at
    once, so ``forward`` takes the full embedding list."""

    def __init__(self, in_dim, out_dim, rng):
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W_self = [
            [rng.uniform(-scale, scale) for _ in range(in_dim)]
            for _ in range(out_dim)
        ]
        self.W_diff = [
            [rng.uniform(-scale, scale) for _ in range(in_dim)]
            for _ in range(out_dim)
        ]
        self.b = [0.0] * out_dim
        self.zero_grad()

    def zero_grad(self):
        self.gW_self = [[0.0] * self.in_dim for _ in range(self.out_dim)]
        self.gW_diff = [[0.0] * self.in_dim for _ in range(self.out_dim)]
        self.gb = [0.0] * self.out_dim

    def forward(self, graph, H):
        """Returns (H_out, cache): cache holds (H_in, differences,
        pre-activations, neighbour lists)."""
        diffs = difference_sums(graph, H)
        pre = []
        for v in range(graph.n):
            s = _matvec(self.W_self, H[v])
            t = _matvec(self.W_diff, diffs[v])
            pre.append([si + ti + bi for si, ti, bi in zip(s, t, self.b)])
        nbrs = [graph.neighbors(v) for v in range(graph.n)]
        return [_relu(p) for p in pre], (H, diffs, pre, nbrs)

    def backward(self, d_out, cache):
        """Backprop one layer. Returns d_H, the gradient w.r.t. the
        input embeddings. The difference operator is symmetric:
        ``d_diff`` contributes ``deg(v) * d_diff`` to the node
        itself and ``-d_diff`` to each of its neighbours."""
        H, diffs, pre, nbrs = cache
        n = len(diffs)
        d_pre = [
            [d * (1.0 if p > 0.0 else 0.0) for d, p in zip(d_out[v], pre[v])]
            for v in range(n)
        ]
        d_H = [[0.0] * self.in_dim for _ in range(n)]
        for v in range(n):
            dv = d_pre[v]
            if not any(dv):
                continue
            for i in range(self.out_dim):
                di = dv[i]
                if di == 0.0:
                    continue
                for j in range(self.in_dim):
                    self.gW_self[i][j] += di * H[v][j]
                    self.gW_diff[i][j] += di * diffs[v][j]
                self.gb[i] += di
            d_self = _mat_t_vec(self.W_self, dv)
            d_diff = _mat_t_vec(self.W_diff, dv)
            deg = len(nbrs[v])
            for j in range(self.in_dim):
                d_H[v][j] += d_self[j] + deg * d_diff[j]
            for u in nbrs[v]:
                row = d_H[u]
                for j in range(self.in_dim):
                    row[j] -= d_diff[j]
        return d_H

    def step(self, lr):
        for i in range(self.out_dim):
            for j in range(self.in_dim):
                self.W_self[i][j] -= lr * self.gW_self[i][j]
                self.W_diff[i][j] -= lr * self.gW_diff[i][j]
            self.b[i] -= lr * self.gb[i]

    def state_dict(self):
        return {
            "W_self": copy.deepcopy(self.W_self),
            "W_diff": copy.deepcopy(self.W_diff),
            "b": copy.deepcopy(self.b),
        }

    def load_state_dict(self, state):
        self.W_self = copy.deepcopy(state["W_self"])
        self.W_diff = copy.deepcopy(state["W_diff"])
        self.b = copy.deepcopy(state["b"])
        self.zero_grad()


class GraphLEConv:
    """Two-layer LEConv for binary node classification.

    Full-graph forward (no sampling: the difference needs true
    degrees), linear plus sigmoid output head, weighted BCE loss
    like :class:`GraphSAGE`.
    """

    def __init__(self, in_dim, hidden_dim=16, lr=0.5, seed=0):
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.rng = random.Random(seed)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.layer1 = LEConvLayer(in_dim, hidden_dim, self.rng)
        self.layer2 = LEConvLayer(hidden_dim, hidden_dim, self.rng)
        scale = math.sqrt(2.0 / (hidden_dim + 1))
        self.w_out = [self.rng.uniform(-scale, scale)
                      for _ in range(hidden_dim)]
        self.b_out = 0.0
        self.gw_out = [0.0] * hidden_dim
        self.gb_out = 0.0

    def forward(self, graph, targets):
        h1, cache1 = self.layer1.forward(graph, graph.features)
        h2, cache2 = self.layer2.forward(graph, h1)
        probs = {}
        for v in targets:
            z = sum(w * h for w, h in zip(self.w_out, h2[v])) + self.b_out
            probs[v] = _sigmoid(z)
        cache = {"h1": h1, "cache1": cache1, "h2": h2,
                 "cache2": cache2, "targets": list(targets)}
        return probs, cache

    def predict(self, graph, targets):
        probs, _ = self.forward(graph, targets)
        return probs

    def train_batch(self, graph, targets, labels, pos_weight=1.0):
        """One full-batch SGD step scored on ``targets``. Returns
        mean weighted BCE loss."""
        self.layer1.zero_grad()
        self.layer2.zero_grad()
        self.gw_out = [0.0] * self.hidden_dim
        self.gb_out = 0.0
        probs, cache = self.forward(graph, targets)
        h2 = cache["h2"]
        B = len(targets)
        weights = {v: (pos_weight if labels[v] == 1 else 1.0)
                   for v in targets}
        loss = sum(weights[v] * _bce(probs[v], labels[v])
                   for v in targets) / B
        dh2 = [[0.0] * self.hidden_dim for _ in range(graph.n)]
        for v in targets:
            d_logit = weights[v] * (probs[v] - labels[v]) / B
            for j in range(self.hidden_dim):
                self.gw_out[j] += d_logit * h2[v][j]
                dh2[v][j] = d_logit * self.w_out[j]
            self.gb_out += d_logit
        dh1 = self.layer2.backward(dh2, cache["cache2"])
        self.layer1.backward(dh1, cache["cache1"])
        self.layer1.step(self.lr)
        self.layer2.step(self.lr)
        for j in range(self.hidden_dim):
            self.w_out[j] -= self.lr * self.gw_out[j]
        self.b_out -= self.lr * self.gb_out
        return loss

    def state_dict(self):
        return {
            "layer1": self.layer1.state_dict(),
            "layer2": self.layer2.state_dict(),
            "w_out": copy.deepcopy(self.w_out),
            "b_out": self.b_out,
        }

    def load_state_dict(self, state):
        self.layer1.load_state_dict(state["layer1"])
        self.layer2.load_state_dict(state["layer2"])
        self.w_out = copy.deepcopy(state["w_out"])
        self.b_out = state["b_out"]

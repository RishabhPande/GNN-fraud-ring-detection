"""FiLM-conditioned graph network (FiLM-GNN, Brockschmidt 2020,
building on FiLM, Perez et al. 2018), from scratch.

Every other model in this repo combines a node's own representation
with its neighborhood through a fixed rule: add them, concatenate
them, sum them, gate them by a shared recurrence. FiLM makes the
combination itself depend on the node. Each layer generates, from
the target node's own input, a per-dimension scale and shift for
the neighbor message::

    gamma_v = W_gamma @ x_v + b_gamma     (scale, before the +1)
    beta_v  = W_beta  @ x_v + b_beta      (shift)
    msg_v   = mean_{u in N(v)} (W_neigh @ x_u)
    h'_v    = relu(W_self @ x_v + (1 + gamma_v) * msg_v + beta_v + b)

Because gamma_v and beta_v depend only on the target, they factor
out of the mean exactly, so the layer costs one neighbor transform
per node, like GraphSAGE mean. What the factoring does not remove
is the interaction: the product (1 + gamma_v) * msg_v is bilinear
in the node's own features and its neighborhood mean, a second-order
term no linear layer over either input alone can express. A node
whose own profile says "watch the balance dimension" can turn that
dimension of the incoming message up and the others down; a plain
SAGE layer applies the same neighbor weights to every account.

The +1 on the scale is the standard residual FiLM trick: with the
FiLM weights at zero the layer is exactly a GraphSAGE mean layer
(self transform plus mean neighbor transform), so training starts
from that model and learns departures from it. The FiLM matrices
are also initialized at a tenth of the main scale, so the starting
point is near, not exactly at, the unmodulated model.

Fraud reading: camouflaged ring members share one profile, so their
generated scales should agree, while a bridge account (ring on one
side, background on the other) carries a different profile and can
learn to damp the ring side of its message. Whether the generator
rewards that is an empirical question the example answers honestly.

Backprop runs through the bilinear term both ways: the scale's
gradient is the message times the output gradient, the message's
gradient is the scale times the output gradient, and both FiLM
matrices take outer-product gradients against the node's own input
(numeric-gradient-checked at the layer and end to end, including
the isolated-node case, where the message is defined as zero and
no gradient flows through the FiLM branch at all). All linear
algebra is hand-rolled lists-of-floats, no numpy, no torch.
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


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _bce(p: float, y: int) -> float:
    p = min(max(p, 1e-7), 1.0 - 1e-7)
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


class FiLMLayer:
    """Single FiLM layer: self transform plus a neighbor mean whose
    every dimension is scaled and shifted by parameters generated
    from the target node's own input. Operates on the whole graph
    at once, so ``forward`` takes the full embedding list indexed
    by node id."""

    def __init__(self, in_dim: int, out_dim: int, rng: random.Random):
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("in_dim and out_dim must be positive")
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        film_scale = 0.1 * scale
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W_self = [
            [rng.uniform(-scale, scale) for _ in range(in_dim)]
            for _ in range(out_dim)
        ]
        self.W_neigh = [
            [rng.uniform(-scale, scale) for _ in range(in_dim)]
            for _ in range(out_dim)
        ]
        self.W_gamma = [
            [rng.uniform(-film_scale, film_scale) for _ in range(in_dim)]
            for _ in range(out_dim)
        ]
        self.b_gamma = [0.0] * out_dim
        self.W_beta = [
            [rng.uniform(-film_scale, film_scale) for _ in range(in_dim)]
            for _ in range(out_dim)
        ]
        self.b_beta = [0.0] * out_dim
        self.b = [0.0] * out_dim
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW_self = [[0.0] * self.in_dim for _ in range(self.out_dim)]
        self.gW_neigh = [[0.0] * self.in_dim for _ in range(self.out_dim)]
        self.gW_gamma = [[0.0] * self.in_dim for _ in range(self.out_dim)]
        self.gb_gamma = [0.0] * self.out_dim
        self.gW_beta = [[0.0] * self.in_dim for _ in range(self.out_dim)]
        self.gb_beta = [0.0] * self.out_dim
        self.gb = [0.0] * self.out_dim

    def modulation(self, x: list[float]) -> tuple[list[float], list[float]]:
        """The (scale, shift) this layer would apply to a neighbor
        message arriving at a node whose input is ``x``. The scale
        includes the residual +1, so a zero FiLM branch returns a
        scale of all ones and a shift of all zeros."""
        gamma = _matvec(self.W_gamma, x)
        beta = _matvec(self.W_beta, x)
        scale = [1.0 + g + bg for g, bg in zip(gamma, self.b_gamma)]
        shift = [b + bb for b, bb in zip(beta, self.b_beta)]
        return scale, shift

    def forward(self, graph, H: list[list[float]]):
        """Returns (H_out, cache). The cache holds the inputs, the
        transformed neighbor messages, their per-node means, the
        per-node scales and shifts, and the pre-activations."""
        msgs = [_matvec(self.W_neigh, h) for h in H]
        nbrs = [graph.neighbors(v) for v in range(graph.n)]
        mean_msg = []
        scales = []
        shifts = []
        pre = []
        out = []
        for v in range(graph.n):
            scale, shift = self.modulation(H[v])
            scales.append(scale)
            shifts.append(shift)
            if nbrs[v]:
                acc = [0.0] * self.out_dim
                for u in nbrs[v]:
                    mu = msgs[u]
                    for j in range(self.out_dim):
                        acc[j] += mu[j]
                mean = [a / len(nbrs[v]) for a in acc]
            else:
                mean = [0.0] * self.out_dim
            mean_msg.append(mean)
            self_part = _matvec(self.W_self, H[v])
            p = []
            for j in range(self.out_dim):
                agg = (scale[j] * mean[j] + shift[j]) if nbrs[v] else 0.0
                p.append(self_part[j] + agg + self.b[j])
            pre.append(p)
            out.append([x if x > 0.0 else 0.0 for x in p])
        cache = (H, msgs, mean_msg, scales, shifts, pre, nbrs)
        return out, cache

    def backward(self, d_out: list[list[float]], cache) -> list[list[float]]:
        """Backprop one layer. Returns d_H, the gradient w.r.t. the
        input embeddings. The bilinear term splits both ways: the
        scale takes the message times the output gradient, and the
        message takes the scale times the output gradient, divided
        by the degree on its way back to each neighbor."""
        H, msgs, mean_msg, scales, shifts, pre, nbrs = cache
        n = len(H)
        d_pre = [
            [d * (1.0 if p > 0.0 else 0.0) for d, p in zip(d_out[v], pre[v])]
            for v in range(n)
        ]
        d_H = [[0.0] * self.in_dim for _ in range(n)]
        d_msg = [[0.0] * self.out_dim for _ in range(n)]
        for v in range(n):
            dv = d_pre[v]
            if not any(dv):
                continue
            x = H[v]
            for i in range(self.out_dim):
                di = dv[i]
                if di == 0.0:
                    continue
                for j in range(self.in_dim):
                    self.gW_self[i][j] += di * x[j]
                self.gb[i] += di
            dh = _mat_t_vec(self.W_self, dv)
            for j in range(self.in_dim):
                d_H[v][j] += dh[j]
            if not nbrs[v]:
                continue
            # agg_j = scale_j * mean_j + shift_j, so:
            # d scale_j = dv_j * mean_j, d shift_j = dv_j,
            # d mean_j = dv_j * scale_j.
            d_scale = [dv[j] * mean_msg[v][j] for j in range(self.out_dim)]
            d_mean = [dv[j] * scales[v][j] for j in range(self.out_dim)]
            for i in range(self.out_dim):
                ds = d_scale[i]
                if ds != 0.0:
                    for j in range(self.in_dim):
                        self.gW_gamma[i][j] += ds * x[j]
                    self.gb_gamma[i] += ds
                if dv[i] != 0.0:
                    for j in range(self.in_dim):
                        self.gW_beta[i][j] += dv[i] * x[j]
                    self.gb_beta[i] += dv[i]
            dg = _mat_t_vec(self.W_gamma, d_scale)
            db = _mat_t_vec(self.W_beta, dv)
            for j in range(self.in_dim):
                d_H[v][j] += dg[j] + db[j]
            share = 1.0 / len(nbrs[v])
            for u in nbrs[v]:
                row = d_msg[u]
                for j in range(self.out_dim):
                    row[j] += share * d_mean[j]
        for u in range(n):
            dm = d_msg[u]
            if not any(dm):
                continue
            for i in range(self.out_dim):
                di = dm[i]
                if di == 0.0:
                    continue
                for j in range(self.in_dim):
                    self.gW_neigh[i][j] += di * H[u][j]
            dh = _mat_t_vec(self.W_neigh, dm)
            for j in range(self.in_dim):
                d_H[u][j] += dh[j]
        return d_H

    def step(self, lr: float) -> None:
        for W, gW in (
            (self.W_self, self.gW_self),
            (self.W_neigh, self.gW_neigh),
            (self.W_gamma, self.gW_gamma),
            (self.W_beta, self.gW_beta),
        ):
            for i in range(self.out_dim):
                for j in range(self.in_dim):
                    W[i][j] -= lr * gW[i][j]
        for b, gb in (
            (self.b_gamma, self.gb_gamma),
            (self.b_beta, self.gb_beta),
            (self.b, self.gb),
        ):
            for i in range(self.out_dim):
                b[i] -= lr * gb[i]

    def state_dict(self):
        return {
            "W_self": copy.deepcopy(self.W_self),
            "W_neigh": copy.deepcopy(self.W_neigh),
            "W_gamma": copy.deepcopy(self.W_gamma),
            "b_gamma": copy.deepcopy(self.b_gamma),
            "W_beta": copy.deepcopy(self.W_beta),
            "b_beta": copy.deepcopy(self.b_beta),
            "b": copy.deepcopy(self.b),
        }

    def load_state_dict(self, state) -> None:
        self.W_self = copy.deepcopy(state["W_self"])
        self.W_neigh = copy.deepcopy(state["W_neigh"])
        self.W_gamma = copy.deepcopy(state["W_gamma"])
        self.b_gamma = copy.deepcopy(state["b_gamma"])
        self.W_beta = copy.deepcopy(state["W_beta"])
        self.b_beta = copy.deepcopy(state["b_beta"])
        self.b = copy.deepcopy(state["b"])
        self.zero_grad()


class GraphFiLM:
    """Two-layer FiLM-GNN for binary node classification.

    Full-graph forward (the mean needs true neighborhoods), linear
    plus sigmoid output head, weighted BCE loss like
    :class:`GraphSAGE`. ``modulation(graph, v)`` reports the first
    layer's (scale, shift) for a node, so a trained model's
    conditioning can be read directly: which message dimensions a
    node's own profile turns up, down, or off.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 16,
        lr: float = 0.1,
        seed: int = 0,
    ):
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.rng = random.Random(seed)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.layer1 = FiLMLayer(in_dim, hidden_dim, self.rng)
        self.layer2 = FiLMLayer(hidden_dim, hidden_dim, self.rng)
        scale = math.sqrt(2.0 / (hidden_dim + 1))
        self.w_out = [self.rng.uniform(-scale, scale)
                      for _ in range(hidden_dim)]
        self.b_out = 0.0
        self.gw_out = [0.0] * hidden_dim
        self.gb_out = 0.0

    def forward(self, graph, targets: list[int]):
        """Forward pass over the whole graph. Returns (probs dict,
        cache) with probs only for the requested targets."""
        h1, cache1 = self.layer1.forward(graph, graph.features)
        h2, cache2 = self.layer2.forward(graph, h1)
        probs: dict[int, float] = {}
        for v in targets:
            z = sum(w * h for w, h in zip(self.w_out, h2[v])) + self.b_out
            probs[v] = _sigmoid(z)
        cache = {"h1": h1, "cache1": cache1, "h2": h2,
                 "cache2": cache2, "targets": list(targets)}
        return probs, cache

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        """Inference, deterministic, no sampling anywhere."""
        probs, _ = self.forward(graph, targets)
        return probs

    def modulation(self, graph, v: int) -> tuple[list[float], list[float]]:
        """First-layer (scale, shift) generated from node ``v``'s raw
        features: how this account's own profile conditions the
        message it accepts from its counterparties."""
        return self.layer1.modulation(graph.features[v])

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    pos_weight: float = 1.0) -> float:
        """One full-graph SGD step scored on ``targets``. Returns mean
        weighted BCE loss. ``lr=0.0`` on the model accumulates
        gradients without moving the weights, which the numeric
        gradient check relies on."""
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
            self.gb_out += d_logit
            dh2[v] = [d_logit * w for w in self.w_out]

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

    def load_state_dict(self, state) -> None:
        self.layer1.load_state_dict(state["layer1"])
        self.layer2.load_state_dict(state["layer2"])
        self.w_out = copy.deepcopy(state["w_out"])
        self.b_out = state["b_out"]

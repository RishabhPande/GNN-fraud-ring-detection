"""Attention-based Graph Neural Network (AGNN, Klicpera et al.,
2019), from scratch.

The propagation-polynomial models in this repo (APPNP, GPR-GNN,
DAGNN) all propagate hidden states with the same fixed normalized
adjacency for every edge. AGNN asks what happens if the propagation
itself listens: before each propagation step, every edge is
re-weighted by how similar its two endpoints currently look::

    H^(0)_v = relu(W1 @ x_v + b1)            (MLP encoder, no graph)
    cos_{vu} = (h_v . h_u) / (||h_v|| ||h_u||)
    a_{vu}   = softmax_{u in N(v) u {v}}(beta * cos_{vu})
    H'_v     = sum_u a_{vu} h_u
    logit    = w_out . H^(K)_v + b_out

``beta`` is one learnable scalar per propagation step, shared by
every edge: it is the temperature of the listening. At beta = 0
every weight is exactly uniform and the step is a plain mean over
the neighborhood plus the node itself. As beta grows the step
concentrates on the neighbors most similar to the target, and beta
is trained by gradient descent like any other parameter, so the
model itself decides how selective the propagation should be.

Two guards keep the cosine well defined: a pair involving a zero
vector gets cosine 0 and no cosine gradient (there is no direction
to compare), and a node paired with itself has cosine exactly 1
with zero derivative, as the formula gives.

Backprop runs through the softmax (the usual Jacobian term), the
cosine (gradients into both endpoints, including the norm terms),
the direct weighted-sum path into each neighbor, and the encoder.
All of it is numeric-gradient-checked end to end, including beta
with beta set off its init, where the attention path is live. All
linear algebra is hand-rolled lists-of-floats -- no numpy, no
torch.
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


def _cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def attention_weights_for(H: list[list[float]], nbrs: list[int],
                          v: int, beta: float) -> list[float]:
    """AGNN edge weights for node ``v``: softmax over ``nbrs`` (which
    should include ``v`` itself) of ``beta * cos(h_v, h_u)``. The
    returned weights align with ``nbrs`` and sum to 1; at beta = 0
    they are exactly uniform."""
    if not nbrs:
        raise ValueError("nbrs must be non-empty")
    scores = [beta * _cosine(H[v], H[u]) for u in nbrs]
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def propagate_attention(graph, H: list[list[float]], beta: float):
    """One AGNN propagation step over the whole graph. Returns
    (H_out, cache); the cache holds the neighbor lists (each
    including the node itself), the cosines, and the attention
    weights backprop needs."""
    nbr_lists = []
    cos_lists = []
    attn_lists = []
    H_out = []
    dim = len(H[0]) if H else 0
    for v in range(graph.n):
        nbrs = sorted(set(graph.neighbors(v)) | {v})
        cosines = [_cosine(H[v], H[u]) for u in nbrs]
        scores = [beta * c for c in cosines]
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        total = sum(exps)
        attn = [e / total for e in exps]
        acc = [0.0] * dim
        for a, u in zip(attn, nbrs):
            hu = H[u]
            for j in range(dim):
                acc[j] += a * hu[j]
        nbr_lists.append(nbrs)
        cos_lists.append(cosines)
        attn_lists.append(attn)
        H_out.append(acc)
    cache = {"nbrs": nbr_lists, "cos": cos_lists, "attn": attn_lists,
             "H_in": H, "beta": beta}
    return H_out, cache


def _propagate_attention_backward(d_out, cache):
    """Backprop one AGNN propagation step. Returns (d_H, d_beta)."""
    H = cache["H_in"]
    beta = cache["beta"]
    n = len(H)
    dim = len(H[0]) if H else 0
    d_H = [[0.0] * dim for _ in range(n)]
    d_beta = 0.0
    for v in range(n):
        dv = d_out[v]
        if not any(dv):
            continue
        nbrs = cache["nbrs"][v]
        cosines = cache["cos"][v]
        attn = cache["attn"][v]
        hv = H[v]
        nv = math.sqrt(sum(x * x for x in hv))
        # Gradient into the attention weights, then the softmax
        # Jacobian: d_score_i = a_i * (g_i - sum_j a_j g_j).
        g = [sum(dv[j] * H[u][j] for j in range(dim)) for u in nbrs]
        g_mean = sum(a * gi for a, gi in zip(attn, g))
        d_scores = [a * (gi - g_mean) for a, gi in zip(attn, g)]
        for idx, u in enumerate(nbrs):
            a = attn[idx]
            # Direct weighted-sum path into the neighbor.
            du = d_H[u]
            for j in range(dim):
                du[j] += a * dv[j]
            # Attention path: score = beta * cos, so beta accumulates
            # d_score * cos and the cosine gets beta * d_score.
            d_beta += d_scores[idx] * cosines[idx]
            d_cos = beta * d_scores[idx]
            if d_cos == 0.0:
                continue
            hu = H[u]
            nu = math.sqrt(sum(x * x for x in hu))
            if nv == 0.0 or nu == 0.0:
                continue
            c = cosines[idx]
            denom = nv * nu
            for j in range(dim):
                d_H[v][j] += d_cos * (hu[j] / denom
                                      - c * hv[j] / (nv * nv))
                d_H[u][j] += d_cos * (hv[j] / denom
                                      - c * hu[j] / (nu * nu))
    return d_H, d_beta


class AGNN:
    """AGNN: an MLP encoder, K rounds of cosine-attention
    propagation of the hidden states with one learnable temperature
    ``beta`` per round, and a linear head.

    ``predict`` takes a graph and target node ids and returns
    ``{v: probability}``, matching the other models' interface.
    ``attention_weights(graph, v)`` returns node ``v``'s first-step
    edge weights, aligned with the sorted neighbor-plus-self list.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 16,
        steps: int = 2,
        lr: float = 0.5,
        seed: int = 0,
        beta_init: float = 1.0,
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
        self.betas = [float(beta_init)] * steps
        out_scale = math.sqrt(2.0 / (hidden_dim + 1))
        self.w_out = [rng.uniform(-out_scale, out_scale)
                      for _ in range(hidden_dim)]
        self.b_out = 0.0
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW1 = [[0.0] * self.in_dim for _ in range(self.hidden_dim)]
        self.gb1 = [0.0] * self.hidden_dim
        self.gbetas = [0.0] * self.steps
        self.gw_out = [0.0] * self.hidden_dim
        self.gb_out = 0.0

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
        cache) with probs only for the requested targets."""
        pre1, h0 = self._encode(graph)
        trajectory = [h0]
        caches = []
        H = h0
        for k in range(self.steps):
            H, cache = propagate_attention(graph, H, self.betas[k])
            trajectory.append(H)
            caches.append(cache)
        probs: dict[int, float] = {}
        for v in targets:
            logit = (sum(w * hj for w, hj in zip(self.w_out, H[v]))
                     + self.b_out)
            probs[v] = _sigmoid(logit)
        cache = {"pre1": pre1, "trajectory": trajectory,
                 "step_caches": caches, "targets": list(targets)}
        return probs, cache

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        probs, _ = self.forward(graph, targets)
        return probs

    def attention_weights(self, graph, v: int) -> list[float]:
        """Node ``v``'s first-step edge weights under the current
        model, aligned with the sorted neighbor-plus-self list."""
        _, h0 = self._encode(graph)
        nbrs = sorted(set(graph.neighbors(v)) | {v})
        return attention_weights_for(h0, nbrs, v, self.betas[0])

    def train_batch(self, graph, targets: list[int],
                    labels: dict[int, int],
                    pos_weight: float = 1.0) -> float:
        """One SGD step on ``targets``. Returns mean weighted BCE
        loss. ``lr=0.0`` accumulates gradients without moving the
        weights, which the numeric gradient check relies on."""
        self.zero_grad()
        probs, cache = self.forward(graph, targets)
        pre1 = cache["pre1"]
        trajectory = cache["trajectory"]
        step_caches = cache["step_caches"]
        H_final = trajectory[-1]

        B = len(targets)
        wts = {v: (pos_weight if labels[v] == 1 else 1.0)
               for v in targets}
        loss = sum(wts[v] * _bce(probs[v], labels[v])
                   for v in targets) / B

        d_H = [[0.0] * self.hidden_dim for _ in range(graph.n)]
        for v in targets:
            dl = wts[v] * (probs[v] - labels[v]) / B
            if dl == 0.0:
                continue
            for j in range(self.hidden_dim):
                d_H[v][j] = dl * self.w_out[j]
                self.gw_out[j] += dl * H_final[v][j]
            self.gb_out += dl

        for k in range(self.steps - 1, -1, -1):
            d_H, d_beta = _propagate_attention_backward(
                d_H, step_caches[k])
            self.gbetas[k] = d_beta

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
        for k in range(self.steps):
            self.betas[k] -= self.lr * self.gbetas[k]
        self.b_out -= self.lr * self.gb_out
        return loss

    def state_dict(self):
        return {
            "W1": copy.deepcopy(self.W1),
            "b1": copy.deepcopy(self.b1),
            "betas": copy.deepcopy(self.betas),
            "w_out": copy.deepcopy(self.w_out),
            "b_out": self.b_out,
        }

    def load_state_dict(self, state) -> None:
        self.W1 = copy.deepcopy(state["W1"])
        self.b1 = copy.deepcopy(state["b1"])
        self.betas = copy.deepcopy(state["betas"])
        self.w_out = copy.deepcopy(state["w_out"])
        self.b_out = state["b_out"]
        self.zero_grad()

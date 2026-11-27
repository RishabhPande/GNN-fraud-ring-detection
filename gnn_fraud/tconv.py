"""TransformerConv (Shi et al., 2020), from scratch - the
dot-product attention member of this repo's attention family.

GAT scores a pair with a learned vector over the concatenation
of the two transformed states, and GATv2 moves the nonlinearity
so the ranking can depend on the query. TransformerConv scores
a pair the way a Transformer does, with separate query, key,
and value projections and a scaled dot product::

    q_v = W_q @ h_v,  k_u = W_k @ h_u,  val_u = W_v @ h_u + b_v
    score(v, u) = (q_v . k_u) / sqrt(d)      u in N(v) U {v}
    alpha(v, u) = softmax_u score(v, u)
    out_v = sum_u alpha(v, u) * val_u

Two layers: the first maps features to ``hidden_dim`` under a
ReLU, the second maps to a single logit. The self term is not
a separate root weight (PyG adds one); here the node attends to
itself through the same query/key/value path as any neighbour,
so an isolated node's attention is exactly 1.0 on itself and
its output is exactly its own value vector.

Why the distinction matters on this generator: the additive
scorers in this repo (GAT, GATv2, AGNN in its cosine form) all
read a pair through a shared bottleneck, while the dot product
lets every hidden dimension vote separately on whether two
accounts match, and the 1 / sqrt(d) scale keeps those votes
from saturating the softmax as the width grows. Whether that
buys ranking on camouflaged profiles is an empirical question
the example answers on the seed-7 dense ring; the attention
introspection helper (``attention_weights``) reports where the
trained first layer actually puts its weight, ring to ring
versus background to ring, so the mechanism claim does not
have to rest on the headline AUC alone.

Full backprop through the softmax and the dot product into
both endpoints of every attended pair (numeric-gradient-
checked end to end, including the query and key matrices).
All linear algebra is hand-rolled lists-of-floats - no numpy,
no torch.
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


def _attend_from(cache, v: int) -> list[tuple[int, float]]:
    return list(zip(cache["nbrs"][v], cache["alpha"][v]))


class TransformerConvLayer:
    """One scaled dot-product attention layer.

    ``forward(H)`` takes a list of per-node state rows and
    returns (out, cache); ``backward(cache, d_out)`` returns
    the gradient with respect to the input rows and accumulates
    parameter gradients, which ``zero_grad`` clears.
    """

    def __init__(self, in_dim: int, out_dim: int, rng: random.Random,
                 relu: bool = True):
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("layer dimensions must be positive")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.relu = relu
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        self.W_q = [[rng.uniform(-scale, scale) for _ in range(in_dim)]
                    for _ in range(out_dim)]
        self.W_k = [[rng.uniform(-scale, scale) for _ in range(in_dim)]
                    for _ in range(out_dim)]
        self.W_v = [[rng.uniform(-scale, scale) for _ in range(in_dim)]
                    for _ in range(out_dim)]
        self.b_v = [0.0] * out_dim
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW_q = [[0.0] * self.in_dim for _ in range(self.out_dim)]
        self.gW_k = [[0.0] * self.in_dim for _ in range(self.out_dim)]
        self.gW_v = [[0.0] * self.in_dim for _ in range(self.out_dim)]
        self.gb_v = [0.0] * self.out_dim

    def forward(self, graph, H):
        n = graph.n
        d = self.out_dim
        sqrt_d = math.sqrt(d)
        Q, K, V = [], [], []
        for v in range(n):
            h = H[v]
            Q.append([sum(w * x for w, x in zip(row, h)) for row in self.W_q])
            K.append([sum(w * x for w, x in zip(row, h)) for row in self.W_k])
            V.append([sum(w * x for w, x in zip(row, h)) + b
                      for row, b in zip(self.W_v, self.b_v)])
        nbrs_all, alpha_all, pre_all, out_all = [], [], [], []
        for v in range(n):
            nbrs = [v] + [u for u in graph.neighbors(v) if u != v]
            scores = []
            for u in nbrs:
                s = sum(q * k for q, k in zip(Q[v], K[u])) / sqrt_d
                scores.append(s)
            m = max(scores)
            exps = [math.exp(s - m) for s in scores]
            total = sum(exps)
            alpha = [e / total for e in exps]
            pre = [0.0] * d
            for a, u in zip(alpha, nbrs):
                vu = V[u]
                for j in range(d):
                    pre[j] += a * vu[j]
            nbrs_all.append(nbrs)
            alpha_all.append(alpha)
            pre_all.append(pre)
            if self.relu:
                out_all.append([p if p > 0.0 else 0.0 for p in pre])
            else:
                out_all.append(list(pre))
        cache = {"H": H, "Q": Q, "K": K, "V": V, "nbrs": nbrs_all,
                 "alpha": alpha_all, "pre": pre_all}
        return out_all, cache

    def backward(self, cache, d_out):
        H, Q, K, V = cache["H"], cache["Q"], cache["K"], cache["V"]
        n = len(H)
        d = self.out_dim
        sqrt_d = math.sqrt(d)
        d_pre = []
        for v in range(n):
            if self.relu:
                d_pre.append([d_out[v][j] if cache["pre"][v][j] > 0.0
                              else 0.0 for j in range(d)])
            else:
                d_pre.append(list(d_out[v]))
        dQ = [[0.0] * d for _ in range(n)]
        dK = [[0.0] * d for _ in range(n)]
        dV = [[0.0] * d for _ in range(n)]
        for v in range(n):
            nbrs = cache["nbrs"][v]
            alpha = cache["alpha"][v]
            dalpha = []
            for a, u in zip(alpha, nbrs):
                da = sum(d_pre[v][j] * V[u][j] for j in range(d))
                dalpha.append(da)
                for j in range(d):
                    dV[u][j] += a * d_pre[v][j]
            weighted = sum(a * da for a, da in zip(alpha, dalpha))
            for idx, u in enumerate(nbrs):
                dscore = alpha[idx] * (dalpha[idx] - weighted) / sqrt_d
                if dscore == 0.0:
                    continue
                for j in range(d):
                    dQ[v][j] += dscore * K[u][j]
                    dK[u][j] += dscore * Q[v][j]
        dH = [[0.0] * self.in_dim for _ in range(n)]
        for v in range(n):
            h = H[v]
            for j in range(d):
                if dQ[v][j] != 0.0:
                    for k in range(self.in_dim):
                        self.gW_q[j][k] += dQ[v][j] * h[k]
                if dK[v][j] != 0.0:
                    for k in range(self.in_dim):
                        self.gW_k[j][k] += dK[v][j] * h[k]
                if dV[v][j] != 0.0:
                    for k in range(self.in_dim):
                        self.gW_v[j][k] += dV[v][j] * h[k]
                    self.gb_v[j] += dV[v][j]
                for k in range(self.in_dim):
                    dH[v][k] += (self.W_q[j][k] * dQ[v][j]
                                 + self.W_k[j][k] * dK[v][j]
                                 + self.W_v[j][k] * dV[v][j])
        return dH

    def step(self, lr: float) -> None:
        for j in range(self.out_dim):
            for k in range(self.in_dim):
                self.W_q[j][k] -= lr * self.gW_q[j][k]
                self.W_k[j][k] -= lr * self.gW_k[j][k]
                self.W_v[j][k] -= lr * self.gW_v[j][k]
            self.b_v[j] -= lr * self.gb_v[j]

    def state_dict(self):
        return {"W_q": copy.deepcopy(self.W_q),
                "W_k": copy.deepcopy(self.W_k),
                "W_v": copy.deepcopy(self.W_v),
                "b_v": copy.deepcopy(self.b_v)}

    def load_state_dict(self, state) -> None:
        self.W_q = copy.deepcopy(state["W_q"])
        self.W_k = copy.deepcopy(state["W_k"])
        self.W_v = copy.deepcopy(state["W_v"])
        self.b_v = copy.deepcopy(state["b_v"])
        self.zero_grad()


class GraphTransformerConv:
    """Two-layer TransformerConv: layer 1 to ``hidden_dim``
    under a ReLU, layer 2 to a single logit.

    ``predict`` takes a graph and target node ids and returns
    ``{v: probability}``, matching the other models' interface,
    so the shared training/evaluation protocol applies unchanged.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 16,
                 lr: float = 0.5, seed: int = 0):
        if in_dim <= 0:
            raise ValueError("in_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        rng = random.Random(seed)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.layer1 = TransformerConvLayer(in_dim, hidden_dim, rng,
                                           relu=True)
        self.layer2 = TransformerConvLayer(hidden_dim, 1, rng,
                                           relu=False)
        self.b_out = 0.0
        self.gb_out = 0.0

    def zero_grad(self) -> None:
        self.layer1.zero_grad()
        self.layer2.zero_grad()
        self.gb_out = 0.0

    def forward(self, graph, targets: list[int]):
        H0 = [list(row) for row in graph.features]
        H1, cache1 = self.layer1.forward(graph, H0)
        H2, cache2 = self.layer2.forward(graph, H1)
        probs: dict[int, float] = {}
        for v in targets:
            probs[v] = _sigmoid(H2[v][0] + self.b_out)
        cache = {"layer1": cache1, "layer2": cache2, "H1": H1,
                 "targets": list(targets)}
        return probs, cache

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        probs, _ = self.forward(graph, targets)
        return probs

    def attention_weights(self, graph, v: int, layer: int = 1):
        """Node ``v``'s attention in the chosen layer (1 or 2):
        a list of ``(neighbour, weight)`` pairs, self first,
        weights summing to 1. An isolated node's list is
        exactly ``[(v, 1.0)]``."""
        if layer not in (1, 2):
            raise ValueError("layer must be 1 or 2")
        if not (0 <= v < graph.n):
            raise IndexError(f"node id out of range: {v}")
        H0 = [list(row) for row in graph.features]
        H1, cache1 = self.layer1.forward(graph, H0)
        if layer == 1:
            return _attend_from(cache1, v)
        _, cache2 = self.layer2.forward(graph, H1)
        return _attend_from(cache2, v)

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    pos_weight: float = 1.0) -> float:
        """One SGD step on ``targets``. Returns mean weighted BCE
        loss. ``lr=0.0`` accumulates gradients without moving the
        weights, which the numeric gradient check relies on."""
        self.zero_grad()
        probs, cache = self.forward(graph, targets)
        B = len(targets)
        wts = {v: (pos_weight if labels[v] == 1 else 1.0) for v in targets}
        loss = sum(wts[v] * _bce(probs[v], labels[v]) for v in targets) / B
        dH2 = [[0.0] for _ in range(graph.n)]
        for v in targets:
            dl = wts[v] * (probs[v] - labels[v]) / B
            dH2[v][0] = dl
            self.gb_out += dl
        dH1 = self.layer2.backward(cache["layer2"], dH2)
        self.layer1.backward(cache["layer1"], dH1)
        self.layer1.step(self.lr)
        self.layer2.step(self.lr)
        self.b_out -= self.lr * self.gb_out
        return loss

    def state_dict(self):
        return {"layer1": self.layer1.state_dict(),
                "layer2": self.layer2.state_dict(),
                "b_out": self.b_out}

    def load_state_dict(self, state) -> None:
        self.layer1.load_state_dict(state["layer1"])
        self.layer2.load_state_dict(state["layer2"])
        self.b_out = state["b_out"]
        self.zero_grad()


def attention_weights(model: GraphTransformerConv, graph, v: int,
                      layer: int = 1):
    """Module-level wrapper for :meth:`GraphTransformerConv.attention_weights`."""
    return model.attention_weights(graph, v, layer=layer)

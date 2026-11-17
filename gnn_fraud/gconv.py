"""GraphConv (Morris et al., 2019, Weisfeiler-Leman GNN), from scratch.

Layer update (one layer, all nodes at once)::

    h'_v = relu(W_self @ h_v + W_neigh @ sum_{u in N(v)} h_u + b)

Two things make this different from its closest neighbours in this repo:

* **Separate self and neighbour matrices, not a scalar.** GIN weights
  the node's own features by one learnable scalar ``(1 + eps)`` and
  runs the combined sum through an MLP. GraphConv gives the self term
  and the neighbour sum their own full matrices, so each output
  dimension can weight "what I am" against "how much neighbourhood I
  have" independently - a strictly richer self/neighbour split than
  GIN's, bought without GIN's second MLP layer.
* **Sum, not mean or a normalized sum.** Like GIN, the neighbour term
  is a raw sum, so neighbourhood count survives: a ring member with a
  dozen ring neighbours produces a neighbour term an order of
  magnitude larger than a background account with one or two, where
  GraphSAGE's mean erases that count and GCN's normalization dilutes
  it. Unlike GIN there is no MLP after the sum - one linear map per
  term, then the ReLU - so this is the minimal count-preserving layer.

Like GCN and GIN there is no neighbour sampling: the sum needs the
true neighbourhood, so training runs full-batch over the whole graph
(``train_batch`` still scores only the target subset, but the forward
pass computes every node's embedding). A final linear layer plus
sigmoid maps the layer-2 embedding to a fraud probability, trained
with binary cross-entropy and full backprop through both layers
(numeric-gradient-checked at the layer and end to end, including the
input gradient). All linear algebra is hand-rolled lists-of-floats -
no numpy, no torch.
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


class GraphConvLayer:
    """Single GraphConv layer: ``relu(W_self @ h_v + W_neigh @
    sum_{u in N(v)} h_u + b)``. Operates on the whole graph at once,
    so ``forward`` takes the full embedding list indexed by node id."""

    def __init__(self, in_dim, out_dim, rng):
        scale = math.sqrt(2.0 / (in_dim + out_dim))
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
        self.b = [0.0] * out_dim
        self.zero_grad()

    def zero_grad(self):
        self.gW_self = [[0.0] * self.in_dim for _ in range(self.out_dim)]
        self.gW_neigh = [[0.0] * self.in_dim for _ in range(self.out_dim)]
        self.gb = [0.0] * self.out_dim

    def forward(self, graph, H):
        """Returns (H_out, cache): cache holds (H_in, neighbour sums,
        pre-activations, neighbour lists)."""
        sums = []
        for v in range(graph.n):
            acc = [0.0] * self.in_dim
            for u in graph.neighbors(v):
                hu = H[u]
                for j in range(self.in_dim):
                    acc[j] += hu[j]
            sums.append(acc)
        pre = []
        for v in range(graph.n):
            s = _matvec(self.W_self, H[v])
            t = _matvec(self.W_neigh, sums[v])
            pre.append([si + ti + bi for si, ti, bi in zip(s, t, self.b)])
        nbrs = [graph.neighbors(v) for v in range(graph.n)]
        return [_relu(p) for p in pre], (H, sums, pre, nbrs)

    def backward(self, d_out, cache):
        """Backprop one layer. Returns d_H, the gradient w.r.t. the
        input embeddings. The neighbour sum is symmetric on an
        undirected graph, so in reverse each node's d_neigh flows back
        unscaled to every neighbour, while d_self stays on the node."""
        H, sums, pre, nbrs = cache
        n = len(sums)
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
                    self.gW_neigh[i][j] += di * sums[v][j]
                self.gb[i] += di
            d_self = _mat_t_vec(self.W_self, dv)
            d_neigh = _mat_t_vec(self.W_neigh, dv)
            for j in range(self.in_dim):
                d_H[v][j] += d_self[j]
            for u in nbrs[v]:
                row = d_H[u]
                for j in range(self.in_dim):
                    row[j] += d_neigh[j]
        return d_H

    def step(self, lr):
        for i in range(self.out_dim):
            for j in range(self.in_dim):
                self.W_self[i][j] -= lr * self.gW_self[i][j]
                self.W_neigh[i][j] -= lr * self.gW_neigh[i][j]
            self.b[i] -= lr * self.gb[i]

    def state_dict(self):
        return {
            "W_self": copy.deepcopy(self.W_self),
            "W_neigh": copy.deepcopy(self.W_neigh),
            "b": copy.deepcopy(self.b),
        }

    def load_state_dict(self, state):
        self.W_self = copy.deepcopy(state["W_self"])
        self.W_neigh = copy.deepcopy(state["W_neigh"])
        self.b = copy.deepcopy(state["b"])
        self.zero_grad()


class GraphConvNet:
    """Two-layer GraphConv for binary node classification.

    Full-graph forward (no sampling), linear plus sigmoid output
    head, weighted BCE loss like :class:`GraphSAGE`.
    """

    def __init__(self, in_dim, hidden_dim=16, lr=0.1, seed=0):
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.rng = random.Random(seed)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.layer1 = GraphConvLayer(in_dim, hidden_dim, self.rng)
        self.layer2 = GraphConvLayer(hidden_dim, hidden_dim, self.rng)
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

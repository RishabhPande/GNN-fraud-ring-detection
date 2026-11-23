"""GraphSAGE with a virtual node, from scratch.

Every model in this repo reads a node's neighborhood, and only
its neighborhood: information between two accounts travels one
edge per layer, so accounts at opposite ends of a thin pattern
(a mule chain, a smurfing fan) never see each other in a two
layer model. A virtual node fixes the reach problem cheaply.
One extra node is connected to every account; in this
implementation it is not stored as a graph node but computed
directly, as the mean of all node states, and each layer adds a
learned transform of that global vector to every node::

    g        = (1 / n) sum_u h_u              (virtual node state)
    pre_v    = W_self @ h_v + W_neigh @ mean_{u in N(v)} h_u
               + W_glob @ g + b
    h'_v     = relu(pre_v)

The virtual node is a broadcast channel, not a shortcut around
the graph: g is the same for every node in a layer, so by itself
it cannot rank anyone. What it gives each node is context - how
unusual its own neighborhood looks against the whole graph's
average state - and it gives distant nodes a two step path
(node, virtual node, node) in a single layer.

Setting ``W_glob`` to zero recovers a plain full-graph
GraphSAGE mean model exactly. Backprop through the global term
distributes each node's output gradient over all nodes at
weight 1 / n, and backprop through the neighbor mean routes
each node's gradient to its neighbors at weight 1 / degree
(numeric-gradient-checked at the layer and end to end). An
isolated node has an empty neighbor mean (zero vector) but
still receives the global term. All linear algebra is
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


def global_mean(H: list[list[float]]) -> list[float]:
    """The virtual node's state: the elementwise mean of all
    node states. This is the vector every node reads through
    ``W_glob``."""
    if not H:
        raise ValueError("H must have at least one row")
    width = len(H[0])
    if any(len(row) != width for row in H):
        raise ValueError("all hidden rows must share one width")
    n = len(H)
    return [sum(row[j] for row in H) / n for j in range(width)]


def neighbor_means(graph, H: list[list[float]]) -> list[list[float]]:
    """Per-node mean of neighbor states. Isolated nodes get the
    zero vector (there is no neighborhood to average)."""
    if len(H) != graph.n:
        raise ValueError("H must have one row per node")
    width = len(H[0])
    out = []
    for v in range(graph.n):
        nbrs = graph.neighbors(v)
        if not nbrs:
            out.append([0.0] * width)
            continue
        acc = [0.0] * width
        for u in nbrs:
            hu = H[u]
            for j in range(width):
                acc[j] += hu[j]
        out.append([a / len(nbrs) for a in acc])
    return out


class VNodeLayer:
    """One virtual-node GraphSAGE layer: self term, neighbor
    mean term, and global (virtual node) term, each with its own
    weight matrix, plus a bias. Returns pre-activations; the
    model applies the nonlinearity."""

    def __init__(self, in_dim: int, out_dim: int, rng: random.Random):
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("layer dims must be positive")
        self.in_dim = in_dim
        self.out_dim = out_dim
        scale = math.sqrt(2.0 / (in_dim + out_dim))

        def mat():
            return [[rng.uniform(-scale, scale) for _ in range(in_dim)]
                    for _ in range(out_dim)]

        self.W_self = mat()
        self.W_neigh = mat()
        self.W_glob = mat()
        self.b = [0.0] * out_dim
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW_self = [[0.0] * self.in_dim for _ in range(self.out_dim)]
        self.gW_neigh = [[0.0] * self.in_dim for _ in range(self.out_dim)]
        self.gW_glob = [[0.0] * self.in_dim for _ in range(self.out_dim)]
        self.gb = [0.0] * self.out_dim

    def forward(self, graph, H: list[list[float]]):
        nmean = neighbor_means(graph, H)
        g = global_mean(H)
        pre = []
        for v in range(graph.n):
            hv = H[v]
            mv = nmean[v]
            row = []
            for j in range(self.out_dim):
                s = self.b[j]
                ws = self.W_self[j]
                wn = self.W_neigh[j]
                wg = self.W_glob[j]
                for k in range(self.in_dim):
                    s += ws[k] * hv[k] + wn[k] * mv[k] + wg[k] * g[k]
                row.append(s)
            pre.append(row)
        cache = {"H": H, "nmean": nmean, "g": g}
        return pre, cache

    def backward(self, graph, d_pre: list[list[float]], cache) -> list[list[float]]:
        """Accumulate parameter gradients and return dH, the
        gradient with respect to the layer input."""
        H = cache["H"]
        nmean = cache["nmean"]
        g = cache["g"]
        n = graph.n
        # Parameter gradients.
        for v in range(n):
            dv = d_pre[v]
            if not any(dv):
                continue
            hv = H[v]
            mv = nmean[v]
            for j in range(self.out_dim):
                d = dv[j]
                if d == 0.0:
                    continue
                gs = self.gW_self[j]
                gn = self.gW_neigh[j]
                gg = self.gW_glob[j]
                for k in range(self.in_dim):
                    gs[k] += d * hv[k]
                    gn[k] += d * mv[k]
                    gg[k] += d * g[k]
                self.gb[j] += d
        # Input gradient: self path, neighbor-mean path (node v
        # feeds every node that lists it as a neighbor, at
        # 1 / that node's degree), and global path (node v feeds
        # every node's global vector at 1 / n).
        d_glob_sum = [0.0] * self.out_dim
        for v in range(n):
            for j in range(self.out_dim):
                d_glob_sum[j] += d_pre[v][j]
        dH = [[0.0] * self.in_dim for _ in range(n)]
        for v in range(n):
            dv = d_pre[v]
            acc = dH[v]
            for j in range(self.out_dim):
                d = dv[j]
                if d == 0.0:
                    continue
                ws = self.W_self[j]
                for k in range(self.in_dim):
                    acc[k] += d * ws[k]
            # v as a neighbor of u: u's mean includes H[v] at
            # weight 1 / degree(u).
            for u in graph.neighbors(v):
                du = d_pre[u]
                deg = graph.degree(u)
                for j in range(self.out_dim):
                    d = du[j]
                    if d == 0.0:
                        continue
                    wn = self.W_neigh[j]
                    for k in range(self.in_dim):
                        acc[k] += d * wn[k] / deg
            # v in the global mean of every node.
            for j in range(self.out_dim):
                d = d_glob_sum[j]
                if d == 0.0:
                    continue
                wg = self.W_glob[j]
                for k in range(self.in_dim):
                    acc[k] += d * wg[k] / n
        return dH

    def step(self, lr: float) -> None:
        for j in range(self.out_dim):
            for k in range(self.in_dim):
                self.W_self[j][k] -= lr * self.gW_self[j][k]
                self.W_neigh[j][k] -= lr * self.gW_neigh[j][k]
                self.W_glob[j][k] -= lr * self.gW_glob[j][k]
            self.b[j] -= lr * self.gb[j]


class VirtualNodeSAGE:
    """Two virtual-node GraphSAGE layers with ReLU, then a
    linear sigmoid head. Full-graph forward (the global mean
    needs every node), matching the other full-graph models'
    interface: ``predict(graph, targets)`` returns
    ``{v: probability}``."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 16,
        lr: float = 0.1,
        seed: int = 0,
    ):
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        rng = random.Random(seed)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.layer1 = VNodeLayer(in_dim, hidden_dim, rng)
        self.layer2 = VNodeLayer(hidden_dim, hidden_dim, rng)
        out_scale = math.sqrt(2.0 / (hidden_dim + 1))
        self.w_out = [rng.uniform(-out_scale, out_scale)
                      for _ in range(hidden_dim)]
        self.b_out = 0.0
        self.zero_grad()

    def zero_grad(self) -> None:
        self.layer1.zero_grad()
        self.layer2.zero_grad()
        self.gw_out = [0.0] * self.hidden_dim
        self.gb_out = 0.0

    def forward(self, graph, targets: list[int]):
        pre1, cache1 = self.layer1.forward(graph, graph.features)
        h1 = [[p if p > 0.0 else 0.0 for p in row] for row in pre1]
        pre2, cache2 = self.layer2.forward(graph, h1)
        h2 = [[p if p > 0.0 else 0.0 for p in row] for row in pre2]
        probs: dict[int, float] = {}
        for v in targets:
            logit = (sum(w * zj for w, zj in zip(self.w_out, h2[v]))
                     + self.b_out)
            probs[v] = _sigmoid(logit)
        cache = {"cache1": cache1, "pre1": pre1, "h1": h1,
                 "cache2": cache2, "pre2": pre2, "h2": h2,
                 "targets": list(targets)}
        return probs, cache

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        probs, _ = self.forward(graph, targets)
        return probs

    def virtual_state(self, graph, layer: int = 1) -> list[float]:
        """The virtual node's state at a layer's input: layer 1
        reads the raw-feature mean, layer 2 the hidden mean.
        Lets a caller inspect what global context the model is
        broadcasting."""
        if layer == 1:
            return global_mean(graph.features)
        if layer == 2:
            pre1, _ = self.layer1.forward(graph, graph.features)
            h1 = [[p if p > 0.0 else 0.0 for p in row] for row in pre1]
            return global_mean(h1)
        raise ValueError("layer must be 1 or 2")

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    pos_weight: float = 1.0) -> float:
        """One SGD step on ``targets``. Returns mean weighted BCE
        loss. ``lr=0.0`` accumulates gradients without moving the
        weights, which the numeric gradient check relies on."""
        self.zero_grad()
        probs, cache = self.forward(graph, targets)
        pre1 = cache["pre1"]
        pre2 = cache["pre2"]
        h2 = cache["h2"]

        B = len(targets)
        wts = {v: (pos_weight if labels[v] == 1 else 1.0)
               for v in targets}
        loss = sum(wts[v] * _bce(probs[v], labels[v])
                   for v in targets) / B

        d_h2 = [[0.0] * self.hidden_dim for _ in range(graph.n)]
        for v in targets:
            dl = wts[v] * (probs[v] - labels[v]) / B
            if dl == 0.0:
                continue
            for j in range(self.hidden_dim):
                d_h2[v][j] = dl * self.w_out[j]
                self.gw_out[j] += dl * h2[v][j]
            self.gb_out += dl

        d_pre2 = [[d_h2[v][j] * (1.0 if pre2[v][j] > 0.0 else 0.0)
                   for j in range(self.hidden_dim)]
                  for v in range(graph.n)]
        d_h1 = self.layer2.backward(graph, d_pre2, cache["cache2"])
        d_pre1 = [[d_h1[v][j] * (1.0 if pre1[v][j] > 0.0 else 0.0)
                   for j in range(self.hidden_dim)]
                  for v in range(graph.n)]
        self.layer1.backward(graph, d_pre1, cache["cache1"])

        self.layer1.step(self.lr)
        self.layer2.step(self.lr)
        for j in range(self.hidden_dim):
            self.w_out[j] -= self.lr * self.gw_out[j]
        self.b_out -= self.lr * self.gb_out
        return loss

    def state_dict(self):
        return {
            "layer1": {
                "W_self": copy.deepcopy(self.layer1.W_self),
                "W_neigh": copy.deepcopy(self.layer1.W_neigh),
                "W_glob": copy.deepcopy(self.layer1.W_glob),
                "b": copy.deepcopy(self.layer1.b),
            },
            "layer2": {
                "W_self": copy.deepcopy(self.layer2.W_self),
                "W_neigh": copy.deepcopy(self.layer2.W_neigh),
                "W_glob": copy.deepcopy(self.layer2.W_glob),
                "b": copy.deepcopy(self.layer2.b),
            },
            "w_out": copy.deepcopy(self.w_out),
            "b_out": self.b_out,
        }

    def load_state_dict(self, state) -> None:
        for name, layer in (("layer1", self.layer1),
                            ("layer2", self.layer2)):
            layer.W_self = copy.deepcopy(state[name]["W_self"])
            layer.W_neigh = copy.deepcopy(state[name]["W_neigh"])
            layer.W_glob = copy.deepcopy(state[name]["W_glob"])
            layer.b = copy.deepcopy(state[name]["b"])
        self.w_out = copy.deepcopy(state["w_out"])
        self.b_out = state["b_out"]
        self.zero_grad()

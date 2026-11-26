"""LINKX (Lim et al., 2021), from scratch - and stripped to the
question this repo can ask it.

Every other model here mixes a node's features with its
neighbours' features, by message passing or by propagation.
LINKX does neither. It runs two separate MLPs and concatenates
their outputs at the head::

    F_v     = relu(W_feat @ x_v + b_feat)          (features)
    A_v     = relu(W_adj @ a_v + b_adj)            (structure)
    logit_v = w_out . [F_v || A_v] + b_out

where ``a_v`` is node ``v``'s own row of the row-normalized
adjacency with a self-loop, ``a_v[u] = 1 / (deg(v) + 1)`` for
``u`` in ``N(v) U {v}`` and 0 elsewhere, read sparsely (only the
``deg(v) + 1`` nonzero columns are ever touched). The structure
branch is not a propagation: no neighbour's *features* enter it,
only the identity of the neighbours, through a learned weight
column per account. A ring member's row names a dozen other ring
members, a background account's row names one or two strangers,
so the branch can learn "wired to these accounts" as a feature
in its own right - the wiring as input, not as a mixing rule.

That is also the price, stated plainly: ``W_adj`` has one column
per account, so the model is tied to the graph it was built for
(like LightGCN's embeddings, and for the same reason). A new
account has no column and cannot be scored until the model is
rebuilt and retrained with it. What LINKX buys for that price is
separation: the head's two blocks can be read independently, so
``branch_norms()`` reports how much of the decision the feature
view carries against the structure view, a split no message-
passing model here can make because its views are already mixed
by the time they reach the head.

An isolated node's row is exactly the one-hot on itself
(``deg = 0``, so its self weight is 1), so its structure state is
exactly column ``v`` of ``W_adj`` passed through the bias and
ReLU. All linear algebra is hand-rolled lists-of-floats - no
numpy, no torch.
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


def adjacency_row(graph, v: int) -> list[tuple[int, float]]:
    """Node ``v``'s sparse row of the row-normalized adjacency
    with a self-loop: ``(v, w)`` first, then one ``(u, w)`` per
    neighbour, all at the same weight ``w = 1 / (deg(v) + 1)``,
    so the row sums to exactly 1. An isolated node's row is
    ``[(v, 1.0)]``."""
    if not (0 <= v < graph.n):
        raise IndexError(f"node id out of range: {v}")
    w = 1.0 / (graph.degree(v) + 1)
    return [(v, w)] + [(u, w) for u in sorted(set(graph.neighbors(v)))]


def adjacency_rows(graph) -> list[list[tuple[int, float]]]:
    """Every node's sparse adjacency row (see :func:`adjacency_row`)."""
    return [adjacency_row(graph, v) for v in range(graph.n)]


class LINKX:
    """LINKX: a feature MLP and a structure MLP (over the node's
    own adjacency row) of ``hidden_dim`` each, concatenated into
    a linear head. No message passing anywhere.

    ``predict`` takes a graph and target node ids and returns
    ``{v: probability}``, matching the other models' interface,
    so the shared training/evaluation protocol applies unchanged.
    The graph must have the ``n_nodes`` the model was built for.
    """

    def __init__(
        self,
        in_dim: int,
        n_nodes: int,
        hidden_dim: int = 16,
        lr: float = 0.5,
        seed: int = 0,
    ):
        if in_dim <= 0:
            raise ValueError("in_dim must be positive")
        if n_nodes <= 0:
            raise ValueError("n_nodes must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        rng = random.Random(seed)
        self.in_dim = in_dim
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.lr = lr
        f_scale = math.sqrt(2.0 / (in_dim + hidden_dim))
        self.W_feat = [
            [rng.uniform(-f_scale, f_scale) for _ in range(in_dim)]
            for _ in range(hidden_dim)
        ]
        self.b_feat = [0.0] * hidden_dim
        a_scale = math.sqrt(2.0 / (n_nodes + hidden_dim))
        self.W_adj = [
            [rng.uniform(-a_scale, a_scale) for _ in range(n_nodes)]
            for _ in range(hidden_dim)
        ]
        self.b_adj = [0.0] * hidden_dim
        out_scale = math.sqrt(2.0 / (2 * hidden_dim + 1))
        self.w_out = [rng.uniform(-out_scale, out_scale)
                      for _ in range(2 * hidden_dim)]
        self.b_out = 0.0
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW_feat = [[0.0] * self.in_dim for _ in range(self.hidden_dim)]
        self.gb_feat = [0.0] * self.hidden_dim
        self.gW_adj = [[0.0] * self.n_nodes for _ in range(self.hidden_dim)]
        self.gb_adj = [0.0] * self.hidden_dim
        self.gw_out = [0.0] * (2 * self.hidden_dim)
        self.gb_out = 0.0

    def branch_norms(self) -> tuple[float, float]:
        """Frobenius-style norms of the head's two blocks:
        ``(feature block, structure block)``. The split the model
        exists for - how much of the logit each view can carry,
        read directly off ``w_out`` without retraining."""
        feat = math.sqrt(sum(w * w for w in self.w_out[:self.hidden_dim]))
        adj = math.sqrt(sum(w * w for w in self.w_out[self.hidden_dim:]))
        return feat, adj

    # -- forward ----------------------------------------------------------

    def forward(self, graph, targets: list[int]):
        """Forward pass over the whole graph. Returns (probs dict,
        cache) with probs only for the requested targets."""
        if graph.n != self.n_nodes:
            raise ValueError(
                f"model was built for {self.n_nodes} nodes, "
                f"graph has {graph.n}")
        rows = adjacency_rows(graph)
        pre_feat, feat, pre_adj, adj = [], [], [], []
        for v in range(graph.n):
            x = graph.features[v]
            pf = [
                sum(w * xi for w, xi in zip(row, x)) + b
                for row, b in zip(self.W_feat, self.b_feat)
            ]
            pre_feat.append(pf)
            feat.append([p if p > 0.0 else 0.0 for p in pf])
            pa = []
            for j in range(self.hidden_dim):
                acc = self.b_adj[j]
                wrow = self.W_adj[j]
                for u, c in rows[v]:
                    acc += c * wrow[u]
                pa.append(acc)
            pre_adj.append(pa)
            adj.append([p if p > 0.0 else 0.0 for p in pa])
        probs: dict[int, float] = {}
        for v in targets:
            z = feat[v] + adj[v]
            logit = (sum(w * zj for w, zj in zip(self.w_out, z))
                     + self.b_out)
            probs[v] = _sigmoid(logit)
        cache = {
            "rows": rows,
            "pre_feat": pre_feat,
            "feat": feat,
            "pre_adj": pre_adj,
            "adj": adj,
            "targets": list(targets),
        }
        return probs, cache

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        """Inference - both branches are deterministic, so this is exact."""
        probs, _ = self.forward(graph, targets)
        return probs

    # -- training step -----------------------------------------------------

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    pos_weight: float = 1.0) -> float:
        """One SGD step on ``targets``. Returns mean weighted BCE loss.

        The two branches backprop independently from the head's
        split gradient: the feature branch through its ReLU into
        ``W_feat``, the structure branch through its ReLU into
        the sparse columns of ``W_adj`` its row actually names
        (each at the row weight ``1 / (deg(v) + 1)``). There is
        no graph backward at all, because nothing was propagated.
        ``lr=0.0`` accumulates gradients without moving the
        weights, which the numeric gradient check relies on.
        """
        self.zero_grad()
        probs, cache = self.forward(graph, targets)
        rows = cache["rows"]
        pre_feat = cache["pre_feat"]
        feat = cache["feat"]
        pre_adj = cache["pre_adj"]
        adj = cache["adj"]
        H = self.hidden_dim

        B = len(targets)
        wts = {v: (pos_weight if labels[v] == 1 else 1.0)
               for v in targets}
        loss = sum(wts[v] * _bce(probs[v], labels[v])
                   for v in targets) / B

        for v in targets:
            dl = wts[v] * (probs[v] - labels[v]) / B
            if dl == 0.0:
                continue
            z = feat[v] + adj[v]
            for j in range(2 * H):
                self.gw_out[j] += dl * z[j]
            self.gb_out += dl
            x = graph.features[v]
            for j in range(H):
                d_pre = (dl * self.w_out[j]
                         * (1.0 if pre_feat[v][j] > 0.0 else 0.0))
                if d_pre != 0.0:
                    for k in range(self.in_dim):
                        self.gW_feat[j][k] += d_pre * x[k]
                    self.gb_feat[j] += d_pre
                d_pa = (dl * self.w_out[H + j]
                        * (1.0 if pre_adj[v][j] > 0.0 else 0.0))
                if d_pa != 0.0:
                    for u, c in rows[v]:
                        self.gW_adj[j][u] += d_pa * c
                    self.gb_adj[j] += d_pa

        # SGD update.
        for j in range(H):
            for k in range(self.in_dim):
                self.W_feat[j][k] -= self.lr * self.gW_feat[j][k]
            self.b_feat[j] -= self.lr * self.gb_feat[j]
            for u in range(self.n_nodes):
                self.W_adj[j][u] -= self.lr * self.gW_adj[j][u]
            self.b_adj[j] -= self.lr * self.gb_adj[j]
            self.w_out[j] -= self.lr * self.gw_out[j]
            self.w_out[H + j] -= self.lr * self.gw_out[H + j]
        self.b_out -= self.lr * self.gb_out
        return loss

    # -- checkpointing ------------------------------------------------------

    def state_dict(self):
        return {
            "W_feat": copy.deepcopy(self.W_feat),
            "b_feat": copy.deepcopy(self.b_feat),
            "W_adj": copy.deepcopy(self.W_adj),
            "b_adj": copy.deepcopy(self.b_adj),
            "w_out": copy.deepcopy(self.w_out),
            "b_out": self.b_out,
        }

    def load_state_dict(self, state) -> None:
        self.W_feat = copy.deepcopy(state["W_feat"])
        self.b_feat = copy.deepcopy(state["b_feat"])
        self.W_adj = copy.deepcopy(state["W_adj"])
        self.b_adj = copy.deepcopy(state["b_adj"])
        self.w_out = copy.deepcopy(state["w_out"])
        self.b_out = state["b_out"]
        self.zero_grad()

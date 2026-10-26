"""Graph Isomorphism Network (Xu et al., 2019), from scratch.

Layer update (one layer, all nodes at once)::

    h'_v = MLP((1 + eps) * h_v + sum_{u in N(v)} h_u)

Three things make this different from every other model in this repo:

* **Sum, not mean or normalized sum.** GraphSAGE averages neighbors
  (multiset size is lost: two suspicious neighbors look like twenty),
  and GCN down-weights by degree (a hub's message is diluted). GIN sums
  raw, so the aggregation is injective over neighbor multisets -- the
  property the paper proves is necessary to distinguish graph
  structures as well as the Weisfeiler-Lehman test can. On fraud
  graphs that matters twice over: a ring member's neighborhood is not
  just suspicious in *kind*, it is suspicious in *count* (a dozen
  ring neighbors vs one or two background neighbors), and sum keeps
  that count where mean and normalization throw it away.
* **An MLP, not a single linear map.** A single linear layer composed
  with sum aggregation cannot represent injective multiset functions
  in general; the paper's fix is a multi-layer perceptron after the
  sum. Here the MLP is two linear layers with a ReLU between them
  (and a ReLU on the output, matching the other layers in this repo).
* **A learnable ``eps``.** ``(1 + eps)`` weights the node's own
  features against its neighbors' sum. It starts at 0 -- the plain sum
  -- and is learned alongside the weights, with its own numeric-checked
  gradient.

Like GCN there is no neighbor sampling: the sum needs the true
neighborhood, so training runs full-batch over the whole graph
(``train_batch`` still scores only the target subset, but the forward
pass computes every node's embedding). A final linear layer + sigmoid
maps the layer-2 embedding to a fraud probability, trained with
binary cross-entropy and full backprop through both layers. All linear
algebra is hand-rolled lists-of-floats -- no numpy, no torch.
"""

from __future__ import annotations

import copy
import math
import random


# ---------------------------------------------------------------------------
# tiny linalg helpers (lists of floats)
# ---------------------------------------------------------------------------


def _matvec(W: list[list[float]], x: list[float]) -> list[float]:
    return [sum(w * xi for w, xi in zip(row, x)) for row in W]


def _mat_t_vec(W: list[list[float]], v: list[float]) -> list[float]:
    n_cols = len(W[0])
    return [sum(W[r][c] * v[r] for r in range(len(W))) for c in range(n_cols)]


def _vec_add(a: list[float], b: list[float]) -> list[float]:
    return [ai + bi for ai, bi in zip(a, b)]


def _scale(v: list[float], s: float) -> list[float]:
    return [vi * s for vi in v]


def _relu(v: list[float]) -> list[float]:
    return [x if x > 0.0 else 0.0 for x in v]


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _bce(p: float, y: int) -> float:
    p = min(max(p, 1e-7), 1.0 - 1e-7)
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


# ---------------------------------------------------------------------------
# one GIN layer
# ---------------------------------------------------------------------------


class GINLayer:
    """Single GIN layer: sum aggregation, learnable ``eps``, then a
    two-linear-layer MLP with ReLU. Operates on the whole graph at once
    (the sum needs true neighborhoods), so ``forward`` takes the full
    embedding list indexed by node id."""

    def __init__(self, in_dim: int, out_dim: int, rng: random.Random):
        scale1 = math.sqrt(2.0 / (in_dim + out_dim))
        scale2 = math.sqrt(2.0 / (out_dim + out_dim))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eps = 0.0
        self.W1 = [
            [rng.uniform(-scale1, scale1) for _ in range(in_dim)]
            for _ in range(out_dim)
        ]
        self.b1 = [0.0] * out_dim
        self.W2 = [
            [rng.uniform(-scale2, scale2) for _ in range(out_dim)]
            for _ in range(out_dim)
        ]
        self.b2 = [0.0] * out_dim
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW1 = [[0.0] * self.in_dim for _ in range(self.out_dim)]
        self.gb1 = [0.0] * self.out_dim
        self.gW2 = [[0.0] * self.out_dim for _ in range(self.out_dim)]
        self.gb2 = [0.0] * self.out_dim
        self.geps = 0.0

    def forward(self, graph, H: list[list[float]]):
        """Returns (H_out, cache) where cache holds everything backprop
        needs: (H_in, agg, pre1, act, pre2)."""
        agg = []
        for v in range(graph.n):
            acc = _scale(H[v], 1.0 + self.eps)
            for u in graph.neighbors(v):
                hu = H[u]
                for j in range(self.in_dim):
                    acc[j] += hu[j]
            agg.append(acc)
        pre1 = [_vec_add(_matvec(self.W1, a), self.b1) for a in agg]
        act = [_relu(p) for p in pre1]
        pre2 = [_vec_add(_matvec(self.W2, a), self.b2) for a in act]
        nbrs = [graph.neighbors(v) for v in range(graph.n)]
        return [_relu(p) for p in pre2], (H, agg, pre1, act, pre2, nbrs)

    def backward(self, d_out: list[list[float]], cache
                 ) -> list[list[float]]:
        """Backprop one layer.

        Returns ``d_H`` -- the gradient w.r.t. the input embeddings,
        indexed by node id. The sum aggregation is symmetric on an
        undirected graph, so in reverse each node's ``d_agg`` flows
        back to itself (scaled by ``1 + eps``) and to every neighbor
        (unscaled).
        """
        H, agg, pre1, act, pre2, nbrs = cache
        n = len(agg)
        d_pre2 = [
            [d * (1.0 if p > 0.0 else 0.0) for d, p in zip(d_out[v], pre2[v])]
            for v in range(n)
        ]
        d_act = [_mat_t_vec(self.W2, d_pre2[v]) for v in range(n)]
        for v in range(n):
            dv = d_pre2[v]
            if not any(dv):
                continue
            for i in range(self.out_dim):
                di = dv[i]
                if di == 0.0:
                    continue
                for j in range(self.out_dim):
                    self.gW2[i][j] += di * act[v][j]
                self.gb2[i] += di
        d_pre1 = [
            [d * (1.0 if p > 0.0 else 0.0) for d, p in zip(d_act[v], pre1[v])]
            for v in range(n)
        ]
        d_agg = [_mat_t_vec(self.W1, d_pre1[v]) for v in range(n)]
        for v in range(n):
            dv = d_pre1[v]
            if not any(dv):
                continue
            for i in range(self.out_dim):
                di = dv[i]
                if di == 0.0:
                    continue
                for j in range(self.in_dim):
                    self.gW1[i][j] += di * agg[v][j]
                self.gb1[i] += di
        # Epsilon gradient: d_agg flows through (1 + eps) * h_v.
        for v in range(n):
            for j in range(self.in_dim):
                self.geps += d_agg[v][j] * H[v][j]
        d_H = [[0.0] * self.in_dim for _ in range(n)]
        for v in range(n):
            dv = d_agg[v]
            for j in range(self.in_dim):
                d_H[v][j] += (1.0 + self.eps) * dv[j]
            for u in nbrs[v]:
                row = d_H[u]
                for j in range(self.in_dim):
                    row[j] += dv[j]
        return d_H

    def step(self, lr: float) -> None:
        for i in range(self.out_dim):
            for j in range(self.in_dim):
                self.W1[i][j] -= lr * self.gW1[i][j]
            self.b1[i] -= lr * self.gb1[i]
            for j in range(self.out_dim):
                self.W2[i][j] -= lr * self.gW2[i][j]
            self.b2[i] -= lr * self.gb2[i]
        self.eps -= lr * self.geps

    def state_dict(self):
        return {
            "W1": copy.deepcopy(self.W1),
            "b1": copy.deepcopy(self.b1),
            "W2": copy.deepcopy(self.W2),
            "b2": copy.deepcopy(self.b2),
            "eps": self.eps,
        }

    def load_state_dict(self, state) -> None:
        self.W1 = copy.deepcopy(state["W1"])
        self.b1 = copy.deepcopy(state["b1"])
        self.W2 = copy.deepcopy(state["W2"])
        self.b2 = copy.deepcopy(state["b2"])
        self.eps = state["eps"]
        self.zero_grad()


# ---------------------------------------------------------------------------
# two-layer model
# ---------------------------------------------------------------------------


class GraphIsomorphism:
    """Two-layer GIN for binary node classification.

    Full-graph forward (no sampling -- see the module docstring), linear
    + sigmoid output head, weighted BCE loss like :class:`GraphSAGE`.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 8,
        lr: float = 0.1,
        seed: int = 0,
    ):
        self.rng = random.Random(seed)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.layer1 = GINLayer(in_dim, hidden_dim, self.rng)
        self.layer2 = GINLayer(hidden_dim, hidden_dim, self.rng)
        scale = math.sqrt(2.0 / (hidden_dim + 1))
        self.w_out = [self.rng.uniform(-scale, scale) for _ in range(hidden_dim)]
        self.b_out = 0.0
        self.gw_out = [0.0] * hidden_dim
        self.gb_out = 0.0

    # -- forward ----------------------------------------------------------

    def forward(self, graph, targets: list[int]):
        """Forward pass over the whole graph. Returns (probs dict, cache)
        with probs only for the requested targets."""
        h1, cache1 = self.layer1.forward(graph, graph.features)
        h2, cache2 = self.layer2.forward(graph, h1)

        probs: dict[int, float] = {}
        for v in targets:
            z = sum(w * h for w, h in zip(self.w_out, h2[v])) + self.b_out
            probs[v] = _sigmoid(z)

        cache = {"h1": h1, "cache1": cache1,
                 "h2": h2, "cache2": cache2, "targets": list(targets)}
        return probs, cache

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        """Inference -- deterministic, no sampling anywhere in GIN."""
        probs, _ = self.forward(graph, targets)
        return probs

    # -- training step -----------------------------------------------------

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    pos_weight: float = 1.0) -> float:
        """One full-batch SGD step scored on ``targets``. Returns mean
        weighted BCE loss. The forward pass still embeds every node:
        targets' embeddings depend on their 2-hop neighborhoods."""
        self.layer1.zero_grad()
        self.layer2.zero_grad()
        self.gw_out = [0.0] * self.hidden_dim
        self.gb_out = 0.0

        probs, cache = self.forward(graph, targets)
        h2, cache2 = cache["h2"], cache["cache2"]

        B = len(targets)
        weights = {v: (pos_weight if labels[v] == 1 else 1.0) for v in targets}
        loss = sum(weights[v] * _bce(probs[v], labels[v])
                   for v in targets) / B

        # Output layer grads.
        dh2 = [[0.0] * self.hidden_dim for _ in range(graph.n)]
        for v in targets:
            d_logit = weights[v] * (probs[v] - labels[v]) / B
            for j in range(self.hidden_dim):
                self.gw_out[j] += d_logit * h2[v][j]
            self.gb_out += d_logit
            dh2[v] = _scale(self.w_out, d_logit)

        # Backprop through layer 2, then layer 1 (features are leaves).
        dh1 = self.layer2.backward(dh2, cache2)
        self.layer1.backward(dh1, cache["cache1"])

        # SGD update.
        self.layer1.step(self.lr)
        self.layer2.step(self.lr)
        for j in range(self.hidden_dim):
            self.w_out[j] -= self.lr * self.gw_out[j]
        self.b_out -= self.lr * self.gb_out
        return loss

    # -- checkpointing ------------------------------------------------------

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

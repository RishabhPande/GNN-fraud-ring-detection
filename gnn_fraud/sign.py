"""SIGN: Scalable Inception Graph Neural Networks (Frasca et al., 2020),
from scratch.

SIGN asks the same question SGC does - how much of a GNN is the graph,
and how much is the learned transform - but answers it the other way.
Where SGC throws the hidden layer away and keeps only the last hop,
SIGN keeps every hop and lets a small MLP decide which ones matter::

    Z = [X || S X || S^2 X || ... || S^K X],   S = D_hat^{-1/2} A_hat D_hat^{-1/2}
    p_v = sigmoid(w2 @ relu(W1 @ Z_v + b1) + b2)

The propagation is parameter-free preprocessing (the same symmetric
normalization :mod:`gnn_fraud.sgc` uses, reused here through
:func:`gnn_fraud.sgc.propagate_features`), so it is computed once per
graph and never backpropagated through. All learning lives in the MLP
on top, which is exactly the point: on a graph where the signal sits at
a specific hop distance, the head can weight that block up and the
others down, while SGC's single linear head over ``S^K X`` alone cannot
tell the hops apart. If SIGN beats SGC here, hop identity matters; if
it ties, the last hop already carried everything, and the extra
parameters bought nothing.

Two details worth naming. First, the paper's full version uses several
different diffusion operators (normalized adjacency, PPR, triangle-based)
and a separate MLP per operator; this repo's version uses one operator,
the SGC normalization, at hops 0..K, with a single shared MLP - the
inception is over hop distance, not over operator type. Second, the
concatenation keeps the raw features (hop 0) in the input, so unlike
GCN the model can always fall back to a node's own features when the
neighborhood is noise. All linear algebra is hand-rolled
lists-of-floats -- no numpy, no torch.
"""

from __future__ import annotations

import copy
import math
import random

from gnn_fraud.sgc import propagate_features


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _bce(p: float, y: int) -> float:
    p = min(max(p, 1e-7), 1.0 - 1e-7)
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


def sign_features(graph, hops: int) -> list[list[float]]:
    """Concatenate the 0..``hops``-hop propagated features per node::

        Z_v = [X_v || (S X)_v || ... || (S^hops X)_v]

    Each block is ``propagate_features(graph, k)`` for one k, so the
    hop-0 block is the raw features and the hop-k block is the SGC
    propagation at that depth. ``hops=0`` returns the raw features
    unchanged (a copy), so the model degrades exactly to an MLP on node
    features. Deterministic and parameter-free.
    """
    if hops < 0:
        raise ValueError("hops must be non-negative")
    blocks = [propagate_features(graph, k) for k in range(hops + 1)]
    return [
        [x for block in blocks for x in block[v]]
        for v in range(graph.n)
    ]


class SIGN:
    """SIGN: multi-hop concatenated features, then a two-layer MLP head.

    ``predict`` takes a graph and target node ids and returns
    ``{v: probability}``, matching the other models' interface, so the
    shared training/evaluation protocol applies unchanged.
    """

    def __init__(
        self,
        in_dim: int,
        hops: int = 2,
        hidden_dim: int = 16,
        lr: float = 0.5,
        seed: int = 0,
    ):
        if hops < 0:
            raise ValueError("hops must be non-negative")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        rng = random.Random(seed)
        self.in_dim = in_dim
        self.hops = hops
        self.hidden_dim = hidden_dim
        self.lr = lr
        concat_dim = (hops + 1) * in_dim
        scale1 = math.sqrt(2.0 / (concat_dim + hidden_dim))
        scale2 = math.sqrt(2.0 / (hidden_dim + 1))
        self.W1 = [
            [rng.uniform(-scale1, scale1) for _ in range(concat_dim)]
            for _ in range(hidden_dim)
        ]
        self.b1 = [0.0] * hidden_dim
        self.W2 = [rng.uniform(-scale2, scale2) for _ in range(hidden_dim)]
        self.b2 = 0.0
        self.zero_grad()

    def zero_grad(self) -> None:
        concat_dim = (self.hops + 1) * self.in_dim
        self.gW1 = [[0.0] * concat_dim for _ in range(self.hidden_dim)]
        self.gb1 = [0.0] * self.hidden_dim
        self.gW2 = [0.0] * self.hidden_dim
        self.gb2 = 0.0

    # -- forward ----------------------------------------------------------

    def forward(self, graph, targets: list[int]):
        """Returns (probs dict, cache) with probs only for targets."""
        Z = sign_features(graph, self.hops)
        probs: dict[int, float] = {}
        cache: dict = {"Z": Z, "targets": list(targets), "hidden": {}}
        for v in targets:
            z = Z[v]
            pre = [
                sum(w * x for w, x in zip(self.W1[i], z)) + self.b1[i]
                for i in range(self.hidden_dim)
            ]
            h = [x if x > 0.0 else 0.0 for x in pre]
            logit = sum(w * x for w, x in zip(self.W2, h)) + self.b2
            probs[v] = _sigmoid(logit)
            cache["hidden"][v] = (pre, h)
        return probs, cache

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        """Inference -- propagation is deterministic, so this is exact."""
        probs, _ = self.forward(graph, targets)
        return probs

    # -- training step -----------------------------------------------------

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    pos_weight: float = 1.0) -> float:
        """One SGD step on ``targets``. Returns mean weighted BCE loss.

        Plain backprop through the two-layer head: ``d_logit`` flows
        into W2/b2, then through the ReLU mask into W1/b1. Gradients
        accumulate over the batch and are divided by the batch size,
        then a plain SGD step. ``lr=0.0`` accumulates gradients without
        moving the weights, which the numeric gradient check relies on.
        """
        self.zero_grad()
        probs, cache = self.forward(graph, targets)
        Z = cache["Z"]
        B = len(targets)
        weights = {v: (pos_weight if labels[v] == 1 else 1.0)
                   for v in targets}
        loss = sum(weights[v] * _bce(probs[v], labels[v])
                   for v in targets) / B
        for v in targets:
            d_logit = weights[v] * (probs[v] - labels[v]) / B
            pre, h = cache["hidden"][v]
            z = Z[v]
            # Head: logit = W2 . h + b2.
            for i in range(self.hidden_dim):
                self.gW2[i] += d_logit * h[i]
            self.gb2 += d_logit
            # Hidden layer: h = relu(W1 @ z + b1).
            for i in range(self.hidden_dim):
                if pre[i] <= 0.0:
                    continue
                d_pre = d_logit * self.W2[i]
                for j in range(len(z)):
                    self.gW1[i][j] += d_pre * z[j]
                self.gb1[i] += d_pre
        for i in range(self.hidden_dim):
            for j in range(len(self.W1[i])):
                self.W1[i][j] -= self.lr * self.gW1[i][j]
            self.b1[i] -= self.lr * self.gb1[i]
        for i in range(self.hidden_dim):
            self.W2[i] -= self.lr * self.gW2[i]
        self.b2 -= self.lr * self.gb2
        return loss

    # -- checkpointing ------------------------------------------------------

    def state_dict(self):
        return {
            "W1": copy.deepcopy(self.W1),
            "b1": list(self.b1),
            "W2": list(self.W2),
            "b2": self.b2,
        }

    def load_state_dict(self, state) -> None:
        self.W1 = copy.deepcopy(state["W1"])
        self.b1 = list(state["b1"])
        self.W2 = list(state["W2"])
        self.b2 = state["b2"]
        self.zero_grad()

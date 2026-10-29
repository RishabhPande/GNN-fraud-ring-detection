"""Simplified Graph Convolution (SGC, Wu et al., 2019), from scratch.

SGC asks how much of a GCN is actually the graph, and how much is the
nonlinearity. Take a two-layer GCN, delete the ReLU between its layers,
and the two weight matrices collapse into one: the whole model becomes
linear feature propagation followed by logistic regression::

    X_hat = S^K X,   S = D_hat^{-1/2} A_hat D_hat^{-1/2},   A_hat = A + I
    p_v   = sigmoid(w @ X_hat_v + b)

That is the entire model. There is no hidden layer, no per-layer
backprop, and no nonlinearity to tune: propagation is a fixed,
parameter-free preprocessing step (``propagate_features``), and training
is ordinary logistic regression on the propagated features, with the
same weighted BCE and early-stopping protocol as the other models in
this repo.

Why it belongs here: it is the cleanest ablation of the GCN in
:mod:`gnn_fraud.gcn`. If SGC matches GCN on the dense ring, the ring's
signal lives in the propagated features themselves, not in a learned
nonlinear transform of them -- and the cheaper, faster, easier-to-audit
model is the right one to ship. If it lags, the gap measures exactly
what the hidden layer buys. All linear algebra is hand-rolled
lists-of-floats -- no numpy, no torch.
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


def propagate_features(graph, hops: int) -> list[list[float]]:
    """Apply ``hops`` rounds of symmetric-normalized propagation to the
    graph's node features: ``X <- S X`` with
    ``S = D_hat^{-1/2} A_hat D_hat^{-1/2}`` and ``A_hat = A + I``, the
    same normalization :mod:`gnn_fraud.gcn` uses.

    ``hops=0`` returns a copy of the raw features, so the propagated
    model degrades exactly to logistic regression on node features.
    Propagation is deterministic and parameter-free, so it can be
    precomputed once per graph; it is recomputed per call here to keep
    the model stateless with respect to the graph.
    """
    if hops < 0:
        raise ValueError("hops must be non-negative")
    d_hat = [graph.degree(v) + 1 for v in range(graph.n)]
    weights: list[list[tuple[int, float]]] = []
    for v in range(graph.n):
        row = []
        for u in sorted(set(graph.neighbors(v)) | {v}):
            row.append((u, 1.0 / math.sqrt(d_hat[v] * d_hat[u])))
        weights.append(row)
    X = [list(row) for row in graph.features]
    for _ in range(hops):
        nxt: list[list[float]] = []
        for v in range(graph.n):
            acc = [0.0] * graph.feature_dim
            for u, c in weights[v]:
                xu = X[u]
                for j in range(graph.feature_dim):
                    acc[j] += c * xu[j]
            nxt.append(acc)
        X = nxt
    return X


class SimpleGraphConvolution:
    """SGC: K-hop normalized propagation, then a single linear head.

    ``predict`` takes a graph and target node ids and returns
    ``{v: probability}``, matching the other models' interface, so the
    shared training/evaluation protocol applies unchanged.
    """

    def __init__(
        self,
        in_dim: int,
        hops: int = 2,
        lr: float = 0.5,
        seed: int = 0,
    ):
        if hops < 0:
            raise ValueError("hops must be non-negative")
        rng = random.Random(seed)
        scale = math.sqrt(2.0 / (in_dim + 1))
        self.in_dim = in_dim
        self.hops = hops
        self.lr = lr
        self.w = [rng.uniform(-scale, scale) for _ in range(in_dim)]
        self.b = 0.0
        self.gw = [0.0] * in_dim
        self.gb = 0.0

    # -- forward ----------------------------------------------------------

    def forward(self, graph, targets: list[int]):
        """Returns (probs dict, cache) with probs only for targets."""
        X = propagate_features(graph, self.hops)
        probs: dict[int, float] = {}
        for v in targets:
            z = sum(w * x for w, x in zip(self.w, X[v])) + self.b
            probs[v] = _sigmoid(z)
        return probs, {"X": X, "targets": list(targets)}

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        """Inference -- propagation is deterministic, so this is exact."""
        probs, _ = self.forward(graph, targets)
        return probs

    # -- training step -----------------------------------------------------

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    pos_weight: float = 1.0) -> float:
        """One SGD step on ``targets``. Returns mean weighted BCE loss.

        The gradient is logistic regression's: ``d_logit = w * (p - y)``,
        accumulated over the batch and divided by the batch size, then
        a plain SGD step. ``lr=0.0`` accumulates gradients without
        moving the weights, which the numeric gradient check relies on.
        """
        self.gw = [0.0] * self.in_dim
        self.gb = 0.0
        probs, cache = self.forward(graph, targets)
        X = cache["X"]
        B = len(targets)
        weights = {v: (pos_weight if labels[v] == 1 else 1.0)
                   for v in targets}
        loss = sum(weights[v] * _bce(probs[v], labels[v])
                   for v in targets) / B
        for v in targets:
            d_logit = weights[v] * (probs[v] - labels[v]) / B
            for j in range(self.in_dim):
                self.gw[j] += d_logit * X[v][j]
            self.gb += d_logit
        for j in range(self.in_dim):
            self.w[j] -= self.lr * self.gw[j]
        self.b -= self.lr * self.gb
        return loss

    # -- checkpointing ------------------------------------------------------

    def state_dict(self):
        return {"w": copy.deepcopy(self.w), "b": self.b}

    def load_state_dict(self, state) -> None:
        self.w = copy.deepcopy(state["w"])
        self.b = state["b"]
        self.gw = [0.0] * self.in_dim
        self.gb = 0.0

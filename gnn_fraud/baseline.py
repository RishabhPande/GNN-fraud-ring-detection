"""MLP-on-node-features baseline (no graph) -- pure Python, no dependencies.

Same training protocol as the GNN (mini-batches, validation, early
stopping) so the comparison is apples-to-apples. Because the generator
camouflages ring members' features, this baseline should struggle while
the GNN -- which sees the wiring -- does not. That gap is the point.
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


class MLP:
    """One hidden layer, ReLU, sigmoid output, BCE + SGD."""

    def __init__(self, in_dim: int, hidden_dim: int = 16,
                 lr: float = 0.1, seed: int = 0):
        rng = random.Random(seed)
        scale = math.sqrt(2.0 / (in_dim + hidden_dim))
        self.W1 = [[rng.uniform(-scale, scale) for _ in range(in_dim)]
                   for _ in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim
        sw = math.sqrt(2.0 / (hidden_dim + 1))
        self.w2 = [rng.uniform(-sw, sw) for _ in range(hidden_dim)]
        self.b2 = 0.0
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim

    def forward(self, x: list[float]):
        pre = [sum(w * xi for w, xi in zip(row, x)) + b
               for row, b in zip(self.W1, self.b1)]
        h = [p if p > 0 else 0.0 for p in pre]
        z = sum(w * hi for w, hi in zip(self.w2, h)) + self.b2
        return _sigmoid(z), (x, pre, h)

    def predict(self, X: list[list[float]]) -> list[float]:
        return [self.forward(x)[0] for x in X]

    def train_step(self, X: list[list[float]], y: list[int],
                   pos_weight: float = 1.0) -> float:
        B = len(X)
        # forward
        outs = [self.forward(x) for x in X]
        loss = 0.0
        for (p, _), yi in zip(outs, y):
            pc = min(max(p, 1e-7), 1.0 - 1e-7)
            w = pos_weight if yi == 1 else 1.0
            loss += w * (-(yi * math.log(pc) + (1 - yi) * math.log(1.0 - pc)))
        loss /= B
        # backward
        gW1 = [[0.0] * self.in_dim for _ in range(self.hidden_dim)]
        gb1 = [0.0] * self.hidden_dim
        gw2 = [0.0] * self.hidden_dim
        gb2 = 0.0
        for (p, (x, pre, h)), yi in zip(outs, y):
            w = pos_weight if yi == 1 else 1.0
            d_z = w * (p - yi) / B
            for j in range(self.hidden_dim):
                gw2[j] += d_z * h[j]
            gb2 += d_z
            for j in range(self.hidden_dim):
                d_pre = d_z * self.w2[j] * (1.0 if pre[j] > 0 else 0.0)
                if d_pre == 0.0:
                    continue
                gb1[j] += d_pre
                for k in range(self.in_dim):
                    gW1[j][k] += d_pre * x[k]
        # sgd
        for j in range(self.hidden_dim):
            for k in range(self.in_dim):
                self.W1[j][k] -= self.lr * gW1[j][k]
            self.b1[j] -= self.lr * gb1[j]
            self.w2[j] -= self.lr * gw2[j]
        self.b2 -= self.lr * gb2
        return loss

    def state_dict(self):
        return copy.deepcopy({"W1": self.W1, "b1": self.b1,
                              "w2": self.w2, "b2": self.b2})

    def load_state_dict(self, state) -> None:
        self.W1 = copy.deepcopy(state["W1"])
        self.b1 = copy.deepcopy(state["b1"])
        self.w2 = copy.deepcopy(state["w2"])
        self.b2 = copy.deepcopy(state["b2"])


def train_mlp(
    features: list[list[float]],
    labels: list[int],
    train_ids: list[int],
    val_ids: list[int],
    hidden_dim: int = 16,
    lr: float = 0.1,
    batch_size: int = 64,
    epochs: int = 30,
    patience: int = 5,
    seed: int = 0,
    pos_weight: float | None = None,
) -> tuple[MLP, dict]:
    """Train the feature-only baseline with the same protocol as the GNN."""
    model = MLP(len(features[0]), hidden_dim=hidden_dim, lr=lr, seed=seed)
    rng = random.Random(seed)
    if pos_weight is None:
        # Same imbalance handling as the GNN: modest upweighting so the
        # baseline doesn't collapse to the base rate.
        pos_weight = 3.0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def _loss(ids: list[int]) -> float:
        X = [features[i] for i in ids]
        y = [labels[i] for i in ids]
        probs = model.predict(X)
        total = 0.0
        for p, yi in zip(probs, y):
            pc = min(max(p, 1e-7), 1.0 - 1e-7)
            total += -(yi * math.log(pc) + (1 - yi) * math.log(1.0 - pc))
        return total / len(ids)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    for _ in range(epochs):
        rng.shuffle(train_ids)
        bl = []
        for i in range(0, len(train_ids), batch_size):
            b = train_ids[i:i + batch_size]
            bl.append(model.train_step([features[j] for j in b],
                                       [labels[j] for j in b],
                                       pos_weight=pos_weight))
        history["train_loss"].append(sum(bl) / len(bl))
        vl = _loss(val_ids)
        history["val_loss"].append(vl)
        if vl < best_val - 1e-6:
            best_val = vl
            best_state = model.state_dict()
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history

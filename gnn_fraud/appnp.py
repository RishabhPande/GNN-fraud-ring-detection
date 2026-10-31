"""Approximate Personalized Propagation of Neural Predictions
(APPNP, Klicpera et al., 2019), from scratch.

APPNP splits prediction and propagation into two separate steps, which
is the opposite order from :mod:`gnn_fraud.sgc`:

* **Predict first.** A small MLP maps each node's own features to an
  initial fraud logit, with no graph involved at all::

      Z^(0)_v = w_out . relu(W1 @ x_v + b1) + b_out

* **Propagate the predictions.** The logits are then smoothed over the
  graph with personalized PageRank steps::

      Z^(k+1) = (1 - alpha) * S Z^(k) + alpha * Z^(0),
      S = D_hat^{-1/2} A_hat D_hat^{-1/2},   A_hat = A + I

  and the final score is ``sigmoid(Z^(K)_v)``.

The teleport term ``alpha * Z^(0)`` is the whole point. Plain
propagation (SGC's ``S^K X``) washes a node's own signal out as ``K``
grows: after enough steps every node in a connected component converges
toward the same smoothed value, which is oversmoothing in its purest
form. APPNP's teleport keeps a fixed fraction of each node's own
prediction at every step, so deep propagation (``K=10`` here, versus
SGC's 2) sharpens the neighborhood consensus without erasing the node
itself. A ring member's neighbors mostly agree it is fraud, so its
score climbs; a benign account next to the ring keeps its own benign
prediction pulling it back down, instead of being dragged up by the
pocket alone.

Backprop runs through the propagation in reverse: the gradient with
respect to ``Z^(0)`` accumulates ``alpha`` times the gradient at every
step (the teleport injects ``Z^(0)`` into each one) plus the gradient
that reaches ``Z^(0)`` as the propagation's starting point, and the
gradient with respect to ``Z^(k)`` is ``(1 - alpha)`` times the
symmetric normalized adjacency applied to the next step's gradient.
The MLP backward is ordinary: one hidden ReLU layer, then the output
head.
All linear algebra is hand-rolled lists-of-floats -- no numpy, no torch.
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


def _norm_weights(graph) -> list[list[tuple[int, float]]]:
    """Symmetric normalization weights, the same ``S`` that
    :mod:`gnn_fraud.gcn` and :mod:`gnn_fraud.sgc` use: for node ``v``,
    the list of ``(u, c_vu)`` over ``u in N(v) u {v}`` with
    ``c_vu = 1 / sqrt(d_hat_v * d_hat_u)``, ``d_hat = degree + 1``."""
    d_hat = [graph.degree(v) + 1 for v in range(graph.n)]
    weights: list[list[tuple[int, float]]] = []
    for v in range(graph.n):
        row = []
        for u in sorted(set(graph.neighbors(v)) | {v}):
            row.append((u, 1.0 / math.sqrt(d_hat[v] * d_hat[u])))
        weights.append(row)
    return weights


def propagate_predictions(graph, z0: list[float], steps: int,
                          alpha: float) -> list[float]:
    """Run ``steps`` personalized PageRank steps on initial logits
    ``z0``: ``Z <- (1 - alpha) * S Z + alpha * z0``.

    ``steps=0`` returns a copy of ``z0``; ``alpha=1`` returns ``z0``
    unchanged at any step count (every step teleports fully back);
    ``alpha=0`` is plain normalized propagation with no teleport.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if len(z0) != graph.n:
        raise ValueError("z0 must have one logit per node")
    weights = _norm_weights(graph)
    z = list(z0)
    for _ in range(steps):
        nxt = []
        for v in range(graph.n):
            acc = 0.0
            for u, c in weights[v]:
                acc += c * z[u]
            nxt.append((1.0 - alpha) * acc + alpha * z0[v])
        z = nxt
    return z


class APPNP:
    """APPNP: an MLP predictor followed by personalized PageRank
    propagation of its logits.

    ``predict`` takes a graph and target node ids and returns
    ``{v: probability}``, matching the other models' interface, so the
    shared training/evaluation protocol applies unchanged.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 8,
        steps: int = 10,
        alpha: float = 0.1,
        lr: float = 0.5,
        seed: int = 0,
    ):
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        rng = random.Random(seed)
        scale = math.sqrt(2.0 / (in_dim + hidden_dim))
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.steps = steps
        self.alpha = alpha
        self.lr = lr
        self.W1 = [
            [rng.uniform(-scale, scale) for _ in range(in_dim)]
            for _ in range(hidden_dim)
        ]
        self.b1 = [0.0] * hidden_dim
        out_scale = math.sqrt(2.0 / (hidden_dim + 1))
        self.w_out = [rng.uniform(-out_scale, out_scale)
                      for _ in range(hidden_dim)]
        self.b_out = 0.0
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW1 = [[0.0] * self.in_dim for _ in range(self.hidden_dim)]
        self.gb1 = [0.0] * self.hidden_dim
        self.gw_out = [0.0] * self.hidden_dim
        self.gb_out = 0.0

    # -- forward ----------------------------------------------------------

    def forward(self, graph, targets: list[int]):
        """Forward pass over the whole graph. Returns (probs dict,
        cache) with probs only for the requested targets. The cache
        holds everything backprop needs: the norm weights, the MLP
        pre-activations and hidden states, the initial logits, and the
        propagated logits at every step."""
        weights = _norm_weights(graph)
        pre1 = []
        hidden = []
        z0 = []
        for v in range(graph.n):
            x = graph.features[v]
            pre = [
                sum(w * xi for w, xi in zip(row, x)) + b
                for row, b in zip(self.W1, self.b1)
            ]
            h = [p if p > 0.0 else 0.0 for p in pre]
            pre1.append(pre)
            hidden.append(h)
            z0.append(sum(w * hv for w, hv in zip(self.w_out, h))
                      + self.b_out)
        trajectory = [list(z0)]
        z = list(z0)
        for _ in range(self.steps):
            nxt = []
            for v in range(graph.n):
                acc = 0.0
                for u, c in weights[v]:
                    acc += c * z[u]
                nxt.append((1.0 - self.alpha) * acc + self.alpha * z0[v])
            z = nxt
            trajectory.append(list(z))
        probs: dict[int, float] = {}
        for v in targets:
            probs[v] = _sigmoid(z[v])
        cache = {
            "weights": weights,
            "pre1": pre1,
            "hidden": hidden,
            "z0": z0,
            "trajectory": trajectory,
            "targets": list(targets),
        }
        return probs, cache

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        """Inference -- propagation is deterministic, so this is exact."""
        probs, _ = self.forward(graph, targets)
        return probs

    # -- training step -----------------------------------------------------

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    pos_weight: float = 1.0) -> float:
        """One SGD step on ``targets``. Returns mean weighted BCE loss.

        The forward pass embeds every node (a target's propagated score
        depends on its neighbors' initial predictions), but the loss
        and its seed gradient cover only ``targets``. Backprop first
        unwinds the propagation steps, then the MLP. ``lr=0.0``
        accumulates gradients without moving the weights, which the
        numeric gradient check relies on.
        """
        self.zero_grad()
        probs, cache = self.forward(graph, targets)
        weights = cache["weights"]
        hidden = cache["hidden"]
        pre1 = cache["pre1"]
        z0 = cache["z0"]

        B = len(targets)
        wts = {v: (pos_weight if labels[v] == 1 else 1.0)
               for v in targets}
        loss = sum(wts[v] * _bce(probs[v], labels[v])
                   for v in targets) / B

        # Seed gradient on the final propagated logits.
        d_final = [0.0] * graph.n
        for v in targets:
            d_final[v] = wts[v] * (probs[v] - labels[v]) / B

        # Unwind the propagation: Z^(0) collects alpha * d at every
        # step (the teleport injects it into each one), each step's
        # gradient flows back through (1-alpha)*S (symmetric, so the
        # reverse pass reuses the same weights), and after the last
        # unwind the remaining gradient is with respect to Z^(0)
        # itself, since Z^(0) is also the propagation's starting point.
        if self.steps == 0:
            d_z0 = list(d_final)
        else:
            d_z0 = [0.0] * graph.n
            d_next = list(d_final)
            for _ in range(self.steps):
                for v in range(graph.n):
                    if d_next[v] != 0.0:
                        d_z0[v] += self.alpha * d_next[v]
                d_cur = [0.0] * graph.n
                for v in range(graph.n):
                    if d_next[v] == 0.0:
                        continue
                    for u, c in weights[v]:
                        d_cur[u] += (1.0 - self.alpha) * c * d_next[v]
                d_next = d_cur
            for v in range(graph.n):
                d_z0[v] += d_next[v]

        # MLP backward, over every node: propagation mixed every node's
        # initial logit into the targets' scores, so every node carries
        # gradient even though the loss covered only the targets.
        for v in range(graph.n):
            dz = d_z0[v]
            if dz == 0.0:
                continue
            h = hidden[v]
            for j in range(self.hidden_dim):
                self.gw_out[j] += dz * h[j]
            self.gb_out += dz
            x = graph.features[v]
            for j in range(self.hidden_dim):
                d_pre = dz * self.w_out[j] * (1.0 if pre1[v][j] > 0.0
                                              else 0.0)
                if d_pre == 0.0:
                    continue
                for k in range(self.in_dim):
                    self.gW1[j][k] += d_pre * x[k]
                self.gb1[j] += d_pre

        # SGD update.
        for j in range(self.hidden_dim):
            for k in range(self.in_dim):
                self.W1[j][k] -= self.lr * self.gW1[j][k]
            self.b1[j] -= self.lr * self.gb1[j]
            self.w_out[j] -= self.lr * self.gw_out[j]
        self.b_out -= self.lr * self.gb_out
        return loss

    # -- checkpointing ------------------------------------------------------

    def state_dict(self):
        return {
            "W1": copy.deepcopy(self.W1),
            "b1": copy.deepcopy(self.b1),
            "w_out": copy.deepcopy(self.w_out),
            "b_out": self.b_out,
        }

    def load_state_dict(self, state) -> None:
        self.W1 = copy.deepcopy(state["W1"])
        self.b1 = copy.deepcopy(state["b1"])
        self.w_out = copy.deepcopy(state["w_out"])
        self.b_out = state["b_out"]
        self.zero_grad()

"""Principal Neighbourhood Aggregation (PNA, Corso et al., 2020), from
scratch.

Every other aggregator in this repo bets on one summary of the
neighborhood: SAGE takes the mean, GIN the sum, the pool aggregator
the max, GCN a degree-normalized sum. Each throws away something the
others keep -- mean loses count, sum loses scale, max loses everything
but the loudest neighbor. PNA's argument is that no single aggregator
can be injective over neighbor multisets the way the Weisfeiler-Lehman
test needs, so it uses several at once and lets the model combine
them.

One PNA layer reads, for each node ``v`` with neighbors ``N(v)``:

* four aggregators over the neighbor embeddings: ``mean``, ``max``,
  ``min``, and population ``std`` (the paper's set; ``sum`` is left
  out here because GIN already covers it, and the four here are the
  ones that disagree most about what a neighborhood *is* -- its
  center, its loudest member, its quietest member, its spread).
* three degree scalers applied to every aggregator: identity (1),
  amplification ``log(1 + d) / delta`` and attenuation
  ``delta / log(1 + d)``, where ``d`` is the node's degree and
  ``delta`` is the average degree over the training graph (the paper's
  normalizer, so a hub's amplification is ~1 on average instead of
  growing with the graph). The scalers let the same neighborhood
  summary count differently for a hub than for a leaf: a ring member
  with twelve suspicious neighbors should not be read the same way
  as a background account with two.

The twelve scaled towers are concatenated with the node's own
embedding (the paper's update function ``U``) and passed through one
linear layer + ReLU::

    h'_v = relu(W [h_v || towers_v] + b)

with ``W`` sized ``out_dim x 13 * in_dim``. Backprop routes through
each tower separately: mean spreads its gradient evenly, max/min send
theirs to the argmax/argmin neighbor only (first index wins ties, as
in the pool aggregator), and std sends
``d * (x_i - mean) / (n * std)`` to each neighbor, which is zero when
the neighborhood is constant. An isolated node has no neighbors, so
all four aggregators are zero and the scalers fall back to identity
(the attenuation ``delta / log(1)`` would divide by zero); the layer
then reads only the node's own features.

Like GIN and GCN there is no neighbor sampling: the std aggregator
and the degree scalers need the true neighborhood, so training runs
full-batch over the whole graph (``train_batch`` still scores only
the target subset, but the forward pass computes every node's
embedding). All linear algebra is hand-rolled lists-of-floats -- no
numpy, no torch.
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


# The aggregators and scalers, in the fixed order the towers are
# concatenated: for each scaler, the four aggregators in turn.
AGGREGATORS = ("mean", "max", "min", "std")
SCALERS = ("identity", "amplification", "attenuation")


def _scalers(degree: int, delta: float) -> list[float]:
    """Degree scalers for one node: [identity, amplification,
    attenuation]. A degree-0 node falls back to identity on all three
    (log(1) is 0, so amplification is 0 and attenuation would divide
    by zero); its aggregators are all zero anyway."""
    if degree == 0:
        return [1.0, 0.0, 1.0]
    log1p = math.log1p(degree)
    return [1.0, log1p / delta, delta / log1p]


def _aggregate(neighbors: list[list[float]], dim: int) -> tuple[
        list[float], list[float], list[float], list[float],
        list[int], list[int]]:
    """The four aggregators over one neighborhood (a list of embedding
    rows). Returns (mean, max, min, std, argmax_idx, argmin_idx),
    where the argmax/argmin lists hold the neighbor position (not id)
    that wins each dimension. An empty neighborhood returns zeros."""
    if not neighbors:
        zeros = [0.0] * dim
        return zeros, list(zeros), list(zeros), list(zeros), [], []
    n = len(neighbors)
    mean = [sum(row[j] for row in neighbors) / n for j in range(dim)]
    maxv = []
    minv = []
    argmax = []
    argmin = []
    for j in range(dim):
        best_max = neighbors[0][j]
        best_min = neighbors[0][j]
        i_max = 0
        i_min = 0
        for i in range(1, n):
            if neighbors[i][j] > best_max:
                best_max = neighbors[i][j]
                i_max = i
            if neighbors[i][j] < best_min:
                best_min = neighbors[i][j]
                i_min = i
        maxv.append(best_max)
        minv.append(best_min)
        argmax.append(i_max)
        argmin.append(i_min)
    std = []
    for j in range(dim):
        m = mean[j]
        var = sum((row[j] - m) ** 2 for row in neighbors) / n
        std.append(math.sqrt(var))
    return mean, maxv, minv, std, argmax, argmin


# ---------------------------------------------------------------------------
# one PNA layer
# ---------------------------------------------------------------------------


class PNALayer:
    """Single PNA layer: four aggregators x three degree scalers over
    the true neighborhood, concatenated with the node's own embedding,
    then one linear layer + ReLU.

    ``delta`` is the average degree the scalers normalize against
    (the paper's ``delta``: mean degree over the training graph).
    ``delta <= 0`` is rejected: the attenuation scaler divides by it.
    """

    def __init__(self, in_dim: int, out_dim: int, rng: random.Random,
                 delta: float = 1.0):
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.delta = delta
        self.n_towers = len(SCALERS) * len(AGGREGATORS)
        self.z_dim = in_dim + self.n_towers * in_dim
        scale = math.sqrt(2.0 / (self.z_dim + out_dim))
        self.W = [
            [rng.uniform(-scale, scale) for _ in range(self.z_dim)]
            for _ in range(out_dim)
        ]
        self.b = [0.0] * out_dim
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW = [[0.0] * self.z_dim for _ in range(self.out_dim)]
        self.gb = [0.0] * self.out_dim

    def forward(self, graph, H: list[list[float]]):
        """Returns (H_out, cache) where cache holds everything backprop
        needs: (H, per-node neighbor lists, per-node aggregator values
        and argmax/argmin positions, per-node scalers, z, pre)."""
        n = graph.n
        nbrs = [graph.neighbors(v) for v in range(n)]
        z = []
        pre = []
        cache_aggs = []
        cache_scalers = []
        for v in range(n):
            rows = [H[u] for u in nbrs[v]]
            mean, maxv, minv, std, argmax, argmin = _aggregate(
                rows, self.in_dim)
            scalers = _scalers(len(nbrs[v]), self.delta)
            towers = []
            for s in scalers:
                for agg in (mean, maxv, minv, std):
                    towers.extend(s * x for x in agg)
            zv = list(H[v]) + towers
            z.append(zv)
            pre.append([sum(w * xi for w, xi in zip(row, zv)) + b
                        for row, b in zip(self.W, self.b)])
            cache_aggs.append((mean, maxv, minv, std, argmax, argmin))
            cache_scalers.append(scalers)
        out = [_relu(p) for p in pre]
        return out, (H, nbrs, cache_aggs, cache_scalers, z, pre)

    def backward(self, d_out: list[list[float]], cache
                 ) -> list[list[float]]:
        """Backprop one layer. Returns ``d_H`` -- the gradient w.r.t.
        the input embeddings, indexed by node id. Each tower's
        gradient is unscaled by its degree scaler and routed through
        its aggregator: mean spreads evenly, max/min go to the winning
        neighbor only, std goes to every neighbor in proportion to its
        deviation from the neighborhood mean."""
        H, nbrs, cache_aggs, cache_scalers, z, pre = cache
        n = len(z)
        d_pre = [
            [d * (1.0 if p > 0.0 else 0.0) for d, p in zip(d_out[v], pre[v])]
            for v in range(n)
        ]
        for v in range(n):
            dv = d_pre[v]
            if not any(dv):
                continue
            for i in range(self.out_dim):
                di = dv[i]
                if di == 0.0:
                    continue
                row = self.W[i]
                grow = self.gW[i]
                zv = z[v]
                for j in range(self.z_dim):
                    grow[j] += di * zv[j]
                self.gb[i] += di
        d_H = [[0.0] * self.in_dim for _ in range(n)]
        for v in range(n):
            dv = d_pre[v]
            if not any(dv):
                continue
            d_z = _mat_t_vec(self.W, dv)
            # The node's own embedding is the first tower of z.
            for j in range(self.in_dim):
                d_H[v][j] += d_z[j]
            mean, _maxv, _minv, std, argmax, argmin = cache_aggs[v]
            scalers = cache_scalers[v]
            rows = [H[u] for u in nbrs[v]]
            deg = len(nbrs[v])
            if deg == 0:
                continue
            # Towers are ordered scaler-major: for each scaler, the
            # four aggregators in AGGREGATORS order.
            offset = self.in_dim
            for s in scalers:
                for agg_name in AGGREGATORS:
                    seg = d_z[offset:offset + self.in_dim]
                    offset += self.in_dim
                    if s == 0.0 or not any(seg):
                        continue
                    d_agg = [s * x for x in seg]
                    if agg_name == "mean":
                        for u in nbrs[v]:
                            row = d_H[u]
                            for j in range(self.in_dim):
                                row[j] += d_agg[j] / deg
                    elif agg_name == "max":
                        for j in range(self.in_dim):
                            u = nbrs[v][argmax[j]]
                            d_H[u][j] += d_agg[j]
                    elif agg_name == "min":
                        for j in range(self.in_dim):
                            u = nbrs[v][argmin[j]]
                            d_H[u][j] += d_agg[j]
                    else:  # std
                        for i, u in enumerate(nbrs[v]):
                            row = d_H[u]
                            hu = rows[i]
                            for j in range(self.in_dim):
                                if std[j] > 0.0:
                                    row[j] += (d_agg[j] * (hu[j] - mean[j])
                                               / (deg * std[j]))
        return d_H

    def step(self, lr: float) -> None:
        for i in range(self.out_dim):
            for j in range(self.z_dim):
                self.W[i][j] -= lr * self.gW[i][j]
            self.b[i] -= lr * self.gb[i]

    def state_dict(self):
        return {
            "W": copy.deepcopy(self.W),
            "b": copy.deepcopy(self.b),
            "delta": self.delta,
        }

    def load_state_dict(self, state) -> None:
        self.W = copy.deepcopy(state["W"])
        self.b = copy.deepcopy(state["b"])
        self.delta = state["delta"]
        self.zero_grad()


# ---------------------------------------------------------------------------
# two-layer model
# ---------------------------------------------------------------------------


class GraphPNA:
    """Two-layer PNA for binary node classification.

    Full-graph forward (no sampling -- see the module docstring),
    linear + sigmoid output head, weighted BCE loss like
    :class:`GraphSAGE`.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 8,
        lr: float = 0.1,
        seed: int = 0,
        delta: float = 1.0,
    ):
        self.rng = random.Random(seed)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.delta = delta
        self.layer1 = PNALayer(in_dim, hidden_dim, self.rng, delta=delta)
        self.layer2 = PNALayer(hidden_dim, hidden_dim, self.rng,
                               delta=delta)
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
        """Inference -- deterministic, no sampling anywhere in PNA."""
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
            dh2[v] = [w * d_logit for w in self.w_out]

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
            "delta": self.delta,
        }

    def load_state_dict(self, state) -> None:
        self.layer1.load_state_dict(state["layer1"])
        self.layer2.load_state_dict(state["layer2"])
        self.w_out = copy.deepcopy(state["w_out"])
        self.b_out = state["b_out"]
        self.delta = state["delta"]

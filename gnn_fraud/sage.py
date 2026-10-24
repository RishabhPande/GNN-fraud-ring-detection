"""GraphSAGE with mean, max-pool, or LSTM aggregation, 2 layers, from scratch.

Layer update for node ``v`` (mean mode)::

    h_v = relu(W_self @ x_v + W_agg @ mean({x_u : u in sampled_neighbors(v)}) + b)

Max-pool mode (the GraphSAGE paper's pooling aggregator) transforms each
neighbor first and keeps the loudest signal per dimension::

    h_v = relu(W_self @ x_v + W_agg @ max_u relu(W_pool @ x_u + b_pool) + b)

LSTM mode (the GraphSAGE paper's LSTM aggregator) feeds the neighbors
through an LSTM in a pseudo-random order and takes the final hidden state
as the aggregation::

    h_v = relu(W_self @ x_v + W_agg @ LSTM({x_u : u in sampled_neighbors(v)}) + b)

The LSTM is not permutation-invariant, so the paper applies it to a random
permutation of the neighbors. Here the permutation is seeded from the
neighbor vectors themselves: the same neighborhood always feeds the LSTM
in the same order (stable training, reproducible inference, and a fixed
loss surface for numeric gradient checks), while different neighborhoods
get different orders, as the paper intends.

A final linear layer + sigmoid maps the layer-2 embedding to a fraud
probability, trained with binary cross-entropy and full backprop through
both message-passing layers. Mini-batches use neighbor sampling: each
target node samples up to ``k`` neighbors per layer, so training cost is
bounded by ``O(batch * k^2)`` instead of the full 2-hop neighborhood.

All linear algebra is hand-rolled lists-of-floats -- no numpy, no torch.
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


def _outer(a: list[float], b: list[float]) -> list[list[float]]:
    return [[ai * bj for bj in b] for ai in a]


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


def _tanh_vec(v: list[float]) -> list[float]:
    return [math.tanh(x) for x in v]


def _lstm_order(h_neighs: list[list[float]]) -> list[int]:
    """Pseudo-random permutation of the neighbors, seeded from the neighbor
    vectors themselves (see the module docstring for why)."""
    order = list(range(len(h_neighs)))
    seed = hash(tuple(tuple(x) for x in h_neighs))
    random.Random(seed).shuffle(order)
    return order


def _mean_vec(vecs: list[list[float]], dim: int) -> list[float]:
    if not vecs:
        return [0.0] * dim
    m = len(vecs)
    return [sum(col) / m for col in zip(*vecs)]


def _bce(p: float, y: int) -> float:
    p = min(max(p, 1e-7), 1.0 - 1e-7)
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


# ---------------------------------------------------------------------------
# one message-passing layer
# ---------------------------------------------------------------------------


class SAGELayer:
    """Single GraphSAGE layer with ReLU: ``aggregator`` is ``"mean"``,
    ``"pool"`` (the GraphSAGE paper's max-pooling aggregator), or ``"lstm"``
    (the paper's LSTM aggregator: neighbors in pseudo-random order through
    an LSTM, final hidden state as the aggregation)."""

    # LSTM gates, in the order the cell equations use them.
    _GATES = ("i", "f", "o", "g")

    def __init__(self, in_dim: int, out_dim: int, rng: random.Random,
                 aggregator: str = "mean"):
        if aggregator not in ("mean", "pool", "lstm"):
            raise ValueError(
                "aggregator must be 'mean', 'pool' or 'lstm', "
                f"got {aggregator!r}")
        scale = math.sqrt(2.0 / (in_dim + out_dim))  # Xavier-ish init
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggregator = aggregator
        self.W_self = [
            [rng.uniform(-scale, scale) for _ in range(in_dim)]
            for _ in range(out_dim)
        ]
        self.W_agg = [
            [rng.uniform(-scale, scale) for _ in range(in_dim)]
            for _ in range(out_dim)
        ]
        # Max-pool mode (the GraphSAGE paper's pooling aggregator): each
        # neighbor is transformed through relu(W_pool @ x_u + b_pool)
        # before the elementwise max, so the layer can keep a single loud
        # tell instead of the neighborhood mean. He init on W_pool: it
        # feeds a ReLU, and a symmetric Xavier init leaves the transform
        # half-dead from the first forward pass.
        pool_scale = math.sqrt(2.0 / in_dim)
        self.W_pool = (
            [[rng.uniform(-pool_scale, pool_scale) for _ in range(in_dim)]
             for _ in range(in_dim)]
            if aggregator == "pool" else None
        )
        self.b_pool = [0.0] * in_dim if aggregator == "pool" else None
        # LSTM mode: one input matrix W, one recurrent matrix U, and one
        # bias per gate, all in_dim x in_dim so the final hidden state
        # has the same dimension as a neighbor vector (and W_agg keeps its
        # out_dim x in_dim shape). The forget-gate bias starts at 1.0 --
        # the standard init that keeps the cell from forgetting everything
        # before training starts.
        lstm_scale = math.sqrt(1.0 / in_dim)
        if aggregator == "lstm":
            self.lstm_W = {
                g: [[rng.uniform(-lstm_scale, lstm_scale)
                     for _ in range(in_dim)] for _ in range(in_dim)]
                for g in self._GATES
            }
            self.lstm_U = {
                g: [[rng.uniform(-lstm_scale, lstm_scale)
                     for _ in range(in_dim)] for _ in range(in_dim)]
                for g in self._GATES
            }
            self.lstm_b = {g: [0.0] * in_dim for g in self._GATES}
            self.lstm_b["f"] = [1.0] * in_dim
        else:
            self.lstm_W = None
            self.lstm_U = None
            self.lstm_b = None
        self.b = [0.0] * out_dim
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW_self = [[0.0] * self.in_dim for _ in range(self.out_dim)]
        self.gW_agg = [[0.0] * self.in_dim for _ in range(self.out_dim)]
        self.gW_pool = (
            [[0.0] * self.in_dim for _ in range(self.in_dim)]
            if self.aggregator == "pool" else None
        )
        self.gb_pool = (
            [0.0] * self.in_dim if self.aggregator == "pool" else None
        )
        if self.aggregator == "lstm":
            self.glstm_W = {
                g: [[0.0] * self.in_dim for _ in range(self.in_dim)]
                for g in self._GATES
            }
            self.glstm_U = {
                g: [[0.0] * self.in_dim for _ in range(self.in_dim)]
                for g in self._GATES
            }
            self.glstm_b = {g: [0.0] * self.in_dim for g in self._GATES}
        else:
            self.glstm_W = None
            self.glstm_U = None
            self.glstm_b = None
        self.gb = [0.0] * self.out_dim

    def _pre(self, h_self: list[float], agg: list[float]) -> list[float]:
        return _vec_add(
            _vec_add(_matvec(self.W_self, h_self), _matvec(self.W_agg, agg)),
            self.b,
        )

    def _pool_agg(self, h_neighs: list[list[float]]
                  ) -> tuple[list[float], list[list[float]], list[int]]:
        """Max-pool aggregation: relu(W_pool @ x_u + b_pool) per neighbor,
        then an elementwise max. Returns (agg, transformed, argmax)."""
        t = [_relu(_vec_add(_matvec(self.W_pool, x), self.b_pool))
             for x in h_neighs]
        if not t:
            return [0.0] * self.in_dim, [], [-1] * self.in_dim
        agg, argmax = [], []
        for i in range(self.in_dim):
            best, bi = t[0][i], 0
            for j in range(1, len(t)):
                if t[j][i] > best:
                    best, bi = t[j][i], j
            agg.append(best)
            argmax.append(bi)
        return agg, t, argmax

    def _lstm_agg(self, h_neighs: list[list[float]]):
        """LSTM aggregation: neighbors in pseudo-random order through the
        cell, final hidden state as the aggregation. Returns (agg, cache)
        where cache is (order, steps) -- one step tuple per neighbor, in
        feed order: (x, h_prev, c_prev, i, f, o, g, c_new, tanh_c_new)."""
        if not h_neighs:
            return [0.0] * self.in_dim, ([], [])
        order = _lstm_order(h_neighs)
        d = self.in_dim
        h = [0.0] * d
        c = [0.0] * d
        steps = []
        for idx in order:
            x = h_neighs[idx]
            gates = {}
            for g in self._GATES:
                pre = _vec_add(
                    _vec_add(_matvec(self.lstm_W[g], x),
                             _matvec(self.lstm_U[g], h)),
                    self.lstm_b[g],
                )
                gates[g] = ([math.tanh(v) for v in pre] if g == "g"
                            else [_sigmoid(v) for v in pre])
            i, f, o, gg = gates["i"], gates["f"], gates["o"], gates["g"]
            c_new = [f[k] * c[k] + i[k] * gg[k] for k in range(d)]
            tanh_c = _tanh_vec(c_new)
            h_new = [o[k] * tanh_c[k] for k in range(d)]
            steps.append((x, h, c, i, f, o, gg, c_new, tanh_c))
            h, c = h_new, c_new
        return h, (order, steps)

    def _lstm_backward(self, d_agg: list[float], cache
                       ) -> list[list[float]]:
        """Backprop through time from the final-hidden-state gradient.
        Returns per-neighbor gradients in the ORIGINAL neighbor order."""
        order, steps = cache
        d = self.in_dim
        d_shuffled: list[list[float]] = [[0.0] * d for _ in steps]
        dh_next = list(d_agg)
        dc_next = [0.0] * d
        for t in range(len(steps) - 1, -1, -1):
            x, h_prev, c_prev, i, f, o, gg, _, tanh_c = steps[t]
            d_o = [dh_next[k] * tanh_c[k] for k in range(d)]
            dc = [dh_next[k] * o[k] * (1.0 - tanh_c[k] ** 2) + dc_next[k]
                  for k in range(d)]
            d_i = [dc[k] * gg[k] for k in range(d)]
            d_f = [dc[k] * c_prev[k] for k in range(d)]
            d_g = [dc[k] * i[k] for k in range(d)]
            dc_prev = [dc[k] * f[k] for k in range(d)]
            pre = {
                "i": [d_i[k] * i[k] * (1.0 - i[k]) for k in range(d)],
                "f": [d_f[k] * f[k] * (1.0 - f[k]) for k in range(d)],
                "o": [d_o[k] * o[k] * (1.0 - o[k]) for k in range(d)],
                "g": [d_g[k] * (1.0 - gg[k] ** 2) for k in range(d)],
            }
            for g in self._GATES:
                gw, gu, gb = (self.glstm_W[g], self.glstm_U[g],
                              self.glstm_b[g])
                for k in range(d):
                    if pre[g][k] == 0.0:
                        continue
                    for j in range(d):
                        gw[k][j] += pre[g][k] * x[j]
                        gu[k][j] += pre[g][k] * h_prev[j]
                    gb[k] += pre[g][k]
            d_x = [0.0] * d
            dh_prev = [0.0] * d
            for g in self._GATES:
                d_x = _vec_add(d_x, _mat_t_vec(self.lstm_W[g], pre[g]))
                dh_prev = _vec_add(dh_prev,
                                   _mat_t_vec(self.lstm_U[g], pre[g]))
            d_shuffled[t] = d_x
            dh_next, dc_next = dh_prev, dc_prev
        d_neighs: list[list[float]] = [[0.0] * d for _ in order]
        for t, idx in enumerate(order):
            d_neighs[idx] = d_shuffled[t]
        return d_neighs

    def forward(self, h_self: list[float], h_neighs: list[list[float]]):
        """Returns (output, cache) where cache holds everything backprop needs."""
        if self.aggregator == "pool":
            agg, t, argmax = self._pool_agg(h_neighs)
            cache = (self._pre(h_self, agg), h_self, agg, t, argmax, h_neighs)
        elif self.aggregator == "lstm":
            agg, lstm_cache = self._lstm_agg(h_neighs)
            cache = (self._pre(h_self, agg), h_self, agg, lstm_cache)
        else:
            agg = _mean_vec(h_neighs, self.in_dim)
            cache = (self._pre(h_self, agg), h_self, agg, len(h_neighs))
        return _relu(cache[0]), cache

    def backward(self, d_out: list[float], cache
                 ) -> tuple[list[float], list[list[float]]]:
        """Backprop one step.

        Returns (d_h_self, d_neighs) -- the gradient w.r.t. the node's own
        input embedding, and per-neighbor gradients aligned with the
        forward pass's neighbor list. Mean mode splits the aggregation
        gradient evenly (bit-for-bit what the old (d_h_self, d_agg)
        contract produced); pool mode routes each dimension's gradient to
        its argmax neighbor, skipping dead ReLUs. Same contract as
        :meth:`TemporalSAGELayer.backward_temporal`.
        """
        pre, h_self, agg = cache[0], cache[1], cache[2]
        d_pre = [d * (1.0 if p > 0.0 else 0.0) for d, p in zip(d_out, pre)]
        for i in range(self.out_dim):
            di = d_pre[i]
            if di == 0.0:
                continue
            row_s = self.gW_self[i]
            row_a = self.gW_agg[i]
            for j in range(self.in_dim):
                row_s[j] += di * h_self[j]
                row_a[j] += di * agg[j]
            self.gb[i] += di
        d_h_self = _mat_t_vec(self.W_self, d_pre)
        d_agg = _mat_t_vec(self.W_agg, d_pre)
        if self.aggregator == "lstm":
            d_neighs = self._lstm_backward(d_agg, cache[3])
        elif self.aggregator == "pool":
            t, argmax, h_neighs = cache[3], cache[4], cache[5]
            d_neighs = [[0.0] * self.in_dim for _ in h_neighs]
            for i in range(self.in_dim):
                bi = argmax[i]
                if bi < 0 or t[bi][i] <= 0.0:
                    continue  # no winner, or a dead ReLU: no gradient flows
                g = d_agg[i]
                row_p = self.gW_pool[i]
                x = h_neighs[bi]
                self.gb_pool[i] += g
                for j in range(self.in_dim):
                    row_p[j] += g * x[j]
                    d_neighs[bi][j] += g * self.W_pool[i][j]
        else:
            m = cache[3]
            if m:
                share = _scale(d_agg, 1.0 / m)
                d_neighs = [list(share) for _ in range(m)]
            else:
                d_neighs = []
        return d_h_self, d_neighs

    def step(self, lr: float) -> None:
        for i in range(self.out_dim):
            for j in range(self.in_dim):
                self.W_self[i][j] -= lr * self.gW_self[i][j]
                self.W_agg[i][j] -= lr * self.gW_agg[i][j]
            self.b[i] -= lr * self.gb[i]
        if self.aggregator == "pool":
            for i in range(self.in_dim):
                for j in range(self.in_dim):
                    self.W_pool[i][j] -= lr * self.gW_pool[i][j]
                self.b_pool[i] -= lr * self.gb_pool[i]
        if self.aggregator == "lstm":
            for g in self._GATES:
                for i in range(self.in_dim):
                    for j in range(self.in_dim):
                        self.lstm_W[g][i][j] -= lr * self.glstm_W[g][i][j]
                        self.lstm_U[g][i][j] -= lr * self.glstm_U[g][i][j]
                    self.lstm_b[g][i] -= lr * self.glstm_b[g][i]

    def state_dict(self):
        state = {
            "W_self": self.W_self,
            "W_agg": self.W_agg,
            "b": self.b,
            "aggregator": self.aggregator,
        }
        if self.aggregator == "pool":
            state["W_pool"] = self.W_pool
            state["b_pool"] = self.b_pool
        if self.aggregator == "lstm":
            state["lstm_W"] = self.lstm_W
            state["lstm_U"] = self.lstm_U
            state["lstm_b"] = self.lstm_b
        return copy.deepcopy(state)

    def load_state_dict(self, state) -> None:
        # The checkpoint defines the model: a mean checkpoint (old or new
        # format) restores a mean layer, a pool checkpoint a pool layer,
        # an lstm checkpoint an lstm layer.
        self.aggregator = state.get("aggregator", "mean")
        self.W_self = copy.deepcopy(state["W_self"])
        self.W_agg = copy.deepcopy(state["W_agg"])
        self.b = copy.deepcopy(state["b"])
        if self.aggregator == "pool":
            self.W_pool = copy.deepcopy(state["W_pool"])
            self.b_pool = copy.deepcopy(state["b_pool"])
        else:
            self.W_pool = None
            self.b_pool = None
        if self.aggregator == "lstm":
            self.lstm_W = copy.deepcopy(state["lstm_W"])
            self.lstm_U = copy.deepcopy(state["lstm_U"])
            self.lstm_b = copy.deepcopy(state["lstm_b"])
        else:
            self.lstm_W = None
            self.lstm_U = None
            self.lstm_b = None
        self.zero_grad()


# ---------------------------------------------------------------------------
# two-layer model with neighbor sampling
# ---------------------------------------------------------------------------


class GraphSAGE:
    """Two-layer GraphSAGE for binary node classification.

    Args:
        aggregator: ``"mean"``, ``"pool"`` (max-pooling aggregator from
            the GraphSAGE paper), or ``"lstm"`` (the paper's LSTM
            aggregator), shared by both layers.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 8,
        lr: float = 0.1,
        seed: int = 0,
        aggregator: str = "mean",
    ):
        self.rng = random.Random(seed)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.aggregator = aggregator
        self.layer1 = SAGELayer(in_dim, hidden_dim, self.rng,
                                aggregator=aggregator)
        self.layer2 = SAGELayer(hidden_dim, hidden_dim, self.rng,
                                aggregator=aggregator)
        scale = math.sqrt(2.0 / (hidden_dim + 1))
        self.w_out = [self.rng.uniform(-scale, scale) for _ in range(hidden_dim)]
        self.b_out = 0.0
        self.gw_out = [0.0] * hidden_dim
        self.gb_out = 0.0

    # -- sampling --------------------------------------------------------

    def _sample(self, candidates: list[int], k: int | None) -> list[int]:
        if k is None or len(candidates) <= k:
            return list(candidates)
        return self.rng.sample(candidates, k)

    def _expand_batch(self, graph, targets: list[int], k1: int | None,
                      k2: int | None):
        """Sample the 2-hop computation graph for a mini-batch.

        Returns (s2, s1): s2[v] = layer-2 sampled neighbors of target v;
        s1[x] = layer-1 sampled neighbors for every node x whose layer-1
        embedding is needed (targets + their s2 nodes).
        """
        s2: dict[int, list[int]] = {}
        layer1_nodes: set[int] = set(targets)
        for v in targets:
            s2[v] = self._sample(graph.neighbors(v), k2)
            layer1_nodes.update(s2[v])
        s1: dict[int, list[int]] = {}
        for x in layer1_nodes:
            s1[x] = self._sample(graph.neighbors(x), k1)
        return s2, s1

    # -- forward ----------------------------------------------------------

    def forward(self, graph, targets: list[int],
                k1: int | None = 3, k2: int | None = 3):
        """Forward pass. Returns (probs dict, cache)."""
        s2, s1 = self._expand_batch(graph, targets, k1, k2)

        h1: dict[int, list[float]] = {}
        cache1: dict[int, tuple] = {}
        for x in s1:
            neighs = [graph.features[n] for n in s1[x]]
            h1[x], cache1[x] = self.layer1.forward(graph.features[x], neighs)

        h2: dict[int, list[float]] = {}
        cache2: dict[int, tuple] = {}
        for v in targets:
            neighs = [h1[u] for u in s2[v]]
            h2[v], cache2[v] = self.layer2.forward(h1[v], neighs)

        probs: dict[int, float] = {}
        logits: dict[int, float] = {}
        for v in targets:
            z = sum(w * h for w, h in zip(self.w_out, h2[v])) + self.b_out
            logits[v] = z
            probs[v] = _sigmoid(z)

        cache = {"s2": s2, "s1": s1, "h1": h1, "cache1": cache1,
                 "h2": h2, "cache2": cache2, "targets": list(targets)}
        return probs, cache

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        """Inference with full (unsampled) neighborhoods -- deterministic."""
        probs, _ = self.forward(graph, targets, k1=None, k2=None)
        return probs

    # -- training step -----------------------------------------------------

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    k1: int | None = 3, k2: int | None = 3,
                    pos_weight: float = 1.0) -> float:
        """One SGD step on a mini-batch. Returns mean weighted BCE loss.

        Args:
            pos_weight: multiplier on the loss of positive (fraud) examples.
                Fraud data is heavily imbalanced; without upweighting,
                the model collapses to predicting the base rate for
                everything. Standard practice in production fraud models.
        """
        self.layer1.zero_grad()
        self.layer2.zero_grad()
        self.gw_out = [0.0] * self.hidden_dim
        self.gb_out = 0.0

        probs, cache = self.forward(graph, targets, k1, k2)
        s2, s1 = cache["s2"], cache["s1"]
        h1, cache1 = cache["h1"], cache["cache1"]
        h2, cache2 = cache["h2"], cache["cache2"]

        B = len(targets)
        weights = {v: (pos_weight if labels[v] == 1 else 1.0) for v in targets}
        loss = sum(weights[v] * _bce(probs[v], labels[v])
                   for v in targets) / B

        # Output layer grads.
        dh2: dict[int, list[float]] = {}
        for v in targets:
            d_logit = weights[v] * (probs[v] - labels[v]) / B
            for j in range(self.hidden_dim):
                self.gw_out[j] += d_logit * h2[v][j]
            self.gb_out += d_logit
            dh2[v] = _scale(self.w_out, d_logit)

        # Backprop through layer 2; accumulate dh1 over layer-1 nodes.
        dh1: dict[int, list[float]] = {x: [0.0] * self.hidden_dim for x in s1}
        for v in targets:
            d_h_self, d_neighs = self.layer2.backward(dh2[v], cache2[v])
            dh1[v] = _vec_add(dh1[v], d_h_self)
            for u, d_h_u in zip(s2[v], d_neighs):
                dh1[u] = _vec_add(dh1[u], d_h_u)

        # Backprop through layer 1 (node features are leaves -- no need to
        # propagate into them).
        for x in s1:
            self.layer1.backward(dh1[x], cache1[x])

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

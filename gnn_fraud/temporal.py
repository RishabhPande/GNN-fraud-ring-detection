"""Temporal GraphSAGE: recency-weighted message passing over timestamped edges.

Fraud rings form fast -- the generator plants ring edges inside a short
burst window late in the timeline, while benign edges spread uniformly
over it. A temporal layer exploits that signature by weighting each
neighbor by recency::

    w_i = softmax(-Δt_i / τ),   Δt_i = t_now - t_(v,i),   τ = exp(log_tau)

so a ring member's aggregation concentrates on its burst-coordinated
neighbors, and the ring's weak tells denoise harder than under a plain
mean. The timescale ``τ`` is learned by backprop -- log-parametrized so
it stays positive -- and the hand-derived gradient is numeric-checked
in ``tests/test_temporal.py``.

All linear algebra is hand-rolled lists-of-floats, reusing the helpers
from :mod:`gnn_fraud.sage`. Pure stdlib, no numpy, no torch.
"""

from __future__ import annotations

import copy
import math
import random

from gnn_fraud.sage import (
    SAGELayer,
    _bce,
    _mat_t_vec,
    _matvec,
    _relu,
    _scale,
    _sigmoid,
    _vec_add,
)


def recency_weights(
    times: list[float], t_now: float, log_tau: float
) -> tuple[list[float], list[float]]:
    """Softmax recency weights over neighbor edge times.

    Args:
        times: timestamp of each neighbor's edge to the target node.
        t_now: reference "now" (the graph's latest timestamp).
        log_tau: log of the decay timescale; τ = exp(log_tau).

    Returns:
        (weights, deltas): normalized weights summing to 1, and the
        per-neighbor ages Δt_i = t_now - t_i (clamped ≥ 0).
    """
    tau = math.exp(log_tau)
    deltas = [max(0.0, t_now - t) for t in times]
    logits = [-d / tau for d in deltas]
    m = max(logits) if logits else 0.0
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps)
    weights = [e / s for e in exps] if s else []
    return weights, deltas


# ---------------------------------------------------------------------------
# one temporal message-passing layer
# ---------------------------------------------------------------------------


class TemporalSAGELayer(SAGELayer):
    """GraphSAGE layer with recency-weighted aggregation.

    Layer update for node ``v``::

        w_i  = softmax(-Δt_i / τ)          # Δt_i = t_now - t_(v,i)
        agg  = Σ_i w_i x_i
        h_v  = relu(W_self @ x_v + W_agg @ agg + b)

    ``τ = exp(log_tau)`` is a learnable timescale shared across the layer's
    neighborhoods.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rng: random.Random,
        log_tau: float = math.log(5.0),
    ):
        # Set before super().__init__, which calls self.zero_grad().
        self.log_tau = float(log_tau)
        super().__init__(in_dim, out_dim, rng)

    def zero_grad(self) -> None:
        super().zero_grad()
        self.g_log_tau = 0.0

    def forward_temporal(
        self,
        h_self: list[float],
        h_neighs: list[list[float]],
        times: list[float],
        t_now: float,
    ):
        """Returns (output, cache) where cache holds everything backprop needs."""
        weights, deltas = recency_weights(times, t_now, self.log_tau)
        if h_neighs:
            agg = [
                sum(w * hn[j] for w, hn in zip(weights, h_neighs))
                for j in range(self.in_dim)
            ]
        else:
            agg = [0.0] * self.in_dim
        pre = _vec_add(
            _vec_add(_matvec(self.W_self, h_self), _matvec(self.W_agg, agg)),
            self.b,
        )
        out = _relu(pre)
        cache = (pre, h_self, agg, weights, deltas, h_neighs, self.log_tau)
        return out, cache

    def backward_temporal(
        self, d_out: list[float], cache
    ) -> tuple[list[float], list[list[float]]]:
        """Backprop one step.

        Returns (d_h_self, d_h_neighs) -- gradients w.r.t. the node's own
        input embedding and w.r.t. each neighbor's embedding, plus the
        accumulated gradient for ``log_tau`` in ``self.g_log_tau``.
        """
        pre, h_self, agg, weights, deltas, h_neighs, log_tau = cache
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

        # Distribute the aggregation gradient across neighbors by weight.
        d_h_neighs = [_scale(d_agg, w) for w in weights]

        # Gradient for log_tau through the softmax. With
        #   w_i = exp(-Δt_i/τ) / Σ_j exp(-Δt_j/τ),   τ = exp(log_tau),
        # ∂w_i/∂log_tau = w_i (Δt_i - Δt̄) / τ,  Δt̄ = Σ_j w_j Δt_j,
        # and dL/dlog_tau = Σ_i (d_agg · h_i) ∂w_i/∂log_tau.
        if weights:
            tau = math.exp(log_tau)
            dbar = sum(w * d for w, d in zip(weights, deltas))
            for w, hn, delta in zip(weights, h_neighs, deltas):
                dL_dw = sum(da * h for da, h in zip(d_agg, hn))
                self.g_log_tau += dL_dw * w * (delta - dbar) / tau

        return d_h_self, d_h_neighs

    def step(self, lr: float) -> None:
        super().step(lr)
        self.log_tau -= lr * self.g_log_tau

    def state_dict(self):
        state = super().state_dict()
        state["log_tau"] = self.log_tau
        return state

    def load_state_dict(self, state) -> None:
        super().load_state_dict(state)
        self.log_tau = state["log_tau"]


# ---------------------------------------------------------------------------
# two-layer temporal model with neighbor sampling
# ---------------------------------------------------------------------------


class TemporalGraphSAGE:
    """Two-layer temporal GraphSAGE for binary node classification.

    Same scaffolding as :class:`gnn_fraud.sage.GraphSAGE` (neighbor-sampled
    mini-batches, sigmoid head, positive-class-weighted BCE), but both
    message-passing layers weight neighbors by recency. Requires a graph
    whose edges carry timestamps (see :meth:`gnn_fraud.graph.Graph.add_edge`
    and ``GraphGenerator.generate(timestamps=True)``).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 8,
        lr: float = 0.1,
        seed: int = 0,
        log_tau: float = math.log(5.0),
    ):
        self.rng = random.Random(seed)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.layer1 = TemporalSAGELayer(in_dim, hidden_dim, self.rng,
                                       log_tau=log_tau)
        self.layer2 = TemporalSAGELayer(hidden_dim, hidden_dim, self.rng,
                                       log_tau=log_tau)
        scale = math.sqrt(2.0 / (hidden_dim + 1))
        self.w_out = [self.rng.uniform(-scale, scale) for _ in range(hidden_dim)]
        self.b_out = 0.0
        self.gw_out = [0.0] * hidden_dim
        self.gb_out = 0.0

    @property
    def taus(self) -> list[float]:
        """Learned decay timescales (days) per layer."""
        return [math.exp(self.layer1.log_tau), math.exp(self.layer2.log_tau)]

    # -- sampling --------------------------------------------------------

    def _sample(self, candidates: list[int], k: int | None) -> list[int]:
        if k is None or len(candidates) <= k:
            return list(candidates)
        return self.rng.sample(candidates, k)

    def _edge_times(self, graph, center: int, nbrs: list[int]) -> list[float]:
        times = []
        for u in nbrs:
            t = graph.edge_time(center, u)
            if t is None:
                raise ValueError(
                    f"edge ({center}, {u}) has no timestamp; "
                    "TemporalGraphSAGE needs timestamped edges -- generate "
                    "with GraphGenerator(...).generate(timestamps=True)"
                )
            times.append(t)
        return times

    def _expand_batch(self, graph, targets: list[int], k1: int | None,
                      k2: int | None):
        """Sample the 2-hop computation graph, carrying edge timestamps.

        Returns (t_now, s2, t2, s1, t1): s2[v] / s1[x] are sampled neighbor
        ids and t2[v] / t1[x] their aligned edge timestamps.
        """
        t_now = graph.max_edge_time()
        if t_now is None:
            raise ValueError(
                "TemporalGraphSAGE needs timestamped edges; "
                "generate with GraphGenerator(...).generate(timestamps=True)"
            )
        s2: dict[int, list[int]] = {}
        t2: dict[int, list[float]] = {}
        layer1_nodes: set[int] = set(targets)
        for v in targets:
            s2[v] = self._sample(graph.neighbors(v), k2)
            t2[v] = self._edge_times(graph, v, s2[v])
            layer1_nodes.update(s2[v])
        s1: dict[int, list[int]] = {}
        t1: dict[int, list[float]] = {}
        for x in layer1_nodes:
            s1[x] = self._sample(graph.neighbors(x), k1)
            t1[x] = self._edge_times(graph, x, s1[x])
        return t_now, s2, t2, s1, t1

    # -- forward ----------------------------------------------------------

    def forward(self, graph, targets: list[int],
                k1: int | None = 3, k2: int | None = 3):
        """Forward pass. Returns (probs dict, cache)."""
        t_now, s2, t2, s1, t1 = self._expand_batch(graph, targets, k1, k2)

        h1: dict[int, list[float]] = {}
        cache1: dict[int, tuple] = {}
        for x in s1:
            neighs = [graph.features[n] for n in s1[x]]
            h1[x], cache1[x] = self.layer1.forward_temporal(
                graph.features[x], neighs, t1[x], t_now)

        h2: dict[int, list[float]] = {}
        cache2: dict[int, tuple] = {}
        for v in targets:
            neighs = [h1[u] for u in s2[v]]
            h2[v], cache2[v] = self.layer2.forward_temporal(
                h1[v], neighs, t2[v], t_now)

        probs: dict[int, float] = {}
        for v in targets:
            z = sum(w * h for w, h in zip(self.w_out, h2[v])) + self.b_out
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
        """One SGD step on a mini-batch. Returns mean weighted BCE loss."""
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
            d_h_self, d_h_neighs = self.layer2.backward_temporal(
                dh2[v], cache2[v])
            dh1[v] = _vec_add(dh1[v], d_h_self)
            for u, d_h_u in zip(s2[v], d_h_neighs):
                dh1[u] = _vec_add(dh1[u], d_h_u)

        # Backprop through layer 1 (node features are leaves).
        for x in s1:
            self.layer1.backward_temporal(dh1[x], cache1[x])

        # SGD update (layer steps also move each layer's log_tau).
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

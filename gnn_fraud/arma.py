"""ARMA-GNN: graph neural network with an AutoRegressive Moving
Average filter (Bianchi et al., 2021), from scratch.

The polynomial models in this repo (ChebNet, GPR-GNN, BernNet)
approximate a spectral filter with a finite sum of propagation
powers. A rational filter can express sharper responses with the
same depth, the way an IIR filter beats an FIR filter of the same
order on a sharp cutoff. ARMA builds one as a recursion: the state
feeds back into itself through the graph, so a finite iteration
count has an infinite impulse response in the limit::

    base_v  = W_in @ x_v                       (input term)
    Z_v^(0) = base_v + b
    M_v^(t) = sum_u S_hat[v, u] * Z_u^(t-1)    (normalized propagation)
    Z_v^(t) = M^(t) @ W_rec^T + base_v + b,    t = 1..T
    out_v   = relu(Z_v^(T))

where ``S_hat = D_hat^-1/2 A_hat D_hat^-1/2`` is the symmetric
normalized adjacency with self-loops, the GCN operator. The input
term is re-injected at every iteration: that is the moving-average
part, and it is also what stops the recursion washing the node's
own features out, the same job APPNP's teleport and GCNII's
initial residual do by hand. Here it falls out of the filter
definition instead. The recurrent weight ``W_rec`` is shared
across iterations, so ``iters`` is a receptive-field and filter-
sharpness knob, not a parameter-count knob: T iterations of a
rational filter reach responses a degree-T polynomial cannot.

The recursion itself is linear, as a filter should be: the ReLU
sits at the layer output, between layers, not inside the loop.
An activation inside the loop would make the response depend on
the signal's sign pattern at every step and, in this repo's
experiments, trains far worse (AUC 0.78-0.84 across schedules,
against 0.93-0.94 for the linear recursion on the same protocol).

Two layers of this recursion, then a linear sigmoid head. With
``iters=0`` a layer degrades exactly to a linear-plus-ReLU layer
over raw features (no graph at all), which the tests pin.

Backprop runs through the recursion in reverse: the gradient
w.r.t. the state at iteration t-1 is the normalized propagation
(which is symmetric, so its transpose is itself) of the recurrent
transpose applied to the pre-activation gradient at t, and the
input term collects gradient from every iteration, since it is
injected at every step (numeric-gradient-checked at the layer and
end to end, including W_rec and the input matrix). All linear
algebra is hand-rolled lists-of-floats, no numpy, no torch.
"""

from __future__ import annotations

import copy
import math
import random


def _matvec(W: list[list[float]], x: list[float]) -> list[float]:
    return [sum(w * xi for w, xi in zip(row, x)) for row in W]


def _mat_t_vec(W: list[list[float]], v: list[float]) -> list[float]:
    n_cols = len(W[0])
    return [sum(W[r][c] * v[r] for r in range(len(W))) for c in range(n_cols)]


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _bce(p: float, y: int) -> float:
    p = min(max(p, 1e-7), 1.0 - 1e-7)
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


def normalized_propagate(graph, H: list[list[float]]) -> list[list[float]]:
    """One step of ``S_hat @ H``: the symmetric normalized adjacency
    with self-loops (``D_hat^-1/2 A_hat D_hat^-1/2``), applied to a
    list of per-node vectors. Self-loops are included, so an
    isolated node's output is exactly its own vector (its hat
    degree is 1). The operator is symmetric, so backprop through it
    is this same function."""
    dim = len(H[0]) if H else 0
    out = []
    for v in range(graph.n):
        dv = math.sqrt(graph.degree(v) + 1)
        acc = [0.0] * dim
        for u in [v] + graph.neighbors(v):
            w = 1.0 / (dv * math.sqrt(graph.degree(u) + 1))
            hu = H[u]
            for j in range(dim):
                acc[j] += w * hu[j]
        out.append(acc)
    return out


class ARMALayer:
    """Single ARMA layer: a base linear transform re-injected at
    every iteration of a recurrent normalized propagation. Operates
    on the whole graph at once, so ``forward`` takes the full
    embedding list indexed by node id."""

    def __init__(self, in_dim: int, out_dim: int, iters: int,
                 rng: random.Random):
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("in_dim and out_dim must be positive")
        if iters < 0:
            raise ValueError("iters must be non-negative")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.iters = iters
        in_scale = math.sqrt(2.0 / (in_dim + out_dim))
        rec_scale = math.sqrt(2.0 / (out_dim + out_dim))
        self.W_in = [
            [rng.uniform(-in_scale, in_scale) for _ in range(in_dim)]
            for _ in range(out_dim)
        ]
        self.W_rec = [
            [rng.uniform(-rec_scale, rec_scale) for _ in range(out_dim)]
            for _ in range(out_dim)
        ]
        self.b = [0.0] * out_dim
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW_in = [[0.0] * self.in_dim for _ in range(self.out_dim)]
        self.gW_rec = [[0.0] * self.out_dim for _ in range(self.out_dim)]
        self.gb = [0.0] * self.out_dim

    def forward(self, graph, H: list[list[float]]):
        """Returns (H_out, cache). The cache holds the graph, the
        inputs, the base transform, and every iteration's propagated
        state and (linear) state, backprop's raw material. The ReLU
        is applied once, to the final state."""
        base = [_matvec(self.W_in, h) for h in H]
        states = []
        props = []
        state = [[base[v][j] + self.b[j] for j in range(self.out_dim)]
                 for v in range(graph.n)]
        states.append(state)
        for _ in range(self.iters):
            prop = normalized_propagate(graph, states[-1])
            props.append(prop)
            state = []
            for v in range(graph.n):
                rec = _matvec(self.W_rec, prop[v])
                state.append([rec[j] + base[v][j] + self.b[j]
                              for j in range(self.out_dim)])
            states.append(state)
        out = [[x if x > 0.0 else 0.0 for x in p] for p in state]
        cache = (graph, H, base, states, props)
        return out, cache

    def backward(self, d_out: list[list[float]], cache) -> list[list[float]]:
        """Backprop one layer in reverse through the recursion.
        Returns d_H, the gradient w.r.t. the input embeddings. The
        base transform is injected at every iteration, so its
        gradient is the sum of every iteration's pre-activation
        gradient; the recurrent gradient at iteration t routes back
        to state t-1 through the symmetric propagation."""
        graph, H, base, states, props = cache
        n = len(H)
        final = states[-1]
        # The only nonlinearity is the output ReLU over the final
        # state, so its mask applies once, here.
        d_state = [
            [d * (1.0 if p > 0.0 else 0.0)
             for d, p in zip(d_out[v], final[v])]
            for v in range(n)
        ]
        d_base = [[0.0] * self.out_dim for _ in range(n)]
        for t in range(self.iters, -1, -1):
            d_pre = d_state
            for v in range(n):
                for j in range(self.out_dim):
                    d_base[v][j] += d_pre[v][j]
                    self.gb[j] += d_pre[v][j]
            if t == 0:
                break
            prop = props[t - 1]
            d_prop = [[0.0] * self.out_dim for _ in range(n)]
            for v in range(n):
                dv = d_pre[v]
                if not any(dv):
                    continue
                pv = prop[v]
                for i in range(self.out_dim):
                    di = dv[i]
                    if di == 0.0:
                        continue
                    for j in range(self.out_dim):
                        self.gW_rec[i][j] += di * pv[j]
                d_prop[v] = _mat_t_vec(self.W_rec, dv)
            # S_hat is symmetric, so the transpose propagation is
            # the same operator applied to d_prop.
            d_state = normalized_propagate(graph, d_prop)
        d_H = [[0.0] * self.in_dim for _ in range(n)]
        for v in range(n):
            dv = d_base[v]
            if not any(dv):
                continue
            x = H[v]
            for i in range(self.out_dim):
                di = dv[i]
                if di == 0.0:
                    continue
                for j in range(self.in_dim):
                    self.gW_in[i][j] += di * x[j]
            dh = _mat_t_vec(self.W_in, dv)
            for j in range(self.in_dim):
                d_H[v][j] += dh[j]
        return d_H

    def step(self, lr: float) -> None:
        for W, gW in ((self.W_in, self.gW_in), (self.W_rec, self.gW_rec)):
            for i in range(len(W)):
                for j in range(len(W[i])):
                    W[i][j] -= lr * gW[i][j]
        for i in range(self.out_dim):
            self.b[i] -= lr * self.gb[i]

    def state_dict(self):
        return {
            "W_in": copy.deepcopy(self.W_in),
            "W_rec": copy.deepcopy(self.W_rec),
            "b": copy.deepcopy(self.b),
        }

    def load_state_dict(self, state) -> None:
        self.W_in = copy.deepcopy(state["W_in"])
        self.W_rec = copy.deepcopy(state["W_rec"])
        self.b = copy.deepcopy(state["b"])
        self.zero_grad()


class GraphARMA:
    """Two-layer ARMA-GNN for binary node classification.

    Full-graph forward (the normalization needs true degrees),
    linear plus sigmoid output head, weighted BCE loss like
    :class:`GraphSAGE`. ``iters`` is shared by both layers.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 16,
        iters: int = 3,
        lr: float = 0.1,
        seed: int = 0,
    ):
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if iters < 0:
            raise ValueError("iters must be non-negative")
        self.rng = random.Random(seed)
        self.hidden_dim = hidden_dim
        self.iters = iters
        self.lr = lr
        self.layer1 = ARMALayer(in_dim, hidden_dim, iters, self.rng)
        self.layer2 = ARMALayer(hidden_dim, hidden_dim, iters, self.rng)
        scale = math.sqrt(2.0 / (hidden_dim + 1))
        self.w_out = [self.rng.uniform(-scale, scale)
                      for _ in range(hidden_dim)]
        self.b_out = 0.0
        self.gw_out = [0.0] * hidden_dim
        self.gb_out = 0.0

    def forward(self, graph, targets: list[int]):
        """Forward pass over the whole graph. Returns (probs dict,
        cache) with probs only for the requested targets."""
        h1, cache1 = self.layer1.forward(graph, graph.features)
        h2, cache2 = self.layer2.forward(graph, h1)
        probs: dict[int, float] = {}
        for v in targets:
            z = sum(w * h for w, h in zip(self.w_out, h2[v])) + self.b_out
            probs[v] = _sigmoid(z)
        cache = {"h1": h1, "cache1": cache1, "h2": h2,
                 "cache2": cache2, "targets": list(targets)}
        return probs, cache

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        """Inference, deterministic, no sampling anywhere."""
        probs, _ = self.forward(graph, targets)
        return probs

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    pos_weight: float = 1.0) -> float:
        """One full-graph SGD step scored on ``targets``. Returns mean
        weighted BCE loss. ``lr=0.0`` on the model accumulates
        gradients without moving the weights, which the numeric
        gradient check relies on."""
        self.layer1.zero_grad()
        self.layer2.zero_grad()
        self.gw_out = [0.0] * self.hidden_dim
        self.gb_out = 0.0

        probs, cache = self.forward(graph, targets)
        h2 = cache["h2"]

        B = len(targets)
        weights = {v: (pos_weight if labels[v] == 1 else 1.0)
                   for v in targets}
        loss = sum(weights[v] * _bce(probs[v], labels[v])
                   for v in targets) / B

        dh2 = [[0.0] * self.hidden_dim for _ in range(graph.n)]
        for v in targets:
            d_logit = weights[v] * (probs[v] - labels[v]) / B
            for j in range(self.hidden_dim):
                self.gw_out[j] += d_logit * h2[v][j]
            self.gb_out += d_logit
            dh2[v] = [d_logit * w for w in self.w_out]

        dh1 = self.layer2.backward(dh2, cache["cache2"])
        self.layer1.backward(dh1, cache["cache1"])

        self.layer1.step(self.lr)
        self.layer2.step(self.lr)
        for j in range(self.hidden_dim):
            self.w_out[j] -= self.lr * self.gw_out[j]
        self.b_out -= self.lr * self.gb_out
        return loss

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

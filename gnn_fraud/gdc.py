"""Graph Diffusion Convolution with a learned heat-kernel time
(Klicpera et al., 2019, diffusion view), from scratch.

Every other propagation model in this repo learns a whole vector
of filter coefficients: GPR-GNN a weight per step, BernNet a value
per Bernstein term, ChebNetII a value per Chebyshev node. GDC asks
the opposite question: what if the filter's *shape* is fixed by a
classical diffusion and only its *scale* is learned?

The filter here is the heat kernel on the graph. With
``S = D^{-1/2} A D^{-1/2}`` (no self-loops, spectrum in [-1, 1])::

    H^(0)_v = relu(W1 @ x_v + b1)          (MLP encoder, no graph)
    Z_v     = sum_k theta_k(t) (S^k H^(0))_v
    logit   = w_out . Z_v + b_out

    theta_k(t) = exp(-t) t^k / k!,    k = 0..K

the Poisson weights of continuous-time diffusion run for time
``t``: a random walker takes a Poisson(t) number of steps, so
``theta_k`` is the probability it has diffused exactly ``k`` hops.
The untruncated weights sum to exactly 1; truncating at ``K``
keeps the mass that has not yet walked past the horizon, so the
filter is a convex-ish smoothing whose reach is one number.

That number is the only learned filter parameter, and it is
learned in log space (``t = exp(log_t)``, init ``t = 2``) so it
stays positive at any step size. Its gradient is closed form:
``d theta_k / d log_t = theta_k (k - t)`` - terms shorter than
the current diffusion time are pushed down as ``t`` grows, terms
longer than it are pushed up, and the term at ``k = t`` does not
move. So training answers a question the coefficient models
cannot ask directly: how far does the fraud signal actually
diffuse before it washes out?

Backprop for the rest is the propagation family's standard one:
the encoder gradient runs the same (symmetric) polynomial in
``S`` backwards over the output gradient. ``K = 0`` degrades
exactly to an MLP on raw features (``theta_0 = exp(-t)`` is a
positive scalar the head absorbs, and no propagation happens).
All linear algebra is hand-rolled lists-of-floats - no numpy,
no torch.
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


def _norm_weights_no_self(graph) -> list[list[tuple[int, float]]]:
    """Symmetric normalization weights *without* self-loops:
    ``S = D^{-1/2} A D^{-1/2}``. Same operator the other spectral
    models use, for the same reason: the diffusion is defined on
    this S, not on the GCN family's self-looped one."""
    d = [graph.degree(v) for v in range(graph.n)]
    weights: list[list[tuple[int, float]]] = []
    for v in range(graph.n):
        row = []
        for u in sorted(set(graph.neighbors(v))):
            row.append((u, 1.0 / math.sqrt(d[v] * d[u])))
        weights.append(row)
    return weights


def _apply_S(weights, H, n: int) -> list[list[float]]:
    width = len(H[0])
    out = []
    for v in range(n):
        acc = [0.0] * width
        for u, c in weights[v]:
            hu = H[u]
            for j in range(width):
                acc[j] += c * hu[j]
        out.append(acc)
    return out


def heat_kernel_coeffs(t: float, K: int) -> list[float]:
    """The truncated heat-kernel (Poisson) weights
    ``theta_k = exp(-t) t^k / k!`` for ``k = 0..K``. Computed by
    the recurrence ``theta_{k+1} = theta_k * t / (k + 1)`` so no
    factorial or power is ever formed directly."""
    if t <= 0:
        raise ValueError("t must be positive")
    if K < 0:
        raise ValueError("K must be non-negative")
    coeffs = [math.exp(-t)]
    for k in range(K):
        coeffs.append(coeffs[-1] * t / (k + 1))
    return coeffs


def diffusion_powers(graph, h0: list[list[float]], K: int):
    """The K + 1 propagation powers ``[h0, S h0, ..., S^K h0]``,
    one application of ``S`` per new power. ``K = 0`` returns
    ``[h0]`` (a copy)."""
    if K < 0:
        raise ValueError("K must be non-negative")
    if len(h0) != graph.n:
        raise ValueError("h0 must have one row per node")
    if any(len(row) != len(h0[0]) for row in h0):
        raise ValueError("all hidden rows must share one width")
    weights = _norm_weights_no_self(graph)
    powers = [[list(row) for row in h0]]
    for _ in range(K):
        powers.append(_apply_S(weights, powers[-1], graph.n))
    return powers


def _apply_poly(weights, H, n: int,
                coeffs: list[float]) -> list[list[float]]:
    """Apply ``sum_j coeffs[j] S^j`` to ``H`` in len(coeffs) - 1
    propagation steps, accumulating the powers of S."""
    width = len(H[0])
    out = [[coeffs[0] * x for x in row] for row in H]
    cur = [list(row) for row in H]
    for j in range(1, len(coeffs)):
        cur = _apply_S(weights, cur, n)
        cj = coeffs[j]
        for v in range(n):
            for t in range(width):
                out[v][t] += cj * cur[v][t]
    return out


class GraphDiffusion:
    """GDC: an MLP encoder, a heat-kernel diffusion of order ``K``
    over the hidden states whose only learned filter parameter is
    the diffusion time ``t = exp(log_t)``, and a linear head.

    ``predict`` takes a graph and target node ids and returns
    ``{v: probability}``, matching the other models' interface, so
    the shared training/evaluation protocol applies unchanged.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 16,
        K: int = 6,
        lr: float = 0.5,
        seed: int = 0,
        t_init: float = 2.0,
    ):
        if K < 0:
            raise ValueError("K must be non-negative")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if t_init <= 0:
            raise ValueError("t_init must be positive")
        rng = random.Random(seed)
        scale = math.sqrt(2.0 / (in_dim + hidden_dim))
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.K = K
        self.lr = lr
        self.W1 = [
            [rng.uniform(-scale, scale) for _ in range(in_dim)]
            for _ in range(hidden_dim)
        ]
        self.b1 = [0.0] * hidden_dim
        self.log_t = math.log(t_init)
        out_scale = math.sqrt(2.0 / (hidden_dim + 1))
        self.w_out = [rng.uniform(-out_scale, out_scale)
                      for _ in range(hidden_dim)]
        self.b_out = 0.0
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW1 = [[0.0] * self.in_dim for _ in range(self.hidden_dim)]
        self.gb1 = [0.0] * self.hidden_dim
        self.glog_t = 0.0
        self.gw_out = [0.0] * self.hidden_dim
        self.gb_out = 0.0

    def diffusion_time(self) -> float:
        """The current diffusion time ``t = exp(log_t)``: roughly
        the mean number of hops the heat kernel spreads a node's
        state over. This is the introspection the model exists
        for - one number saying how far the signal diffuses."""
        return math.exp(self.log_t)

    # -- forward ----------------------------------------------------------

    def forward(self, graph, targets: list[int]):
        """Forward pass over the whole graph. Returns (probs dict,
        cache) with probs only for the requested targets."""
        weights = _norm_weights_no_self(graph)
        pre1 = []
        h0 = []
        for v in range(graph.n):
            x = graph.features[v]
            pre = [
                sum(w * xi for w, xi in zip(row, x)) + b
                for row, b in zip(self.W1, self.b1)
            ]
            pre1.append(pre)
            h0.append([p if p > 0.0 else 0.0 for p in pre])
        powers = diffusion_powers(graph, h0, self.K)
        coeffs = heat_kernel_coeffs(self.diffusion_time(), self.K)
        z = []
        for v in range(graph.n):
            acc = [0.0] * self.hidden_dim
            for k in range(self.K + 1):
                ck = coeffs[k]
                term = powers[k][v]
                for j in range(self.hidden_dim):
                    acc[j] += ck * term[j]
            z.append(acc)
        probs: dict[int, float] = {}
        for v in targets:
            logit = (sum(w * zj for w, zj in zip(self.w_out, z[v]))
                     + self.b_out)
            probs[v] = _sigmoid(logit)
        cache = {
            "weights": weights,
            "pre1": pre1,
            "h0": h0,
            "powers": powers,
            "coeffs": coeffs,
            "z": z,
            "targets": list(targets),
        }
        return probs, cache

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        """Inference - the diffusion is deterministic, so this is exact."""
        probs, _ = self.forward(graph, targets)
        return probs

    # -- training step -----------------------------------------------------

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    pos_weight: float = 1.0) -> float:
        """One SGD step on ``targets``. Returns mean weighted BCE loss.

        The coefficient gradient is the dot product of the output
        gradient with each propagation power; the log-time gradient
        chains it through ``d theta_k / d log_t = theta_k (k - t)``;
        the encoder gradient runs the filter polynomial backwards
        over the output gradient - valid because every term is a
        symmetric polynomial in ``S`` - then through the MLP.
        ``lr=0.0`` accumulates gradients without moving the weights,
        which the numeric gradient check relies on.
        """
        self.zero_grad()
        probs, cache = self.forward(graph, targets)
        weights = cache["weights"]
        pre1 = cache["pre1"]
        powers = cache["powers"]
        coeffs = cache["coeffs"]
        z = cache["z"]
        t = self.diffusion_time()

        B = len(targets)
        wts = {v: (pos_weight if labels[v] == 1 else 1.0)
               for v in targets}
        loss = sum(wts[v] * _bce(probs[v], labels[v])
                   for v in targets) / B

        d_z = [[0.0] * self.hidden_dim for _ in range(graph.n)]
        for v in targets:
            dl = wts[v] * (probs[v] - labels[v]) / B
            if dl == 0.0:
                continue
            for j in range(self.hidden_dim):
                d_z[v][j] = dl * self.w_out[j]
                self.gw_out[j] += dl * z[v][j]
            self.gb_out += dl

        # Coefficient backward, then the closed-form log-time chain.
        self.glog_t = 0.0
        for k in range(self.K + 1):
            term = powers[k]
            gsum = 0.0
            for v in range(graph.n):
                dzv = d_z[v]
                if not any(dzv):
                    continue
                tv = term[v]
                for j in range(self.hidden_dim):
                    gsum += dzv[j] * tv[j]
            self.glog_t += gsum * coeffs[k] * (k - t)

        # Encoder backward: the filter is a symmetric polynomial in S.
        d_h0 = _apply_poly(weights, d_z, graph.n, coeffs)
        for v in range(graph.n):
            dh = d_h0[v]
            if not any(dh):
                continue
            x = graph.features[v]
            for j in range(self.hidden_dim):
                d_pre = dh[j] * (1.0 if pre1[v][j] > 0.0 else 0.0)
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
        self.log_t -= self.lr * self.glog_t
        self.b_out -= self.lr * self.gb_out
        return loss

    # -- checkpointing ------------------------------------------------------

    def state_dict(self):
        return {
            "W1": copy.deepcopy(self.W1),
            "b1": copy.deepcopy(self.b1),
            "log_t": self.log_t,
            "w_out": copy.deepcopy(self.w_out),
            "b_out": self.b_out,
        }

    def load_state_dict(self, state) -> None:
        self.W1 = copy.deepcopy(state["W1"])
        self.b1 = copy.deepcopy(state["b1"])
        self.log_t = state["log_t"]
        self.w_out = copy.deepcopy(state["w_out"])
        self.b_out = state["b_out"]
        self.zero_grad()

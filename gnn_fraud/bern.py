"""BernNet: Bernstein-polynomial spectral filtering (He et al., 2021),
from scratch.

GPR-GNN learns a polynomial of the normalized adjacency directly: one
coefficient per propagation step. BernNet asks what basis that
polynomial is written in, and answers with the Bernstein basis of the
rescaled Laplacian. With ``S = D^{-1/2} A D^{-1/2}`` (no self-loops)
and the scaled Laplacian ``L / 2 = (I - S) / 2``, whose spectrum lies
in [0, 1], the filter is::

    H^(0)_v = relu(W1 @ x_v + b1)          (MLP encoder, no graph)
    Z_v     = sum_k theta_k B_k(H^(0))_v   (Bernstein filter)
    logit   = w_out . Z_v + b_out

    B_k(H)  = C(K, k) * ((I - S) / 2)^k ((I + S) / 2)^(K - k) H

The Bernstein basis has two properties the monomial (GPR) basis does
not. First, the terms form a partition of unity: they sum to the
identity operator, so a filter with every ``theta_k = 1`` is exactly
no filtering. Second, ``theta_k`` is (approximately) the filter's
*value* at the k-th Bernstein point of the spectrum, ``lambda =
2k / K``: the coefficients are the filter shape read off at K + 1
frequencies, not abstract step weights, so a learned theta can be
plotted and read directly. Low frequencies (smooth signals, a node
agreeing with its neighborhood) sit at small k; high frequencies
(a node differing from its neighborhood) sit at large k.

The initialization uses the second property: ``theta_k = 1 - k / K``
is the Bernstein form of the filter ``g(lambda) = 1 - lambda / 2``,
whose operator is exactly ``(I + S) / 2`` - one half-strength
smoothing step, the P operator of the basis itself - so the model
starts as a plain low-pass filter, in Bernstein coordinates, and
learns from there.

Backprop needs one observation: every Bernstein term is a polynomial
in the symmetric ``S``, so the terms commute, each term is symmetric,
and the whole filter is one degree-K polynomial in ``S`` whose scalar
coefficients are a closed-form function of theta
(:func:`bernstein_filter_coeffs`). The theta gradient is the dot
product of the output gradient with each term (computed in the
forward pass); the encoder gradient runs the same polynomial
backwards over the output gradient in K propagation steps instead of
replaying all K^2 / 2 term applications. All linear algebra is
hand-rolled lists-of-floats -- no numpy, no torch.
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
    ``S = D^{-1/2} A D^{-1/2}``, so for node ``v`` the list of
    ``(u, c_vu)`` over ``u in N(v)`` with
    ``c_vu = 1 / sqrt(d_v * d_u)``, ``d = degree``. An isolated node
    gets an empty row (its S row is zero, so ``(I +/- S) / 2`` still
    acts on it through the identity part). BernNet's Laplacian is
    defined on this S, unlike the repo's GCN-family ``S`` which adds
    self-loops first."""
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


def _apply_half(weights, H, n: int, sign: float) -> list[list[float]]:
    """Apply ``(I + sign * S) / 2``: sign=+1 is the P operator,
    sign=-1 is the M operator of the Bernstein terms."""
    SH = _apply_S(weights, H, n)
    return [
        [0.5 * (H[v][j] + sign * SH[v][j]) for j in range(len(H[0]))]
        for v in range(n)
    ]


def bernstein_terms(graph, h0: list[list[float]],
                    K: int) -> list[list[list[float]]]:
    """The K + 1 Bernstein term matrices applied to ``h0``::

        B_k(h0) = C(K, k) * M^k P^(K - k) h0,
        P = (I + S) / 2,   M = (I - S) / 2

    Returns ``[B_0(h0), ..., B_K(h0)]``. The terms sum to ``h0``
    exactly (partition of unity: ``sum_k B_k = (M + P)^K = I``), and
    ``K=0`` returns ``[h0]`` (a copy). Computed by first tabulating
    ``P^j h0`` for j = 0..K (K propagations), then applying ``M^k``
    to ``P^(K-k) h0`` for each k.
    """
    if K < 0:
        raise ValueError("K must be non-negative")
    if len(h0) != graph.n:
        raise ValueError("h0 must have one row per node")
    if any(len(row) != len(h0[0]) for row in h0):
        raise ValueError("all hidden rows must share one width")
    weights = _norm_weights_no_self(graph)
    powers = [[list(row) for row in h0]]
    for _ in range(K):
        powers.append(_apply_half(weights, powers[-1], graph.n, +1.0))
    terms = []
    for k in range(K + 1):
        T = powers[K - k]
        for _ in range(k):
            T = _apply_half(weights, T, graph.n, -1.0)
        c = float(math.comb(K, k))
        terms.append([[c * x for x in row] for row in T])
    return terms


def bernstein_filter_coeffs(theta: list[float]) -> list[float]:
    """Scalar coefficients of the Bernstein filter as a polynomial in
    ``S``: returns ``coeffs`` with ``coeffs[j]`` the coefficient of
    ``S^j`` in ``sum_k theta_k B_k``, where the B_k are the operator
    polynomials of :func:`bernstein_terms`. Degree is
    ``len(theta) - 1``. Two reference points: all-ones theta gives the
    identity (``[1, 0, ..., 0]``, partition of unity), and the linear
    init ``theta_k = 1 - k / K`` gives exactly ``(I + S) / 2``
    (``[0.5, 0.5, 0, ..., 0]``)."""
    K = len(theta) - 1
    total = [0.0] * (K + 1)
    for k in range(K + 1):
        # ((1 - s) / 2)^k * ((1 + s) / 2)^(K - k), expanded.
        p = [1.0]
        for _ in range(k):
            q = [0.0] * (len(p) + 1)
            for i, ci in enumerate(p):
                q[i] += ci * 0.5
                q[i + 1] += ci * -0.5
            p = q
        for _ in range(K - k):
            q = [0.0] * (len(p) + 1)
            for i, ci in enumerate(p):
                q[i] += ci * 0.5
                q[i + 1] += ci * 0.5
            p = q
        c = float(math.comb(K, k)) * theta[k]
        for j in range(K + 1):
            total[j] += c * p[j]
    return total


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


def bern_theta_init(K: int) -> list[float]:
    """The linear low-pass init ``theta_k = 1 - k / K``: the Bernstein
    form of ``g(lambda) = 1 - lambda / 2``, whose operator is exactly
    ``(I + S) / 2`` (one half-strength smoothing step). ``K=0`` gives
    ``[1.0]``."""
    if K < 0:
        raise ValueError("K must be non-negative")
    if K == 0:
        return [1.0]
    return [1.0 - k / K for k in range(K + 1)]


class BernNet:
    """BernNet: an MLP encoder, a Bernstein-polynomial spectral filter
    of order ``K`` over the hidden states with learned coefficients
    ``theta``, and a linear head.

    ``predict`` takes a graph and target node ids and returns
    ``{v: probability}``, matching the other models' interface, so the
    shared training/evaluation protocol applies unchanged.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 16,
        K: int = 4,
        lr: float = 0.5,
        seed: int = 0,
    ):
        if K < 0:
            raise ValueError("K must be non-negative")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
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
        self.theta = bern_theta_init(K)
        out_scale = math.sqrt(2.0 / (hidden_dim + 1))
        self.w_out = [rng.uniform(-out_scale, out_scale)
                      for _ in range(hidden_dim)]
        self.b_out = 0.0
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW1 = [[0.0] * self.in_dim for _ in range(self.hidden_dim)]
        self.gb1 = [0.0] * self.hidden_dim
        self.gtheta = [0.0] * (self.K + 1)
        self.gw_out = [0.0] * self.hidden_dim
        self.gb_out = 0.0

    # -- forward ----------------------------------------------------------

    def forward(self, graph, targets: list[int]):
        """Forward pass over the whole graph. Returns (probs dict,
        cache) with probs only for the requested targets. The cache
        holds the no-self-loop norm weights, the encoder
        pre-activations, the Bernstein terms, and the filtered
        representations backprop needs."""
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
        terms = bernstein_terms(graph, h0, self.K)
        z = []
        for v in range(graph.n):
            acc = [0.0] * self.hidden_dim
            for k in range(self.K + 1):
                tk = self.theta[k]
                term = terms[k][v]
                for j in range(self.hidden_dim):
                    acc[j] += tk * term[j]
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
            "terms": terms,
            "z": z,
            "targets": list(targets),
        }
        return probs, cache

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        """Inference -- the filter is deterministic, so this is exact."""
        probs, _ = self.forward(graph, targets)
        return probs

    # -- training step -----------------------------------------------------

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    pos_weight: float = 1.0) -> float:
        """One SGD step on ``targets``. Returns mean weighted BCE loss.

        The forward pass embeds every node (a target's filtered score
        depends on its neighbors' encoder states), but the loss and its
        seed gradient cover only ``targets``. Theta's gradient is the
        dot product of the output gradient with each Bernstein term;
        the encoder gradient runs the filter polynomial (the same
        coefficients theta defines, via
        :func:`bernstein_filter_coeffs`) backwards over the output
        gradient - valid because every term is a symmetric polynomial
        in ``S`` - then through the MLP. ``lr=0.0`` accumulates
        gradients without moving the weights, which the numeric
        gradient check relies on.
        """
        self.zero_grad()
        probs, cache = self.forward(graph, targets)
        weights = cache["weights"]
        pre1 = cache["pre1"]
        terms = cache["terms"]
        z = cache["z"]

        B = len(targets)
        wts = {v: (pos_weight if labels[v] == 1 else 1.0)
               for v in targets}
        loss = sum(wts[v] * _bce(probs[v], labels[v])
                   for v in targets) / B

        # Seed gradient on the final logits, then the head backward.
        d_z = [[0.0] * self.hidden_dim for _ in range(graph.n)]
        for v in targets:
            dl = wts[v] * (probs[v] - labels[v]) / B
            if dl == 0.0:
                continue
            for j in range(self.hidden_dim):
                d_z[v][j] = dl * self.w_out[j]
                self.gw_out[j] += dl * z[v][j]
            self.gb_out += dl

        # Theta backward: theta_k multiplies term k, so its gradient
        # is <dZ, B_k(H^(0))> over the nodes that carry gradient.
        for k in range(self.K + 1):
            term = terms[k]
            gsum = 0.0
            for v in range(graph.n):
                dzv = d_z[v]
                if not any(dzv):
                    continue
                tv = term[v]
                for j in range(self.hidden_dim):
                    gsum += dzv[j] * tv[j]
            self.gtheta[k] = gsum

        # Encoder backward: the filter is a symmetric polynomial in S,
        # so its transpose is itself with the same coefficients.
        coeffs = bernstein_filter_coeffs(self.theta)
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
        for k in range(self.K + 1):
            self.theta[k] -= self.lr * self.gtheta[k]
        self.b_out -= self.lr * self.gb_out
        return loss

    # -- checkpointing ------------------------------------------------------

    def state_dict(self):
        return {
            "W1": copy.deepcopy(self.W1),
            "b1": copy.deepcopy(self.b1),
            "theta": copy.deepcopy(self.theta),
            "w_out": copy.deepcopy(self.w_out),
            "b_out": self.b_out,
        }

    def load_state_dict(self, state) -> None:
        self.W1 = copy.deepcopy(state["W1"])
        self.b1 = copy.deepcopy(state["b1"])
        self.theta = copy.deepcopy(state["theta"])
        self.w_out = copy.deepcopy(state["w_out"])
        self.b_out = state["b_out"]
        self.zero_grad()

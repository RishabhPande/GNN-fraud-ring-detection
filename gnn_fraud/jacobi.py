"""JacobiConv: Jacobi-polynomial spectral filtering (Wang and
Zhang, 2022), from scratch.

GPR-GNN learns a polynomial of the normalized adjacency in the
monomial basis and BernNet learns one in the Bernstein basis.
JacobiConv asks the same question with a third basis: the Jacobi
polynomials ``P_k^(a, b)``, a two-parameter family orthogonal on
[-1, 1] against the weight ``(1 - x)^a (1 + x)^b``. With
``S = D^{-1/2} A D^{-1/2}`` (no self-loops, spectrum in [-1, 1])
the filter is::

    H^(0)_v = relu(W1 @ x_v + b1)          (MLP encoder, no graph)
    Z_v     = sum_k theta_k P_k^(a, b)(S) H^(0)_v
    logit   = w_out . Z_v + b_out

The two parameters are the point. Special cases recover bases
this repo already has relatives of: ``a = b = -0.5`` is the
Chebyshev basis of the first kind (ChebNet's family), ``a = b = 0``
is Legendre, ``a = b = 0.5`` is Chebyshev of the second kind. Away
from those points the weight tilts toward one end of the spectrum:
``a`` weights the high-frequency end (``x`` near -1 for ``S``,
nodes disagreeing with their neighborhood) and ``b`` the
low-frequency end (``x`` near +1, smooth signals), so the basis
itself can be shaped toward the band where the signal lives before
any coefficient is learned.

The default here is the Chebyshev corner, ``a = b = -0.5``, and
that choice is an empirical finding of this repo, not the paper's
default: at that corner each ``P_k`` is a scalar multiple of the
Chebyshev polynomial ``T_k`` with the multiple shrinking in ``k``
(1/2, 3/8, 5/16 for k = 1, 2, 3), so every term's operator norm
stays under 1 and the filter trains at the GCN family's lr 0.5
like BernNet does. The other corners tried here amplify their
terms (``P_1``'s slope alone is ``(a + b + 2) / 2``, which is 2
at ``a = b = 1``), and on the seed-7 dense ring they land at AUC
0.78-0.85 at every step size tried (0.05-0.5): large steps
destabilize the shared theta, small steps undertrain the encoder.
Basis conditioning, not filter capacity, is the bottleneck --
the degree-4 polynomial space is the same at every corner.

The terms come from the three-term recurrence::

    P_0(x) = 1
    P_1(x) = (a - b) / 2 + (a + b + 2) / 2 * x
    P_{n+1}(x) = (A_n x + B_n) P_n(x) - C_n P_{n-1}(x)

with ``A_n``, ``B_n``, ``C_n`` the standard Jacobi recurrence
coefficients in ``n``, ``a`` and ``b`` (see
:func:`_recurrence_coeffs`). K terms cost K propagations, one per
step of the recurrence, cheaper per term than BernNet's K^2 / 2.

The initialization is the same low-pass BernNet starts from,
written in Jacobi coordinates: ``theta`` is chosen so the filter
is exactly ``(I + S) / 2`` at init (one half-strength smoothing
step), with every coefficient past ``theta_1`` at zero
(:func:`jacobi_theta_init` solves the two equations ``P_0`` and
``P_1`` impose). Learning moves from there.

Backprop needs the same observation BernNet uses: every Jacobi
term is a polynomial in the symmetric ``S``, so the terms commute,
each term is symmetric, and the whole filter is one degree-K
polynomial in ``S`` whose scalar coefficients are a closed-form
function of theta (:func:`jacobi_filter_coeffs`, the recurrence
run on coefficient lists instead of matrices). The theta gradient
is the dot product of the output gradient with each term; the
encoder gradient runs that polynomial backwards over the output
gradient in K propagation steps. All linear algebra is
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


def _check_ab(a: float, b: float) -> None:
    if a <= -1.0 or b <= -1.0:
        raise ValueError("Jacobi parameters a and b must exceed -1")


def _norm_weights_no_self(graph) -> list[list[tuple[int, float]]]:
    """Symmetric normalization weights *without* self-loops:
    ``S = D^{-1/2} A D^{-1/2}``, so for node ``v`` the list of
    ``(u, c_vu)`` over ``u in N(v)`` with
    ``c_vu = 1 / sqrt(d_v * d_u)``, ``d = degree``. An isolated node
    gets an empty row (its S row is zero, so the polynomial still
    acts on it through its constant term). Same operator BernNet
    uses, for the same reason: the spectral basis is defined on
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


def _recurrence_coeffs(n: int, a: float, b: float):
    """The (A_n, B_n, C_n) of the Jacobi three-term recurrence:
    ``P_{n+1} = (A_n x + B_n) P_n - C_n P_{n-1}``, for ``n >= 1``."""
    s = 2.0 * n + a + b
    A = (s + 1.0) * (s + 2.0) / (2.0 * (n + 1.0) * (n + a + b + 1.0))
    B = ((a * a - b * b) * (s + 1.0)
         / (2.0 * (n + 1.0) * (n + a + b + 1.0) * s))
    C = ((n + a) * (n + b) * (s + 2.0)
         / ((n + 1.0) * (n + a + b + 1.0) * s))
    return A, B, C


def jacobi_polys(K: int, a: float, b: float) -> list[list[float]]:
    """Coefficient lists of the first K + 1 Jacobi polynomials:
    entry ``k`` is ``[c_0, ..., c_k]`` with
    ``P_k^(a, b)(x) = sum_j c_j x^j``. ``P_0 = [1]`` and
    ``P_1 = [(a - b) / 2, (a + b + 2) / 2]``; higher orders run
    the three-term recurrence on the coefficient lists, which is
    also how :func:`jacobi_filter_coeffs` expands a learned
    filter into a plain polynomial in ``S``."""
    _check_ab(a, b)
    if K < 0:
        raise ValueError("K must be non-negative")
    polys = [[1.0]]
    if K == 0:
        return polys
    polys.append([(a - b) / 2.0, (a + b + 2.0) / 2.0])
    for n in range(1, K):
        A, B, C = _recurrence_coeffs(n, a, b)
        pn, pnm1 = polys[n], polys[n - 1]
        nxt = [0.0] * (n + 2)
        for j, cj in enumerate(pn):
            nxt[j] += B * cj
            nxt[j + 1] += A * cj
        for j, cj in enumerate(pnm1):
            nxt[j] -= C * cj
        polys.append(nxt)
    return polys


def jacobi_terms(graph, h0: list[list[float]], K: int,
                 a: float = -0.5, b: float = -0.5):
    """The K + 1 Jacobi term matrices applied to ``h0``::

        [P_0(S) h0, P_1(S) h0, ..., P_K(S) h0]

    Computed by the three-term recurrence on matrices: one
    propagation (one application of ``S``) per new term, so K
    propagations in total. ``K = 0`` returns ``[h0]`` (a copy)."""
    _check_ab(a, b)
    if K < 0:
        raise ValueError("K must be non-negative")
    if len(h0) != graph.n:
        raise ValueError("h0 must have one row per node")
    if any(len(row) != len(h0[0]) for row in h0):
        raise ValueError("all hidden rows must share one width")
    weights = _norm_weights_no_self(graph)
    width = len(h0[0])
    terms = [[list(row) for row in h0]]
    if K == 0:
        return terms
    p0 = (a - b) / 2.0
    p1 = (a + b + 2.0) / 2.0
    SH = _apply_S(weights, h0, graph.n)
    terms.append([
        [p0 * h0[v][j] + p1 * SH[v][j] for j in range(width)]
        for v in range(graph.n)
    ])
    for n in range(1, K):
        A, B, C = _recurrence_coeffs(n, a, b)
        ST = _apply_S(weights, terms[n], graph.n)
        terms.append([
            [A * ST[v][j] + B * terms[n][v][j] - C * terms[n - 1][v][j]
             for j in range(width)]
            for v in range(graph.n)
        ])
    return terms


def jacobi_filter_coeffs(theta: list[float], a: float = -0.5,
                         b: float = -0.5) -> list[float]:
    """Scalar coefficients of the Jacobi filter as a polynomial in
    ``S``: returns ``coeffs`` with ``coeffs[j]`` the coefficient of
    ``S^j`` in ``sum_k theta_k P_k^(a, b)``. Degree is
    ``len(theta) - 1``. Reference point: the init from
    :func:`jacobi_theta_init` gives exactly ``(I + S) / 2``
    (``[0.5, 0.5, 0, ..., 0]``) for any ``a`` and ``b``."""
    polys = jacobi_polys(len(theta) - 1, a, b)
    total = [0.0] * len(theta)
    for k, poly in enumerate(polys):
        tk = theta[k]
        for j, cj in enumerate(poly):
            total[j] += tk * cj
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


def jacobi_theta_init(K: int, a: float = -0.5,
                      b: float = -0.5) -> list[float]:
    """The low-pass init whose filter is exactly ``(I + S) / 2``:
    ``theta_1 = 1 / (a + b + 2)`` makes the ``S`` coefficient 0.5
    through ``P_1``'s slope, and ``theta_0`` cancels the constant
    ``P_1`` contributes, ``theta_0 = 1/2 - (a - b) / (2 (a + b +
    2))``. Every later coefficient starts at zero. ``K = 0`` gives
    ``[1.0]`` (the identity filter, the only order-0 choice)."""
    _check_ab(a, b)
    if K < 0:
        raise ValueError("K must be non-negative")
    if K == 0:
        return [1.0]
    theta = [0.0] * (K + 1)
    theta[1] = 1.0 / (a + b + 2.0)
    theta[0] = 0.5 - (a - b) / (2.0 * (a + b + 2.0))
    return theta


class JacobiConv:
    """JacobiConv: an MLP encoder, a Jacobi-polynomial spectral
    filter of order ``K`` over the hidden states with learned
    coefficients ``theta`` and basis parameters ``a`` and ``b``
    (fixed, not learned -- the paper tunes them as hyperparameters),
    and a linear head.

    ``predict`` takes a graph and target node ids and returns
    ``{v: probability}``, matching the other models' interface, so
    the shared training/evaluation protocol applies unchanged.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 16,
        K: int = 4,
        a: float = -0.5,
        b: float = -0.5,
        lr: float = 0.5,
        seed: int = 0,
    ):
        _check_ab(a, b)
        if K < 0:
            raise ValueError("K must be non-negative")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        rng = random.Random(seed)
        scale = math.sqrt(2.0 / (in_dim + hidden_dim))
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.K = K
        self.a = a
        self.b = b
        self.lr = lr
        self.W1 = [
            [rng.uniform(-scale, scale) for _ in range(in_dim)]
            for _ in range(hidden_dim)
        ]
        self.b1 = [0.0] * hidden_dim
        self.theta = jacobi_theta_init(K, a, b)
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
        pre-activations, the Jacobi terms, and the filtered
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
        terms = jacobi_terms(graph, h0, self.K, self.a, self.b)
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
        dot product of the output gradient with each Jacobi term; the
        encoder gradient runs the filter polynomial (the same
        coefficients theta defines, via :func:`jacobi_filter_coeffs`)
        backwards over the output gradient -- valid because every term
        is a symmetric polynomial in ``S`` -- then through the MLP.
        ``lr=0.0`` accumulates gradients without moving the weights,
        which the numeric gradient check relies on.
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
        # is <dZ, P_k(S) H^(0)> over the nodes that carry gradient.
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
        coeffs = jacobi_filter_coeffs(self.theta, self.a, self.b)
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

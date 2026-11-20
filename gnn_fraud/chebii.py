"""ChebNetII: Chebyshev interpolation spectral filtering (He et
al., 2022), from scratch.

ChebNet, GPR-GNN, BernNet, and JacobiConv all learn a polynomial
filter's coefficients directly, in one basis or another. ChebNetII
asks why the coefficients should be the learned parameters at all.
Direct coefficient learning in the monomial and Chebyshev bases is
ill conditioned at higher orders - small coefficient changes swing
the filter wildly between the points that matter - and the paper
traces ChebNet's weak higher-order behaviour to exactly that.

The fix is interpolation. Fix the Chebyshev nodes of the first
kind on the spectrum of ``S = D^{-1/2} A D^{-1/2}`` (no self-loops,
spectrum in [-1, 1])::

    x_j = cos((j + 1/2) pi / (K + 1)),    j = 0..K

and learn the filter's *values* at those nodes, ``gamma_j =
g(x_j)``, instead of any basis coefficients. The Chebyshev
coefficients of the interpolating polynomial are then a fixed
linear map of the values (a discrete cosine transform)::

    c_0 = (1 / (K + 1)) sum_j gamma_j
    c_k = (2 / (K + 1)) sum_j gamma_j cos(k (j + 1/2) pi / (K + 1))

and the filter itself is the usual Chebyshev sum::

    H^(0)_v = relu(W1 @ x_v + b1)          (MLP encoder, no graph)
    Z_v     = sum_k c_k T_k(S) H^(0)_v
    logit   = w_out . Z_v + b_out

with ``T_k`` from the recurrence ``T_0 = 1``, ``T_1 = x``,
``T_{k+1} = 2 x T_k - T_{k-1}``. Learning values instead of
coefficients keeps every learned parameter on the scale of the
filter itself - a gamma of 2 means the filter passes that
frequency at gain 2, wherever it sits - and the interpolation
is exact for any filter of degree at most K, so no capacity is
lost: the degree-K polynomial space is the same one JacobiConv
searches, reached through better conditioned coordinates.

The initialization uses that readability: ``gamma_j = (1 + x_j)
/ 2`` is the value table of the low-pass filter ``g(x) = (1 + x)
/ 2``, whose operator is exactly ``(I + S) / 2`` - the same
half-strength smoothing step BernNet and JacobiConv start from,
written in value coordinates. Interpolation of a degree-1 filter
at K + 1 nodes is exact for any K >= 1, so the init filter is
exact, not approximate. ``K = 0`` is the one exception: a
degree-0 filter cannot represent ``(I + S) / 2`` at all, so its
init is the identity (``gamma = [1.0]``).

Backprop runs through both linear maps. The gamma gradient is
the cosine transform of the Chebyshev-coefficient gradient
(``d gamma_j = sum_k (dc_k / d gamma_j) d c_k``, the transpose
of the interpolation map); the encoder gradient runs the whole
filter, expanded to a plain polynomial in ``S`` via
:func:`chebii_filter_coeffs`, backwards over the output gradient
in K propagation steps - valid because every ``T_k(S)`` is a
symmetric polynomial in ``S``. All linear algebra is hand-rolled
lists-of-floats - no numpy, no torch.
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
    gets an empty row (its S row is zero, so the polynomial still
    acts on it through its constant term). Same operator BernNet
    and JacobiConv use, for the same reason: the spectral basis is
    defined on this S, not on the GCN family's self-looped one."""
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


def chebii_nodes(K: int) -> list[float]:
    """The K + 1 Chebyshev nodes of the first kind on [-1, 1]:
    ``x_j = cos((j + 1/2) pi / (K + 1))`` for ``j = 0..K``, in
    decreasing order (``x_0`` near +1, the low-frequency end for
    ``S``, and ``x_K`` near -1, the high-frequency end)."""
    if K < 0:
        raise ValueError("K must be non-negative")
    return [math.cos((j + 0.5) * math.pi / (K + 1))
            for j in range(K + 1)]


def chebii_cheb_coeffs(gamma: list[float]) -> list[float]:
    """Chebyshev coefficients of the polynomial interpolating the
    values ``gamma`` at the nodes :func:`chebii_nodes`: returns
    ``c`` with the filter ``sum_k c_k T_k(x)`` passing through
    ``(x_j, gamma_j)`` at every node. ``c_0`` carries the half
    weight of the interpolation formula; every other ``c_k``
    carries the factor 2 / (K + 1)."""
    K = len(gamma) - 1
    if K < 0:
        raise ValueError("gamma must be non-empty")
    coeffs = []
    for k in range(K + 1):
        s = 0.0
        for j in range(K + 1):
            s += gamma[j] * math.cos(k * (j + 0.5) * math.pi / (K + 1))
        factor = 1.0 if k == 0 else 2.0
        coeffs.append(factor * s / (K + 1))
    return coeffs


def _cheb_polys(K: int) -> list[list[float]]:
    """Coefficient lists of the first K + 1 Chebyshev polynomials
    of the first kind: entry ``k`` is ``[c_0, ..., c_k]`` with
    ``T_k(x) = sum_j c_j x^j``, from ``T_0 = [1]``, ``T_1 = [0, 1]``
    and the recurrence ``T_{k+1} = 2 x T_k - T_{k-1}`` run on the
    coefficient lists."""
    polys = [[1.0]]
    if K == 0:
        return polys
    polys.append([0.0, 1.0])
    for k in range(1, K):
        tk, tkm1 = polys[k], polys[k - 1]
        nxt = [0.0] * (k + 2)
        for j, cj in enumerate(tk):
            nxt[j + 1] += 2.0 * cj
        for j, cj in enumerate(tkm1):
            nxt[j] -= cj
        polys.append(nxt)
    return polys


def chebii_filter_coeffs(gamma: list[float]) -> list[float]:
    """Scalar coefficients of the ChebNetII filter as a polynomial
    in ``S``: returns ``coeffs`` with ``coeffs[j]`` the coefficient
    of ``S^j`` in the filter the values ``gamma`` define (first
    the interpolation map to Chebyshev coefficients, then the
    Chebyshev expansion). Degree is ``len(gamma) - 1``. Reference
    point: the init from :func:`chebii_gamma_init` gives exactly
    ``(I + S) / 2`` (``[0.5, 0.5, 0, ..., 0]``) for any K >= 1."""
    cheb = chebii_cheb_coeffs(gamma)
    polys = _cheb_polys(len(gamma) - 1)
    total = [0.0] * len(gamma)
    for k, poly in enumerate(polys):
        ck = cheb[k]
        for j, cj in enumerate(poly):
            total[j] += ck * cj
    return total


def chebii_terms(graph, h0: list[list[float]], K: int):
    """The K + 1 Chebyshev term matrices applied to ``h0``::

        [T_0(S) h0, T_1(S) h0, ..., T_K(S) h0]

    Computed by the three-term recurrence on matrices: one
    propagation (one application of ``S``) per new term, so K
    propagations in total. ``K = 0`` returns ``[h0]`` (a copy)."""
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
    terms.append(_apply_S(weights, h0, graph.n))
    for k in range(1, K):
        ST = _apply_S(weights, terms[k], graph.n)
        terms.append([
            [2.0 * ST[v][j] - terms[k - 1][v][j] for j in range(width)]
            for v in range(graph.n)
        ])
    return terms


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


def chebii_gamma_init(K: int) -> list[float]:
    """The low-pass init whose filter is exactly ``(I + S) / 2``
    for K >= 1: the value table ``gamma_j = (1 + x_j) / 2`` of
    ``g(x) = (1 + x) / 2`` at the Chebyshev nodes, which the
    interpolation reproduces exactly since ``g`` has degree 1.
    ``K = 0`` gives ``[1.0]`` (the identity filter, the only
    sensible order-0 choice, since no degree-0 filter is
    ``(I + S) / 2``)."""
    if K < 0:
        raise ValueError("K must be non-negative")
    if K == 0:
        return [1.0]
    return [(1.0 + x) / 2.0 for x in chebii_nodes(K)]


class ChebNetII:
    """ChebNetII: an MLP encoder, a Chebyshev-interpolation spectral
    filter of order ``K`` over the hidden states whose learned
    parameters ``gamma`` are the filter's values at the Chebyshev
    nodes, and a linear head.

    ``predict`` takes a graph and target node ids and returns
    ``{v: probability}``, matching the other models' interface, so
    the shared training/evaluation protocol applies unchanged.
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
        self.gamma = chebii_gamma_init(K)
        out_scale = math.sqrt(2.0 / (hidden_dim + 1))
        self.w_out = [rng.uniform(-out_scale, out_scale)
                      for _ in range(hidden_dim)]
        self.b_out = 0.0
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW1 = [[0.0] * self.in_dim for _ in range(self.hidden_dim)]
        self.gb1 = [0.0] * self.hidden_dim
        self.ggamma = [0.0] * (self.K + 1)
        self.gw_out = [0.0] * self.hidden_dim
        self.gb_out = 0.0

    # -- forward ----------------------------------------------------------

    def forward(self, graph, targets: list[int]):
        """Forward pass over the whole graph. Returns (probs dict,
        cache) with probs only for the requested targets. The cache
        holds the no-self-loop norm weights, the encoder
        pre-activations, the Chebyshev terms, the interpolated
        Chebyshev coefficients, and the filtered representations
        backprop needs."""
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
        terms = chebii_terms(graph, h0, self.K)
        cheb = chebii_cheb_coeffs(self.gamma)
        z = []
        for v in range(graph.n):
            acc = [0.0] * self.hidden_dim
            for k in range(self.K + 1):
                ck = cheb[k]
                term = terms[k][v]
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
            "terms": terms,
            "cheb": cheb,
            "z": z,
            "targets": list(targets),
        }
        return probs, cache

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        """Inference - the filter is deterministic, so this is exact."""
        probs, _ = self.forward(graph, targets)
        return probs

    def filter_values(self) -> list[float]:
        """The learned filter read back in value coordinates: a copy
        of ``gamma``, the filter's value at each Chebyshev node
        :func:`chebii_nodes` (low frequencies first). This is the
        introspection ChebNetII exists for - the parameters *are*
        the filter shape, no basis conversion needed."""
        return list(self.gamma)

    # -- training step -----------------------------------------------------

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    pos_weight: float = 1.0) -> float:
        """One SGD step on ``targets``. Returns mean weighted BCE loss.

        The forward pass embeds every node (a target's filtered score
        depends on its neighbors' encoder states), but the loss and its
        seed gradient cover only ``targets``. The Chebyshev-coefficient
        gradient is the dot product of the output gradient with each
        Chebyshev term; the gamma gradient is that gradient run
        through the transpose of the interpolation map (the same
        cosine sum with the same 1-or-2 factor); the encoder gradient
        runs the filter polynomial (via :func:`chebii_filter_coeffs`)
        backwards over the output gradient - valid because every term
        is a symmetric polynomial in ``S`` - then through the MLP.
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

        # Chebyshev-coefficient backward: c_k multiplies term k, so
        # its gradient is <dZ, T_k(S) H^(0)>.
        gcheb = [0.0] * (self.K + 1)
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
            gcheb[k] = gsum

        # Gamma backward: gamma is upstream of c through the
        # interpolation map, so its gradient is that map's transpose
        # applied to gcheb - the same cosine sum, factor included.
        for j in range(self.K + 1):
            gsum = 0.0
            for k in range(self.K + 1):
                factor = 1.0 if k == 0 else 2.0
                gsum += (factor / (self.K + 1)) * gcheb[k] * math.cos(
                    k * (j + 0.5) * math.pi / (self.K + 1))
            self.ggamma[j] = gsum

        # Encoder backward: the filter is a symmetric polynomial in S,
        # so its transpose is itself with the same coefficients.
        coeffs = chebii_filter_coeffs(self.gamma)
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
            self.gamma[k] -= self.lr * self.ggamma[k]
        self.b_out -= self.lr * self.gb_out
        return loss

    # -- checkpointing ------------------------------------------------------

    def state_dict(self):
        return {
            "W1": copy.deepcopy(self.W1),
            "b1": copy.deepcopy(self.b1),
            "gamma": copy.deepcopy(self.gamma),
            "w_out": copy.deepcopy(self.w_out),
            "b_out": self.b_out,
        }

    def load_state_dict(self, state) -> None:
        self.W1 = copy.deepcopy(state["W1"])
        self.b1 = copy.deepcopy(state["b1"])
        self.gamma = copy.deepcopy(state["gamma"])
        self.w_out = copy.deepcopy(state["w_out"])
        self.b_out = state["b_out"]
        self.zero_grad()

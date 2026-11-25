"""LightGCN (He et al., 2020), from scratch - and stripped to the
question this repo can ask it.

Every other model here starts from the accounts' features: an MLP
or a message-passing layer transforms ``x_v``, and the graph only
mixes those transforms. LightGCN removes the transformation
entirely. The only learned node states are free embeddings, one
row per account, and the propagation has no weight matrices and
no nonlinearity at all::

    E^(0)   = E                          (free embeddings, learned)
    E^(k+1) = S E^(k)                    S = D^{-1/2} A D^{-1/2}
    Z       = (1 / (K + 1)) sum_k E^(k)   (uniform layer average)
    logit_v = w_out . Z_v + b_out

``S`` is the symmetric normalization *without* self-loops - the
LightGCN paper's operator, not the GCN family's ``S_hat`` - so a
node's own embedding reaches its final state only through the
``k = 0`` term of the average, at weight ``1 / (K + 1)``.

That makes this the repo's structure-only control: the model
never sees a single account feature, so whatever it ranks, it
ranks from the wiring alone. A fraud ring is exactly the pattern
that should survive that ablation - the generator camouflages
the features and leaves the dense pocket in the wiring - and the
transductive price is stated plainly: the embeddings are per
account, so a brand-new account has no row and cannot be scored
until it is added and the model retrained. That is the opposite
trade from the inductive check elsewhere in this repo.

``K = 0`` degrades exactly to a linear head over free embeddings
(no propagation at all). The operator is symmetric, so the
backward pass is the same layer average applied to the output
gradient. All linear algebra is hand-rolled lists-of-floats - no
numpy, no torch.
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


def _norm_weights_no_self_loop(graph) -> list[list[tuple[int, float]]]:
    """Symmetric normalization weights without self-loops:
    ``S = D^{-1/2} A D^{-1/2}``, the LightGCN paper's operator.
    An isolated node gets an empty row, so its propagated state
    is exactly zero at every power ``k >= 1``."""
    weights: list[list[tuple[int, float]]] = []
    for v in range(graph.n):
        dv = graph.degree(v)
        row = []
        for u in sorted(set(graph.neighbors(v))):
            row.append((u, 1.0 / math.sqrt(dv * graph.degree(u))))
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


def embedding_powers(graph, embeddings: list[list[float]], K: int):
    """The K + 1 propagation powers ``[E, S E, ..., S^K E]``,
    one application of ``S`` per new power. ``K = 0`` returns
    ``[E]`` (a copy)."""
    if K < 0:
        raise ValueError("K must be non-negative")
    if len(embeddings) != graph.n:
        raise ValueError("embeddings must have one row per node")
    if any(len(row) != len(embeddings[0]) for row in embeddings):
        raise ValueError("all embedding rows must share one width")
    weights = _norm_weights_no_self_loop(graph)
    powers = [[list(row) for row in embeddings]]
    for _ in range(K):
        powers.append(_apply_S(weights, powers[-1], graph.n))
    return powers


def averaged_embeddings(graph, embeddings: list[list[float]], K: int):
    """The LightGCN final state: the elementwise mean of the
    propagation powers ``(1 / (K + 1)) sum_k S^k E``. ``K = 0``
    returns a copy of ``embeddings``."""
    powers = embedding_powers(graph, embeddings, K)
    n = graph.n
    width = len(embeddings[0])
    scale = 1.0 / (K + 1)
    out = [[0.0] * width for _ in range(n)]
    for power in powers:
        for v in range(n):
            pv = power[v]
            ov = out[v]
            for j in range(width):
                ov[j] += scale * pv[j]
    return out


class LightGCN:
    """LightGCN: free per-node embeddings, parameter-free
    propagation of order ``K``, a uniform layer average, and a
    linear head. No node features are used anywhere.

    ``predict`` takes a graph and target node ids and returns
    ``{v: probability}``, matching the other models' interface,
    so the shared training/evaluation protocol applies unchanged.
    The graph must have the ``n_nodes`` the model was built for.
    """

    def __init__(
        self,
        n_nodes: int,
        emb_dim: int = 16,
        K: int = 3,
        lr: float = 0.5,
        seed: int = 0,
    ):
        if n_nodes <= 0:
            raise ValueError("n_nodes must be positive")
        if emb_dim <= 0:
            raise ValueError("emb_dim must be positive")
        if K < 0:
            raise ValueError("K must be non-negative")
        rng = random.Random(seed)
        scale = math.sqrt(2.0 / emb_dim)
        self.n_nodes = n_nodes
        self.emb_dim = emb_dim
        self.K = K
        self.lr = lr
        self.embeddings = [
            [rng.uniform(-scale, scale) for _ in range(emb_dim)]
            for _ in range(n_nodes)
        ]
        out_scale = math.sqrt(2.0 / (emb_dim + 1))
        self.w_out = [rng.uniform(-out_scale, out_scale)
                      for _ in range(emb_dim)]
        self.b_out = 0.0
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gE = [[0.0] * self.emb_dim for _ in range(self.n_nodes)]
        self.gw_out = [0.0] * self.emb_dim
        self.gb_out = 0.0

    # -- forward ----------------------------------------------------------

    def forward(self, graph, targets: list[int]):
        """Forward pass over the whole graph. Returns (probs dict,
        cache) with probs only for the requested targets."""
        if graph.n != self.n_nodes:
            raise ValueError(
                f"model was built for {self.n_nodes} nodes, "
                f"graph has {graph.n}")
        weights = _norm_weights_no_self_loop(graph)
        z = averaged_embeddings(graph, self.embeddings, self.K)
        probs: dict[int, float] = {}
        for v in targets:
            logit = (sum(w * zj for w, zj in zip(self.w_out, z[v]))
                     + self.b_out)
            probs[v] = _sigmoid(logit)
        cache = {
            "weights": weights,
            "z": z,
            "targets": list(targets),
        }
        return probs, cache

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        """Inference - the averaging is deterministic, so this is exact."""
        probs, _ = self.forward(graph, targets)
        return probs

    # -- training step -----------------------------------------------------

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    pos_weight: float = 1.0) -> float:
        """One SGD step on ``targets``. Returns mean weighted BCE loss.

        The propagation is symmetric and parameter-free, so the
        embedding gradient is the same layer average applied to
        the output gradient. Every node's embedding can receive
        gradient from a target up to K hops away, which is the
        whole mechanism: a held-out ring member's embedding is
        trained by its labeled ring neighbors. ``lr=0.0``
        accumulates gradients without moving the weights, which
        the numeric gradient check relies on.
        """
        self.zero_grad()
        probs, cache = self.forward(graph, targets)
        weights = cache["weights"]
        z = cache["z"]

        B = len(targets)
        wts = {v: (pos_weight if labels[v] == 1 else 1.0)
               for v in targets}
        loss = sum(wts[v] * _bce(probs[v], labels[v])
                   for v in targets) / B

        d_z = [[0.0] * self.emb_dim for _ in range(graph.n)]
        for v in targets:
            dl = wts[v] * (probs[v] - labels[v]) / B
            if dl == 0.0:
                continue
            for j in range(self.emb_dim):
                d_z[v][j] = dl * self.w_out[j]
                self.gw_out[j] += dl * z[v][j]
            self.gb_out += dl

        # Embedding backward: the averaged propagation is a
        # symmetric polynomial in S, so its transpose is itself.
        d_e = _apply_average(weights, d_z, graph.n, self.K)
        for v in range(graph.n):
            for j in range(self.emb_dim):
                self.gE[v][j] = d_e[v][j]

        # SGD update.
        for v in range(self.n_nodes):
            for j in range(self.emb_dim):
                self.embeddings[v][j] -= self.lr * self.gE[v][j]
        for j in range(self.emb_dim):
            self.w_out[j] -= self.lr * self.gw_out[j]
        self.b_out -= self.lr * self.gb_out
        return loss

    # -- checkpointing ------------------------------------------------------

    def state_dict(self):
        return {
            "embeddings": copy.deepcopy(self.embeddings),
            "w_out": copy.deepcopy(self.w_out),
            "b_out": self.b_out,
        }

    def load_state_dict(self, state) -> None:
        self.embeddings = copy.deepcopy(state["embeddings"])
        self.w_out = copy.deepcopy(state["w_out"])
        self.b_out = state["b_out"]
        self.zero_grad()


def _apply_average(weights, H, n: int, K: int) -> list[list[float]]:
    """Apply ``(1 / (K + 1)) sum_k S^k`` to ``H`` in K propagation
    steps, accumulating the powers of S."""
    width = len(H[0])
    scale = 1.0 / (K + 1)
    out = [[scale * x for x in row] for row in H]
    cur = [list(row) for row in H]
    for _ in range(K):
        cur = _apply_S(weights, cur, n)
        for v in range(n):
            for t in range(width):
                out[v][t] += scale * cur[v][t]
    return out

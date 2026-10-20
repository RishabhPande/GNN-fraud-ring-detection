"""Edge classification: score individual edges as fraud-related or benign.

Node classification answers "which accounts look suspicious". Investigators
also need "which flows move the money". This module scores every edge of the
transaction graph with a small MLP head on top of *frozen* node embeddings
from a trained GNN encoder::

    phi(u, v) = [h_u, h_v, h_u (x) h_v, |h_u - h_v|]
    p(u, v)   = sigmoid(w2^T . relu(W1 . phi(u, v) + b1) + b2)

The interaction terms (elementwise product and absolute difference) are
what let the head see *pairs*, not just endpoints: a ring edge is two
suspicious nodes wired together, and the head learns that joint signature.
Edges are labeled 1 when both endpoints are fraud (a ring-internal flow)
and 0 otherwise; training uses edge-level mini-batches with optional
positive-class weighting, since ring edges are the minority on most
graphs.

The encoder is deliberately frozen: the head is a *probe* of what the
node model already learned about graph structure, not a second
message-passing learner. Joint fine-tuning of the encoder on the edge
objective is future work.

All data is synthetic. No real accounts, no real transactions, no PII.
"""

from __future__ import annotations

import copy
import math
import random

# ---------------------------------------------------------------------------
# edge labels and features
# ---------------------------------------------------------------------------


def edge_phi(h_u: list[float], h_v: list[float]) -> list[float]:
    """Edge feature vector from a pair of node embeddings.

    Concatenates the two endpoint embeddings with their elementwise
    product and absolute difference (dim 4 * embedding_dim).
    """
    return (
        list(h_u)
        + list(h_v)
        + [a * b for a, b in zip(h_u, h_v)]
        + [abs(a - b) for a, b in zip(h_u, h_v)]
    )


def edge_labels(graph) -> dict[tuple[int, int], int]:
    """Label every edge 1 if it is ring-internal, 0 otherwise.

    An edge is ring-internal when *both* endpoints carry the fraud label.
    Ring-to-benign bridge edges count as 0: the ring may touch the outside
    world, but its internal money flows are the positive class.

    Keys are (min(u, v), max(u, v)) so undirected edges map to one entry.
    """
    out: dict[tuple[int, int], int] = {}
    for u in range(graph.n):
        for v in graph.adj[u]:
            if v <= u:
                continue
            out[(u, v)] = 1 if (graph.labels[u] == 1 and
                                graph.labels[v] == 1) else 0
    return out


def edge_train_val_split(
    edges: list[tuple[int, int]],
    labels: dict[tuple[int, int], int],
    val_frac: float = 0.25,
    seed: int = 0,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Stratified train/validation split of edge ids."""
    rng = random.Random(seed)
    pos = [e for e in edges if labels[e] == 1]
    neg = [e for e in edges if labels[e] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos_val = max(1, int(len(pos) * val_frac))
    n_neg_val = max(1, int(len(neg) * val_frac))
    val = pos[:n_pos_val] + neg[:n_neg_val]
    train = pos[n_pos_val:] + neg[n_neg_val:]
    rng.shuffle(train)
    return train, val


# ---------------------------------------------------------------------------
# the edge head: one-hidden-layer MLP over edge_phi
# ---------------------------------------------------------------------------


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _bce(p: float, y: int) -> float:
    p = min(max(p, 1e-7), 1.0 - 1e-7)
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


class EdgeHead:
    """MLP edge scorer: edge_phi -> hidden ReLU -> sigmoid fraud probability."""

    def __init__(self, in_dim: int, hidden_dim: int = 16,
                 lr: float = 0.1, seed: int = 0):
        rng = random.Random(seed)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        s1 = math.sqrt(2.0 / (in_dim + hidden_dim))  # Xavier-ish init
        self.W1 = [[rng.uniform(-s1, s1) for _ in range(in_dim)]
                   for _ in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim
        s2 = math.sqrt(2.0 / (hidden_dim + 1))
        self.w2 = [rng.uniform(-s2, s2) for _ in range(hidden_dim)]
        self.b2 = 0.0
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW1 = [[0.0] * self.in_dim for _ in range(self.hidden_dim)]
        self.gb1 = [0.0] * self.hidden_dim
        self.gw2 = [0.0] * self.hidden_dim
        self.gb2 = 0.0

    def forward(self, phi: list[float]):
        """Returns (probability, cache) for one edge feature vector."""
        pre1 = [sum(w * x for w, x in zip(row, phi)) + b
                for row, b in zip(self.W1, self.b1)]
        h = [x if x > 0.0 else 0.0 for x in pre1]
        z = sum(w * x for w, x in zip(self.w2, h)) + self.b2
        return _sigmoid(z), (phi, pre1, h, z)

    def backward(self, d_z: float, cache) -> None:
        """Backprop one edge example.

        Args:
            d_z: gradient of the batch loss w.r.t. the pre-sigmoid logit.
                (The sigmoid's derivative cancels against the BCE
                derivative, leaving d_z = w * (p - y) / batch_size.)
            cache: the forward cache for this example.
        """
        phi, pre1, h, _ = cache
        # Output layer.
        for j in range(self.hidden_dim):
            self.gw2[j] += d_z * h[j]
        self.gb2 += d_z
        # Hidden layer.
        for j in range(self.hidden_dim):
            d_h = self.w2[j] * d_z
            d_pre = d_h if pre1[j] > 0.0 else 0.0
            if d_pre == 0.0:
                continue
            row = self.gW1[j]
            for k in range(self.in_dim):
                row[k] += d_pre * phi[k]
            self.gb1[j] += d_pre

    def step(self, lr: float) -> None:
        for j in range(self.hidden_dim):
            for k in range(self.in_dim):
                self.W1[j][k] -= lr * self.gW1[j][k]
            self.b1[j] -= lr * self.gb1[j]
            self.w2[j] -= lr * self.gw2[j]
        self.b2 -= lr * self.gb2

    def state_dict(self):
        return copy.deepcopy({"W1": self.W1, "b1": self.b1,
                              "w2": self.w2, "b2": self.b2})

    def load_state_dict(self, state) -> None:
        self.W1 = copy.deepcopy(state["W1"])
        self.b1 = copy.deepcopy(state["b1"])
        self.w2 = copy.deepcopy(state["w2"])
        self.b2 = copy.deepcopy(state["b2"])


# ---------------------------------------------------------------------------
# training the head on frozen encoder embeddings
# ---------------------------------------------------------------------------


def train_edge_head(
    embeddings: dict[int, list[float]],
    edges: list[tuple[int, int]],
    labels: dict[tuple[int, int], int],
    train_edges: list[tuple[int, int]] | None = None,
    val_edges: list[tuple[int, int]] | None = None,
    hidden_dim: int = 16,
    lr: float = 0.1,
    batch_size: int = 64,
    epochs: int = 40,
    patience: int = 8,
    seed: int = 0,
    pos_weight: float = 1.0,
) -> tuple[EdgeHead, dict]:
    """Train an EdgeHead on frozen node embeddings.

    Args:
        embeddings: node id -> embedding vector (e.g. layer-2 output of a
            trained GraphSAGE encoder, full-neighborhood inference).
        edges: all edge ids to use, as (u, v) with u < v.
        labels: edge id -> 0/1 from :func:`edge_labels`.
        train_edges / val_edges: explicit split; if omitted, a stratified
            75/25 split is made with ``seed``.
        pos_weight: multiplier on the loss of ring-internal edges.
        seed: used for the edge split, mini-batch shuffling, and head init.

    Returns (head, history) with per-epoch train/val mean weighted BCE.
    Early-stops on validation loss and restores the best checkpoint.
    """
    if train_edges is None or val_edges is None:
        train_edges, val_edges = edge_train_val_split(
            edges, labels, seed=seed)
    in_dim = len(next(iter(embeddings.values())))
    head = EdgeHead(in_dim * 4, hidden_dim=hidden_dim, lr=lr, seed=seed)
    rng = random.Random(seed)
    train_edges = list(train_edges)  # copy: shuffling must not reorder it

    # Precompute edge features once -- the encoder is frozen, so they never
    # change. Edge-level training then costs O(edges * hidden) per epoch.
    phis = {e: edge_phi(embeddings[e[0]], embeddings[e[1]]) for e in edges}

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    def mean_loss(edge_list: list[tuple[int, int]]) -> float:
        total = 0.0
        for e in edge_list:
            p, _ = head.forward(phis[e])
            w = pos_weight if labels[e] == 1 else 1.0
            total += w * _bce(p, labels[e])
        return total / len(edge_list)

    for _ in range(epochs):
        rng.shuffle(train_edges)
        for i in range(0, len(train_edges), batch_size):
            batch = train_edges[i:i + batch_size]
            head.zero_grad()
            B = len(batch)
            for e in batch:
                p, cache = head.forward(phis[e])
                w = pos_weight if labels[e] == 1 else 1.0
                head.backward(w * (p - labels[e]) / B, cache)
            head.step(lr)
        train_loss = mean_loss(train_edges)
        val_loss = mean_loss(val_edges)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = head.state_dict()
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    return head, history


def score_edges(
    head: EdgeHead,
    embeddings: dict[int, list[float]],
    edges: list[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    """Score a list of edges. Returns edge id -> fraud probability."""
    return {e: head.forward(edge_phi(embeddings[e[0]], embeddings[e[1]]))[0]
            for e in edges}


def full_embeddings(encoder, graph, node_ids: list[int]
                    ) -> dict[int, list[float]]:
    """Layer-2 embeddings for every node, deterministic full neighborhood.

    Works with any encoder in this package whose ``forward`` returns
    ``(probs, cache)`` with ``cache["h2"]`` -- GraphSAGE, GAT,
    RelationalGraphSAGE, and TemporalGraphSAGE all qualify.
    """
    _, cache = encoder.forward(graph, list(node_ids), k1=None, k2=None)
    return cache["h2"]

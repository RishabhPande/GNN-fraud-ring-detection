"""Training loop for GraphSAGE: mini-batches, validation, early stopping."""

from __future__ import annotations

import random

from gnn_fraud.edge_cls import (
    edge_labels,
    edge_train_val_split,
    full_embeddings,
    train_edge_head,
)
from gnn_fraud.gat import GraphAttention
from gnn_fraud.rsage import RelationalGraphSAGE
from gnn_fraud.sage import GraphSAGE
from gnn_fraud.temporal import TemporalGraphSAGE


def train_val_split(labels: list[int], val_frac: float = 0.2,
                    seed: int = 0) -> tuple[list[int], list[int]]:
    """Stratified train/validation split of node ids."""
    rng = random.Random(seed)
    pos = [i for i, y in enumerate(labels) if y == 1]
    neg = [i for i, y in enumerate(labels) if y == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos_val = max(1, int(len(pos) * val_frac))
    n_neg_val = max(1, int(len(neg) * val_frac))
    val = pos[:n_pos_val] + neg[:n_neg_val]
    train = pos[n_pos_val:] + neg[n_neg_val:]
    rng.shuffle(train)
    return train, val


def _val_loss(model, graph, val_ids: list[int]) -> float:
    import math

    probs = model.predict(graph, val_ids)
    total = 0.0
    for v in val_ids:
        p = min(max(probs[v], 1e-7), 1.0 - 1e-7)
        y = graph.labels[v]
        total += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
    return total / len(val_ids)


def _train_loop(
    model,
    graph,
    train_ids: list[int],
    val_ids: list[int],
    k1: int,
    k2: int,
    batch_size: int,
    epochs: int,
    patience: int,
    seed: int,
    pos_weight: float,
):
    """Shared mini-batch loop: early-stop on validation loss, restore best."""
    labels = {v: graph.labels[v] for v in train_ids}
    rng = random.Random(seed)
    # Copy: shuffling must not reorder the caller's list, or training a
    # second model on the same ids would see a different batch sequence.
    train_ids = list(train_ids)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for _ in range(epochs):
        rng.shuffle(train_ids)
        batch_losses = []
        for i in range(0, len(train_ids), batch_size):
            batch = train_ids[i:i + batch_size]
            batch_losses.append(
                model.train_batch(graph, batch, labels, k1, k2,
                                  pos_weight=pos_weight))
        train_loss = sum(batch_losses) / len(batch_losses)
        val_loss = _val_loss(model, graph, val_ids)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = model.state_dict()
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_sage(
    graph,
    train_ids: list[int] | None = None,
    val_ids: list[int] | None = None,
    hidden_dim: int = 16,
    k1: int = 3,
    k2: int = 3,
    lr: float = 0.1,
    batch_size: int = 64,
    epochs: int = 30,
    patience: int = 5,
    seed: int = 0,
    pos_weight: float | None = None,
) -> tuple[GraphSAGE, dict]:
    """Train a 2-layer GraphSAGE with neighbor-sampled mini-batches.

    Early-stops on validation loss and restores the best checkpoint.
    Returns (model, history) where history holds per-epoch train/val loss.

    Args:
        pos_weight: upweighting of fraud examples in the training loss.
            Defaults to #neg/#pos on the training set (standard for
            imbalanced fraud data).
    """
    if train_ids is None or val_ids is None:
        train_ids, val_ids = train_val_split(graph.labels, seed=seed)
    if pos_weight is None:
        # Fraud data is imbalanced; a modest upweighting keeps the model
        # from collapsing to the base rate. (The full neg/pos ratio
        # over-corrects and predicts positive for everything.)
        pos_weight = 3.0

    model = GraphSAGE(graph.feature_dim, hidden_dim=hidden_dim, lr=lr, seed=seed)
    return _train_loop(model, graph, train_ids, val_ids, k1, k2, batch_size,
                       epochs, patience, seed, pos_weight)


def train_gat(
    graph,
    train_ids: list[int] | None = None,
    val_ids: list[int] | None = None,
    hidden_dim: int = 16,
    k1: int = 3,
    k2: int = 3,
    lr: float = 0.1,
    batch_size: int = 64,
    epochs: int = 30,
    patience: int = 5,
    seed: int = 0,
    pos_weight: float | None = None,
    negative_slope: float = 0.2,
) -> tuple[GraphAttention, dict]:
    """Train a 2-layer single-head GAT with neighbor-sampled mini-batches.

    Same protocol as :func:`train_sage` (early stopping on validation
    loss, best-checkpoint restore). Returns (model, history).

    Args:
        negative_slope: LeakyReLU slope used in the attention mechanism.
        pos_weight: upweighting of fraud examples in the training loss.
    """
    if train_ids is None or val_ids is None:
        train_ids, val_ids = train_val_split(graph.labels, seed=seed)
    if pos_weight is None:
        pos_weight = 3.0

    model = GraphAttention(graph.feature_dim, hidden_dim=hidden_dim, lr=lr,
                           seed=seed, negative_slope=negative_slope)
    return _train_loop(model, graph, train_ids, val_ids, k1, k2, batch_size,
                       epochs, patience, seed, pos_weight)


def train_temporal(
    graph,
    train_ids: list[int] | None = None,
    val_ids: list[int] | None = None,
    hidden_dim: int = 16,
    k1: int = 3,
    k2: int = 3,
    lr: float = 0.1,
    batch_size: int = 64,
    epochs: int = 30,
    patience: int = 5,
    seed: int = 0,
    pos_weight: float | None = None,
    log_tau: float | None = None,
) -> tuple[TemporalGraphSAGE, dict]:
    """Train a 2-layer temporal GraphSAGE on a timestamped graph.

    Same protocol as :func:`train_sage` (neighbor-sampled mini-batches,
    early stopping on validation loss, best-checkpoint restore), but each
    layer weights neighbors by recency with a learned decay timescale.
    Raises ``ValueError`` if the graph's edges have no timestamps.

    Args:
        log_tau: initial log-timescale for both layers (τ = exp(log_tau)
            days). Defaults to log(5 days). τ is learned per layer from
            there.
        pos_weight: upweighting of fraud examples in the training loss.
    """
    import math

    if train_ids is None or val_ids is None:
        train_ids, val_ids = train_val_split(graph.labels, seed=seed)
    if pos_weight is None:
        pos_weight = 3.0
    if log_tau is None:
        log_tau = math.log(5.0)

    model = TemporalGraphSAGE(graph.feature_dim, hidden_dim=hidden_dim,
                              lr=lr, seed=seed, log_tau=log_tau)
    return _train_loop(model, graph, train_ids, val_ids, k1, k2, batch_size,
                       epochs, patience, seed, pos_weight)


def train_hetero(
    graph,
    train_ids: list[int] | None = None,
    val_ids: list[int] | None = None,
    hidden_dim: int = 16,
    k1: int = 3,
    k2: int = 3,
    lr: float = 0.1,
    batch_size: int = 64,
    epochs: int = 30,
    patience: int = 5,
    seed: int = 0,
    pos_weight: float | None = None,
) -> tuple[RelationalGraphSAGE, dict]:
    """Train a 2-layer relational GraphSAGE on a heterogeneous graph.

    Only account nodes are split, scored, and supervised; devices and
    merchants participate in message passing but are never labeled.
    Otherwise the same protocol as :func:`train_sage` (neighbor-sampled
    mini-batches, early stopping on validation loss, best-checkpoint
    restore).
    """
    accounts = graph.account_ids()
    if train_ids is None or val_ids is None:
        tr, va = train_val_split([graph.labels[v] for v in accounts],
                                 seed=seed)
        train_ids, val_ids = [accounts[i] for i in tr], [accounts[i] for i in va]
    if pos_weight is None:
        pos_weight = 3.0

    model = RelationalGraphSAGE(graph.type_dims, hidden_dim=hidden_dim,
                                lr=lr, seed=seed)
    return _train_loop(model, graph, train_ids, val_ids, k1, k2, batch_size,
                       epochs, patience, seed, pos_weight)


def train_edge_cls(
    graph,
    encoder,
    train_edges: list[tuple[int, int]] | None = None,
    val_edges: list[tuple[int, int]] | None = None,
    hidden_dim: int = 16,
    lr: float = 0.1,
    batch_size: int = 64,
    epochs: int = 40,
    patience: int = 8,
    seed: int = 0,
    pos_weight: float = 1.0,
) -> tuple:
    """Train an edge classifier on top of a frozen GNN encoder.

    The encoder (typically a GraphSAGE trained with :func:`train_sage`)
    is NOT updated here: this freezes what the node model learned and
    trains only the edge head, an MLP over endpoint-pair features
    (see :mod:`gnn_fraud.edge_cls`). Embeddings come from a deterministic
    full-neighborhood forward pass, so inference is reproducible.

    Args:
        graph: the (homogeneous) transaction graph.
        encoder: any model whose ``forward`` returns ``(probs, cache)``
            with ``cache["h2"]`` -- GraphSAGE, GAT, RelationalGraphSAGE,
            and TemporalGraphSAGE all qualify.
        train_edges / val_edges: explicit edge split as (u, v), u < v.
            If omitted, edges are stratified 75/25 with ``seed``.
        pos_weight: multiplier on the loss of ring-internal edges.

    Returns (head, embeddings, history): the trained
    :class:`~gnn_fraud.edge_cls.EdgeHead`, the frozen node embeddings
    (for scoring), and the per-epoch train/val loss history.
    """
    nodes = list(range(graph.n))
    embeddings = full_embeddings(encoder, graph, nodes)
    labels = edge_labels(graph)
    edges = sorted(labels)
    head, history = train_edge_head(
        embeddings, edges, labels, train_edges, val_edges,
        hidden_dim=hidden_dim, lr=lr, batch_size=batch_size,
        epochs=epochs, patience=patience, seed=seed,
        pos_weight=pos_weight)
    return head, embeddings, history

"""Inductive evaluation: score unseen nodes with a frozen model, no retraining.

GraphSAGE's headline claim over transductive GNNs is *inductive* learning.
The model here keeps no per-node parameters -- only the aggregation
matrices (W_self, W_agg per layer) and the output layer -- so a trained
model can score accounts it never saw during training, including accounts
on a freshly generated graph. Inference is a pure function of a node's
features and its sampled neighborhood.

This module implements the honest evaluation protocol for that claim:

1. Train on graph A; tune the operating threshold on A's validation split
   and FREEZE it. Re-tuning on the holdout graph would leak its labels
   into the evaluation -- the frozen threshold is what a production
   system would actually deploy.
2. Generate a fresh graph B with the same generator (new seed: new ring,
   new wiring, new feature draws) and score its accounts.
3. Report precision/recall/F1 on B at the FROZEN threshold, plus AUC --
   the threshold-free metric, which is the cleanest inductive signal.

The API takes a ``predict_fn`` instead of a model object so the same
protocol covers every model in the package: ``model.predict`` for
GraphSAGE / GAT / the relational model, and a thin adapter for the
feature-only MLP.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from gnn_fraud.evaluate import best_threshold_f1, evaluate_all

# predict_fn(graph, ids) -> mapping id -> fraud probability.
PredictFn = Callable[[object, Sequence[int]], Mapping[int, float]]


def score_new_nodes(predict_fn: PredictFn, graph, node_ids=None):
    """Score nodes the model never saw during training.

    Args:
        predict_fn: ``(graph, ids) -> dict[id, probability]``. For the GNN
            models this is just ``model.predict`` (deterministic
            full-neighborhood inference); for the MLP baseline it is a
            small adapter over feature vectors.
        graph: the holdout graph -- typically a freshly generated one the
            model never trained on.
        node_ids: node ids to score. Defaults to every node carrying a
            0/1 label, so heterogeneous graphs score accounts only and
            skip unlabeled node types.

    Returns ``(y_true, y_score, scored_ids)``: parallel label and
    probability lists plus the node ids they correspond to.
    """
    if node_ids is None:
        node_ids = [v for v in range(graph.n) if graph.labels[v] in (0, 1)]
    node_ids = list(node_ids)
    if not node_ids:
        raise ValueError("no labeled nodes to score")
    probs = predict_fn(graph, node_ids)
    y_true = [graph.labels[v] for v in node_ids]
    y_score = [probs[v] for v in node_ids]
    return y_true, y_score, node_ids


def inductive_evaluate(predict_fn: PredictFn, graph, node_ids=None,
                       threshold: float | None = None) -> dict:
    """Evaluate a trained model on unseen nodes.

    Args:
        predict_fn: ``(graph, ids) -> dict[id, probability]``.
        graph: holdout graph with unseen nodes (labels are read here --
            in production these would be the accounts under review).
        node_ids: nodes to score; default all 0/1-labeled nodes.
        threshold: frozen decision threshold, tuned on the *training*
            graph's validation split -- the honest inductive protocol.
            If None, the threshold is tuned on the holdout set itself,
            which is optimistic; the result says so.

    Returns a dict with precision/recall/F1 at the applied threshold,
    AUC (threshold-free), the threshold used, whether it was frozen,
    and node counts.
    """
    y_true, y_score, scored = score_new_nodes(predict_fn, graph, node_ids)
    frozen = threshold is not None
    if not frozen:
        threshold, _ = best_threshold_f1(y_true, y_score)
    metrics = evaluate_all(y_true, y_score, threshold=threshold)
    metrics.update(
        {
            "threshold": threshold,
            "threshold_frozen": frozen,
            "n_scored": len(scored),
            "n_positive": sum(y_true),
        }
    )
    return metrics

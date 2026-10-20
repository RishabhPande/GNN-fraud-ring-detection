"""gnn-fraud: GraphSAGE for fraud detection, from scratch in pure Python.

Everything in this package operates on SYNTHETIC data only. The generator
plants a fraud ring in a random transaction graph so GNN detectors can be
built, tested, and benchmarked without touching real accounts, real
transactions, or real PII.
"""

from gnn_fraud.baseline import MLP, train_mlp
from gnn_fraud.edge_cls import (
    EdgeHead,
    edge_labels,
    edge_phi,
    edge_train_val_split,
    full_embeddings,
    score_edges,
    train_edge_head,
)
from gnn_fraud.evaluate import (
    best_threshold_f1,
    evaluate_all,
    precision_recall_f1,
    roc_auc,
)
from gnn_fraud.gat import GATLayer, GraphAttention
from gnn_fraud.generator import FEATURE_NAMES, GraphGenerator
from gnn_fraud.graph import Graph
from gnn_fraud.hetero import (
    ACCOUNT,
    DEVICE,
    MERCHANT,
    PAYS,
    PAID_BY,
    TRANSFERS,
    USED_BY,
    USES,
    HeteroGraph,
    RELATION_NAMES,
)
from gnn_fraud.hetero_gen import HeteroGenerator
from gnn_fraud.inductive import inductive_evaluate, score_new_nodes
from gnn_fraud.rsage import RelSAGELayer, RelationalGraphSAGE
from gnn_fraud.sage import GraphSAGE, SAGELayer
from gnn_fraud.temporal import TemporalGraphSAGE, TemporalSAGELayer, recency_weights
from gnn_fraud.train import (
    train_edge_cls,
    train_gat,
    train_hetero,
    train_sage,
    train_temporal,
    train_val_split,
)

__all__ = [
    "ACCOUNT",
    "DEVICE",
    "FEATURE_NAMES",
    "MERCHANT",
    "MLP",
    "PAYS",
    "PAID_BY",
    "TRANSFERS",
    "USED_BY",
    "USES",
    "GATLayer",
    "Graph",
    "GraphAttention",
    "GraphGenerator",
    "GraphSAGE",
    "HeteroGenerator",
    "HeteroGraph",
    "RELATION_NAMES",
    "RelSAGELayer",
    "RelationalGraphSAGE",
    "SAGELayer",
    "TemporalGraphSAGE",
    "TemporalSAGELayer",
    "EdgeHead",
    "best_threshold_f1",
    "edge_labels",
    "edge_phi",
    "edge_train_val_split",
    "evaluate_all",
    "full_embeddings",
    "inductive_evaluate",
    "precision_recall_f1",
    "recency_weights",
    "roc_auc",
    "score_edges",
    "score_new_nodes",
    "train_edge_cls",
    "train_edge_head",
    "train_gat",
    "train_hetero",
    "train_mlp",
    "train_sage",
    "train_temporal",
    "train_val_split",
]

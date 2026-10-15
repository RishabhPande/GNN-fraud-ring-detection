"""gnn-fraud: GraphSAGE for fraud detection, from scratch in pure Python.

Everything in this package operates on SYNTHETIC data only. The generator
plants a fraud ring in a random transaction graph so GNN detectors can be
built, tested, and benchmarked without touching real accounts, real
transactions, or real PII.
"""

from gnn_fraud.baseline import MLP, train_mlp
from gnn_fraud.evaluate import (
    best_threshold_f1,
    evaluate_all,
    precision_recall_f1,
    roc_auc,
)
from gnn_fraud.generator import FEATURE_NAMES, GraphGenerator
from gnn_fraud.graph import Graph
from gnn_fraud.sage import GraphSAGE, SAGELayer
from gnn_fraud.train import train_sage, train_val_split

__all__ = [
    "FEATURE_NAMES",
    "MLP",
    "Graph",
    "GraphGenerator",
    "GraphSAGE",
    "SAGELayer",
    "best_threshold_f1",
    "evaluate_all",
    "precision_recall_f1",
    "roc_auc",
    "train_mlp",
    "train_sage",
    "train_val_split",
]

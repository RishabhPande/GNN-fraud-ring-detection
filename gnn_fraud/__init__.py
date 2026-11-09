"""gnn-fraud: GraphSAGE for fraud detection, from scratch in pure Python.

Everything in this package operates on SYNTHETIC data only. The generator
plants a fraud ring in a random transaction graph so GNN detectors can be
built, tested, and benchmarked without touching real accounts, real
transactions, or real PII.
"""

from gnn_fraud.appnp import APPNP, propagate_predictions
from gnn_fraud.baseline import MLP, train_mlp
from gnn_fraud.cheb import ChebLayer, GraphChebyshev
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
from gnn_fraud.gatv2 import GATv2Layer, GraphAttentionV2
from gnn_fraud.gcn import GCNLayer, GraphConvolution
from gnn_fraud.gcnii import GCNIIConv, GraphGCNII
from gnn_fraud.gin import GINLayer, GraphIsomorphism
from gnn_fraud.gpr import GPRGNN, ppr_gamma_init, propagate_hidden
from gnn_fraud.h2gcn import GraphH2GCN, H2GCNLayer
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
from gnn_fraud.jk import JumpingKnowledgeGCN
from gnn_fraud.mixhop import GraphMixHop, MixHopLayer
from gnn_fraud.pna import GraphPNA, PNALayer
from gnn_fraud.rsage import RelSAGELayer, RelationalGraphSAGE
from gnn_fraud.sage import GraphSAGE, SAGELayer
from gnn_fraud.sgc import SimpleGraphConvolution, propagate_features
from gnn_fraud.sign import SIGN, sign_features
from gnn_fraud.tag import GraphTAG, TAGLayer
from gnn_fraud.temporal import TemporalGraphSAGE, TemporalSAGELayer, recency_weights
from gnn_fraud.train import (
    train_appnp,
    train_cheb,
    train_edge_cls,
    train_gcnii,
    train_gat,
    train_gatv2,
    train_gcn,
    train_gin,
    train_gpr,
    train_h2gcn,
    train_hetero,
    train_jk,
    train_mixhop,
    train_pna,
    train_sage,
    train_sgc,
    train_sign,
    train_tag,
    train_temporal,
    train_val_split,
)

__all__ = [
    "ACCOUNT",
    "APPNP",
    "DEVICE",
    "FEATURE_NAMES",
    "MERCHANT",
    "MLP",
    "PAYS",
    "PAID_BY",
    "TRANSFERS",
    "USED_BY",
    "USES",
    "ChebLayer",
    "GATLayer",
    "GATv2Layer",
    "GCNLayer",
    "GCNIIConv",
    "GraphChebyshev",
    "GraphGCNII",
    "GPRGNN",
    "GraphH2GCN",
    "H2GCNLayer",
    "GINLayer",
    "Graph",
    "GraphAttention",
    "GraphAttentionV2",
    "GraphConvolution",
    "GraphIsomorphism",
    "GraphMixHop",
    "GraphGenerator",
    "GraphPNA",
    "GraphSAGE",
    "GraphTAG",
    "HeteroGenerator",
    "HeteroGraph",
    "RELATION_NAMES",
    "PNALayer",
    "RelSAGELayer",
    "RelationalGraphSAGE",
    "SAGELayer",
    "SIGN",
    "SimpleGraphConvolution",
    "TAGLayer",
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
    "JumpingKnowledgeGCN",
    "MixHopLayer",
    "precision_recall_f1",
    "ppr_gamma_init",
    "propagate_features",
    "sign_features",
    "propagate_hidden",
    "propagate_predictions",
    "recency_weights",
    "roc_auc",
    "score_edges",
    "score_new_nodes",
    "train_appnp",
    "train_cheb",
    "train_edge_cls",
    "train_edge_head",
    "train_gat",
    "train_gatv2",
    "train_gcn",
    "train_gcnii",
    "train_gin",
    "train_gpr",
    "train_h2gcn",
    "train_hetero",
    "train_jk",
    "train_mixhop",
    "train_mlp",
    "train_pna",
    "train_sage",
    "train_sgc",
    "train_sign",
    "train_tag",
    "train_temporal",
    "train_val_split",
]

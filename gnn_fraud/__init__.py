"""gnn-fraud: GraphSAGE for fraud detection, from scratch in pure Python.

Everything in this package operates on SYNTHETIC data only. The generator
plants a fraud ring in a random transaction graph so GNN detectors can be
built, tested, and benchmarked without touching real accounts, real
transactions, or real PII.
"""

from gnn_fraud.agnn import AGNN, attention_weights_for, propagate_attention
from gnn_fraud.appnp import APPNP, propagate_predictions
from gnn_fraud.arma import ARMALayer, GraphARMA, normalized_propagate
from gnn_fraud.baseline import MLP, train_mlp
from gnn_fraud.bern import BernNet, bern_theta_init, bernstein_filter_coeffs, bernstein_terms
from gnn_fraud.cheb import ChebLayer, GraphChebyshev
from gnn_fraud.dagnn import DAGNN, hop_attention
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
from gnn_fraud.fagcn import FAGCN, frequency_split
from gnn_fraud.film import FiLMLayer, GraphFiLM
from gnn_fraud.gat import GATLayer, GraphAttention
from gnn_fraud.gatv2 import GATv2Layer, GraphAttentionV2
from gnn_fraud.gcn import GCNLayer, GraphConvolution
from gnn_fraud.gconv import GraphConvLayer, GraphConvNet
from gnn_fraud.ggnn import GGNN
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
from gnn_fraud.jacobi import (
    JacobiConv,
    jacobi_filter_coeffs,
    jacobi_polys,
    jacobi_terms,
    jacobi_theta_init,
)
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
    train_agnn,
    train_appnp,
    train_arma,
    train_bern,
    train_cheb,
    train_dagnn,
    train_edge_cls,
    train_fagcn,
    train_film,
    train_gcnii,
    train_gat,
    train_ggnn,
    train_gatv2,
    train_gcn,
    train_gconv,
    train_gin,
    train_gpr,
    train_h2gcn,
    train_hetero,
    train_jacobi,
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
    "AGNN",
    "APPNP",
    "ARMALayer",
    "BernNet",
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
    "DAGNN",
    "FiLMLayer",
    "GATLayer",
    "GATv2Layer",
    "GCNLayer",
    "GCNIIConv",
    "GraphChebyshev",
    "GraphGCNII",
    "GPRGNN",
    "GraphARMA",
    "GraphH2GCN",
    "H2GCNLayer",
    "GGNN",
    "GINLayer",
    "Graph",
    "GraphAttention",
    "GraphAttentionV2",
    "GraphFiLM",
    "GraphConvLayer",
    "GraphConvNet",
    "GraphConvolution",
    "GraphIsomorphism",
    "GraphMixHop",
    "GraphGenerator",
    "GraphPNA",
    "GraphSAGE",
    "GraphTAG",
    "JacobiConv",
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
    "FAGCN",
    "frequency_split",
    "attention_weights_for",
    "bern_theta_init",
    "bernstein_filter_coeffs",
    "bernstein_terms",
    "best_threshold_f1",
    "edge_labels",
    "edge_phi",
    "edge_train_val_split",
    "evaluate_all",
    "full_embeddings",
    "hop_attention",
    "inductive_evaluate",
    "jacobi_filter_coeffs",
    "jacobi_polys",
    "jacobi_terms",
    "jacobi_theta_init",
    "JumpingKnowledgeGCN",
    "MixHopLayer",
    "precision_recall_f1",
    "normalized_propagate",
    "ppr_gamma_init",
    "propagate_attention",
    "propagate_features",
    "sign_features",
    "propagate_hidden",
    "propagate_predictions",
    "recency_weights",
    "roc_auc",
    "score_edges",
    "score_new_nodes",
    "train_agnn",
    "train_appnp",
    "train_arma",
    "train_bern",
    "train_cheb",
    "train_dagnn",
    "train_edge_cls",
    "train_edge_head",
    "train_fagcn",
    "train_film",
    "train_gat",
    "train_gatv2",
    "train_ggnn",
    "train_gcn",
    "train_gconv",
    "train_gcnii",
    "train_gin",
    "train_gpr",
    "train_h2gcn",
    "train_hetero",
    "train_jacobi",
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

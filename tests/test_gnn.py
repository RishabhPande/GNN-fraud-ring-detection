"""Tests for gnn-fraud: hand-computed message passing, training dynamics,
generator planting, metrics, and sampling."""

import math
import random

from gnn_fraud import (
    Graph,
    GraphGenerator,
    GraphSAGE,
    SAGELayer,
    best_threshold_f1,
    evaluate_all,
    precision_recall_f1,
    roc_auc,
    train_mlp,
    train_sage,
    train_val_split,
)


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


# ---------------------------------------------------------------------------
# graph data structure
# ---------------------------------------------------------------------------


def test_graph_edges_symmetric():
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(0, 1)  # duplicate is a no-op
    assert g.neighbors(0) == [1]
    assert g.neighbors(1) == [0]
    assert g.degree(0) == 1
    assert g.edge_count() == 1


def test_graph_density_clique():
    g = Graph(5, 2)
    for i in range(5):
        for j in range(i + 1, 5):
            g.add_edge(i, j)
    approx(g.density(list(range(5))), 1.0)
    approx(g.density([0, 1, 2]), 1.0)
    assert g.density([0]) == 0.0


# ---------------------------------------------------------------------------
# message passing: hand-computable case
# ---------------------------------------------------------------------------


def test_sage_layer_forward_hand_computed():
    # 2 nodes, 0 -- 1. Identity weights, zero bias:
    #   h_0 = relu([1,0] + mean([0,2])) = relu([1,2]) = [1,2]
    layer = SAGELayer(2, 2, random.Random(0))
    layer.W_self = [[1.0, 0.0], [0.0, 1.0]]
    layer.W_agg = [[1.0, 0.0], [0.0, 1.0]]
    layer.b = [0.0, 0.0]
    out, cache = layer.forward([1.0, 0.0], [[0.0, 2.0]])
    approx(out[0], 1.0)
    approx(out[1], 2.0)

    # Empty neighborhood -> aggregation is the zero vector.
    out2, _ = layer.forward([1.0, 0.0], [])
    approx(out2[0], 1.0)
    approx(out2[1], 0.0)


def test_sage_layer_backward_hand_computed():
    layer = SAGELayer(2, 2, random.Random(0))
    layer.W_self = [[1.0, 0.0], [0.0, 1.0]]
    layer.W_agg = [[1.0, 0.0], [0.0, 1.0]]
    layer.b = [0.0, 0.0]
    _, cache = layer.forward([1.0, 0.0], [[0.0, 2.0]])
    d_h_self, d_neighs = layer.backward([1.0, 1.0], cache)
    # d_pre = [1,1] (both pre-activations > 0)
    assert layer.gW_self == [[1.0, 0.0], [1.0, 0.0]]
    assert layer.gW_agg == [[0.0, 2.0], [0.0, 2.0]]
    assert layer.gb == [1.0, 1.0]
    # identity weights: W^T @ [1,1] = [1,1]
    approx(d_h_self[0], 1.0)
    approx(d_h_self[1], 1.0)
    # mean mode: per-neighbor gradients; the single neighbor gets the full
    # aggregation gradient (same contract as backward_temporal)
    assert len(d_neighs) == 1
    approx(d_neighs[0][0], 1.0)
    approx(d_neighs[0][1], 1.0)


def test_sage_layer_relu_blocks_negative_gradient():
    layer = SAGELayer(1, 1, random.Random(0))
    layer.W_self = [[1.0]]
    layer.W_agg = [[1.0]]
    layer.b = [-5.0]  # pre = 1*2 + 1*0 - 5 < 0 -> dead unit
    _, cache = layer.forward([2.0], [])
    d_h_self, d_agg = layer.backward([3.0], cache)
    assert layer.gW_self == [[0.0]]
    assert layer.gb == [0.0]
    assert d_h_self == [0.0]


# ---------------------------------------------------------------------------
# neighbor sampling
# ---------------------------------------------------------------------------


def test_sample_respects_k():
    model = GraphSAGE(4, hidden_dim=4, seed=0)
    got = model._sample([1, 2, 3, 4, 5], 3)
    assert len(got) == 3
    assert set(got) <= {1, 2, 3, 4, 5}
    assert model._sample([1, 2], 5) == [1, 2]  # fewer than k -> all
    assert model._sample([1, 2, 3], None) == [1, 2, 3]  # None -> full


# ---------------------------------------------------------------------------
# training dynamics
# ---------------------------------------------------------------------------


def _small_graph():
    gen = GraphGenerator(seed=123)
    return gen.generate(n=60, ring_size=8)


def test_sage_loss_decreases():
    g = _small_graph()
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=1)
    _, hist = train_sage(g, train_ids, val_ids, hidden_dim=8, k1=3, k2=3,
                         lr=0.1, batch_size=32, epochs=8, patience=8, seed=1)
    assert hist["train_loss"][-1] < hist["train_loss"][0]


def test_mlp_loss_decreases():
    g = _small_graph()
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=1)
    _, hist = train_mlp(g.features, g.labels, train_ids, val_ids,
                        hidden_dim=8, lr=0.1, batch_size=32,
                        epochs=8, patience=8, seed=1)
    assert hist["train_loss"][-1] < hist["train_loss"][0]


def test_train_val_split_stratified():
    g = _small_graph()
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=1)
    assert sum(g.labels[v] for v in val_ids) >= 1
    assert sum(g.labels[v] for v in train_ids) >= 1
    assert len(set(train_ids) & set(val_ids)) == 0
    assert len(train_ids) + len(val_ids) == g.n


def test_sage_predict_deterministic():
    g = _small_graph()
    train_ids, val_ids = train_val_split(g.labels, seed=1)
    model, _ = train_sage(g, train_ids, val_ids, epochs=2, patience=2, seed=1)
    p1 = model.predict(g, val_ids)
    p2 = model.predict(g, val_ids)
    assert p1 == p2  # full-neighborhood inference has no sampling noise


# ---------------------------------------------------------------------------
# generator: the ring is planted and camouflaged
# ---------------------------------------------------------------------------


def test_generator_plants_ring():
    gen = GraphGenerator(seed=42)
    g = gen.generate(n=200, ring_size=18)
    ring = gen.ring_nodes(g)
    assert len(ring) == 18
    assert sum(g.labels) == 18
    # The ring is a dense subgraph...
    assert g.density(ring) > 0.4
    # ...in a sparse graph.
    assert g.edge_count() / (g.n * (g.n - 1) / 2) < 0.05


def test_generator_near_camouflage():
    # Most features match benign; the three behavioral tells are shifted
    # only slightly (weak individually -- the GNN denoises them).
    gen = GraphGenerator(seed=42)
    g = gen.generate(n=300, ring_size=24)
    ring = gen.ring_nodes(g)
    benign = [u for u in range(g.n) if g.labels[u] == 0]

    def means(nodes):
        return [sum(g.features[u][j] for u in nodes) / len(nodes)
                for j in range(g.feature_dim)]

    mr, mb = means(ring), means(benign)
    idx = {name: j for j, name in
           enumerate(["account_age", "avg_txn_amount", "txn_count",
                      "balance", "night_txn_ratio", "unique_counterparties"])}
    # camouflaged features: means close
    for name in ["account_age", "avg_txn_amount", "balance"]:
        assert abs(mr[idx[name]] - mb[idx[name]]) < 0.15, name
    # tells: shifted upward, but weakly (each < 0.15)
    for name in ["txn_count", "night_txn_ratio", "unique_counterparties"]:
        shift = mr[idx[name]] - mb[idx[name]]
        assert 0.0 < shift < 0.15, (name, shift)


def test_generator_seedable():
    g1 = GraphGenerator(seed=9).generate(n=100, ring_size=10)
    g2 = GraphGenerator(seed=9).generate(n=100, ring_size=10)
    assert g1.labels == g2.labels
    assert g1.features == g2.features
    assert [g1.neighbors(u) for u in range(100)] == \
           [g2.neighbors(u) for u in range(100)]


# ---------------------------------------------------------------------------
# metrics on known cases
# ---------------------------------------------------------------------------


def test_metrics_perfect_predictions():
    y = [0, 0, 1, 1, 1]
    s = [0.1, 0.2, 0.8, 0.9, 0.7]
    m = precision_recall_f1(y, s)
    approx(m["precision"], 1.0)
    approx(m["recall"], 1.0)
    approx(m["f1"], 1.0)
    approx(roc_auc(y, s), 1.0)


def test_roc_auc_known_value():
    # positives 0.35, 0.8 vs negatives 0.1, 0.4 -> 3 of 4 pairs won = 0.75
    approx(roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)


def test_roc_auc_ties_count_half():
    approx(roc_auc([0, 1], [0.5, 0.5]), 0.5)


def test_metrics_all_negative_predictions():
    m = precision_recall_f1([0, 1, 1], [0.1, 0.2, 0.3])
    approx(m["precision"], 0.0)
    approx(m["recall"], 0.0)
    approx(m["f1"], 0.0)


def test_evaluate_all_keys():
    m = evaluate_all([0, 1], [0.2, 0.9])
    assert set(m) >= {"precision", "recall", "f1", "auc"}


def test_best_threshold_f1():
    # Scores rank perfectly; the tuner should find a threshold that
    # separates them with F1 = 1.
    y = [0, 0, 1, 1]
    s = [0.1, 0.3, 0.6, 0.9]
    thr, m = best_threshold_f1(y, s)
    assert m["f1"] == 1.0
    assert 0.3 < thr <= 0.6


def test_gnn_beats_feature_baseline():
    # The learning story, pinned by seed: message passing over the ring's
    # dense neighborhood must beat node features alone. Deterministic --
    # every RNG involved is seeded.
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
    y_true = [g.labels[v] for v in val_ids]

    sage, _ = train_sage(g, train_ids, val_ids, hidden_dim=16,
                         k1=5, k2=5, lr=0.1, batch_size=64,
                         epochs=40, patience=40, seed=7)
    mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                       hidden_dim=16, lr=0.3, batch_size=32,
                       epochs=100, patience=100, seed=7)
    sage_auc = roc_auc(y_true, [sage.predict(g, [v])[v] for v in val_ids])
    mlp_auc = roc_auc(y_true, mlp.predict([g.features[v] for v in val_ids]))
    assert sage_auc > 0.85, f"sage auc {sage_auc}"
    assert sage_auc > mlp_auc + 0.15, f"sage {sage_auc} vs mlp {mlp_auc}"

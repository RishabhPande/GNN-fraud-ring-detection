"""Tests for Jumping Knowledge GCN: concatenation semantics, numeric
gradient checks through the layer split, checkpointing, and training
dynamics on the dense ring."""

import math

from gnn_fraud import (
    Graph,
    GraphGenerator,
    JumpingKnowledgeGCN,
    evaluate_all,
    train_gcn,
    train_jk,
    train_mlp,
    train_val_split,
)


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def _path_graph():
    g = Graph(3, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    return g


def _identity_model():
    # Both layers identity (bias zero), so h1 and h2 are both
    # normalized-aggregation outputs, and the head alone decides
    # which layer's embedding scores a node.
    model = JumpingKnowledgeGCN(2, hidden_dim=2, lr=0.0, seed=0)
    for layer in (model.layer1, model.layer2):
        layer.W = [[1.0, 0.0], [0.0, 1.0]]
        layer.b = [0.0, 0.0]
    return model


def test_jk_head_reads_both_layers():
    g = _path_graph()
    model = _identity_model()
    # Head reads only h1[0].
    model.w_out = [1.0, 0.0, 0.0, 0.0]
    model.b_out = 0.0
    p_h1 = model.predict(g, [0, 1, 2])
    # Head reads only h2[0]: different embedding, so at least one
    # node's score must move (h2 is a second aggregation pass).
    model.w_out = [0.0, 0.0, 1.0, 0.0]
    p_h2 = model.predict(g, [0, 1, 2])
    assert any(abs(p_h1[v] - p_h2[v]) > 1e-6 for v in (0, 1, 2))
    for v in (0, 1, 2):
        assert 0.0 < p_h1[v] < 1.0
        assert 0.0 < p_h2[v] < 1.0


def test_jk_head_width_is_two_hidden():
    model = JumpingKnowledgeGCN(6, hidden_dim=16, lr=0.5, seed=0)
    assert len(model.w_out) == 32
    assert len(model.gw_out) == 32
    assert model.layer1.out_dim == 16
    assert model.layer2.out_dim == 16


def test_jk_model_gradient_check():
    # Full model + concat head, numeric-checked end to end. Layer 1
    # is the interesting case: perturbing it moves the score both
    # directly (head's first half) and through layer 2, so the check
    # fails if the split gradient is not added.
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    g.set_features(3, [0.1, 0.4])
    labels = {0: 1, 1: 0, 2: 1, 3: 0}
    model = JumpingKnowledgeGCN(2, hidden_dim=3, lr=0.0, seed=5)

    def loss_fn():
        probs, _ = model.forward(g, [0, 1, 2, 3])
        total = 0.0
        for v in [0, 1, 2, 3]:
            p = min(max(probs[v], 1e-7), 1.0 - 1e-7)
            y = labels[v]
            total += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
        return total / 4

    # lr=0.0, so train_batch accumulates gradients without moving weights.
    model.train_batch(g, [0, 1, 2, 3], labels)
    eps = 1e-6
    for layer in (model.layer1, model.layer2):
        gW = layer.gW
        for i in range(len(layer.W)):
            for j in range(len(layer.W[0])):
                orig = layer.W[i][j]
                layer.W[i][j] = orig + eps
                plus = loss_fn()
                layer.W[i][j] = orig - eps
                minus = loss_fn()
                layer.W[i][j] = orig
                approx(gW[i][j], (plus - minus) / (2 * eps), tol=1e-5)
        for i in range(len(layer.b)):
            orig = layer.b[i]
            layer.b[i] = orig + eps
            plus = loss_fn()
            layer.b[i] = orig - eps
            minus = loss_fn()
            layer.b[i] = orig
            approx(layer.gb[i], (plus - minus) / (2 * eps), tol=1e-5)
    for j in range(6):
        orig = model.w_out[j]
        model.w_out[j] = orig + eps
        plus = loss_fn()
        model.w_out[j] = orig - eps
        minus = loss_fn()
        model.w_out[j] = orig
        approx(model.gw_out[j], (plus - minus) / (2 * eps), tol=1e-5)
    orig = model.b_out
    model.b_out = orig + eps
    plus = loss_fn()
    model.b_out = orig - eps
    minus = loss_fn()
    model.b_out = orig
    approx(model.gb_out, (plus - minus) / (2 * eps), tol=1e-5)


def test_jk_state_dict_round_trip():
    g = _path_graph()
    model = JumpingKnowledgeGCN(2, hidden_dim=3, lr=0.5, seed=5)
    before = model.predict(g, [0, 1, 2])
    state = model.state_dict()
    model2 = JumpingKnowledgeGCN(2, hidden_dim=3, lr=0.5, seed=99)
    model2.load_state_dict(state)
    after = model2.predict(g, [0, 1, 2])
    for v in (0, 1, 2):
        approx(before[v], after[v])


def test_jk_trains_to_high_auc_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
    model, _ = train_jk(g, train_ids, val_ids, hidden_dim=16,
                        lr=0.5, batch_size=64, epochs=100,
                        patience=25, seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    y_true = [g.labels[v] for v in val_ids]
    metrics = evaluate_all(y_true, scores)
    assert metrics["auc"] >= 0.90, metrics

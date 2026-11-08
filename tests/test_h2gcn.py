"""Tests for H2GCN: hand-computed forward/backward, numeric gradient
checks through the ego/neighbor separation and the 2-hop aggregate,
checkpointing, and training dynamics on the dense ring."""

import math
import random

from gnn_fraud import (
    Graph,
    GraphGenerator,
    GraphH2GCN,
    H2GCNLayer,
    evaluate_all,
    train_gcn,
    train_h2gcn,
    train_mlp,
    train_val_split,
)
from gnn_fraud.h2gcn import _row_norm_weights


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def _two_node_graph():
    """2 nodes, 0 -- 1, 1-dim features 1.0 and 2.0."""
    g = Graph(2, 1)
    g.add_edge(0, 1)
    g.set_features(0, [1.0])
    g.set_features(1, [2.0])
    return g


def _layer_with_weights():
    layer = H2GCNLayer(1, 1, rng=random.Random(0))
    layer.W = [[0.0, 0.0, 0.0]]
    layer.b = [0.0]
    return layer


# ---------------------------------------------------------------------------
# forward: hand-computable cases
# ---------------------------------------------------------------------------


def test_h2gcn_row_norm_weights_no_self_loop():
    # No self-loop: each row is 1 / degree over true neighbors only.
    g = Graph(5, 1)
    for u in range(1, 5):
        g.add_edge(0, u)
    weights = _row_norm_weights(g)
    approx(sum(c for _, c in weights[0]), 1.0)
    approx(weights[0][0][1], 0.25)
    # A leaf's only neighbor is the hub, weight 1.
    assert weights[1] == [(0, 1.0)]
    # An isolated node gets an empty row, not a division by zero.
    g2 = Graph(2, 1)
    g2.add_edge(0, 1)
    assert _row_norm_weights(g2)[0] == [(1, 1.0)]
    g3 = Graph(3, 1)
    g3.add_edge(0, 1)
    assert _row_norm_weights(g3)[2] == []


def test_h2gcn_layer_forward_ego_block_only():
    # W reads only the ego block: output is the node's own feature.
    g = _two_node_graph()
    layer = _layer_with_weights()
    layer.W = [[1.0, 0.0, 0.0]]
    out, _ = layer.forward(g, g.features)
    assert out[0] == [1.0]
    assert out[1] == [2.0]


def test_h2gcn_layer_forward_one_hop_block_only():
    # W reads only the 1-hop block: each node sees its neighbor.
    g = _two_node_graph()
    layer = _layer_with_weights()
    layer.W = [[0.0, 1.0, 0.0]]
    out, _ = layer.forward(g, g.features)
    assert out[0] == [2.0]
    assert out[1] == [1.0]


def test_h2gcn_layer_forward_two_hop_block_only():
    # W reads only the 2-hop block: on a 2-node graph the 2-hop
    # neighborhood of each node is itself (0 -> 1 -> 0).
    g = _two_node_graph()
    layer = _layer_with_weights()
    layer.W = [[0.0, 0.0, 1.0]]
    out, _ = layer.forward(g, g.features)
    assert out[0] == [1.0]
    assert out[1] == [2.0]


def test_h2gcn_layer_forward_two_hop_on_star():
    # Star: center 0 (feature 10) connected to leaves 1, 2, 3
    # (features 1, 2, 3). The 2-hop aggregate is the mean over
    # 2-hop *walks*: the center hears its own feature back through
    # each leaf (10), and each leaf hears the center's 1-hop mean
    # (2). A set-of-2-hop-neighbors reading would give the same
    # numbers here, so the check is the walk mean, not the set.
    g = Graph(4, 1)
    for u in (1, 2, 3):
        g.add_edge(0, u)
    g.set_features(0, [10.0])
    g.set_features(1, [1.0])
    g.set_features(2, [2.0])
    g.set_features(3, [3.0])
    layer = _layer_with_weights()
    layer.W = [[0.0, 0.0, 1.0]]
    out, _ = layer.forward(g, g.features)
    assert out[0] == [10.0]
    assert out[1] == [2.0]
    assert out[2] == [2.0]
    assert out[3] == [2.0]


def test_h2gcn_layer_forward_isolated_node_zero_aggregate():
    # An isolated node keeps its ego block and gets zeros for both
    # neighbor blocks: no crash, no NaN.
    g = Graph(3, 1)
    g.add_edge(0, 1)
    g.set_features(0, [1.0])
    g.set_features(1, [2.0])
    g.set_features(2, [5.0])
    layer = _layer_with_weights()
    layer.W = [[1.0, 1.0, 1.0]]
    out, _ = layer.forward(g, g.features)
    assert out[0] == [4.0]  # 1 (ego) + 2 (1-hop) + 1 (2-hop, itself)
    assert out[1] == [5.0]  # 2 + 1 + 2 (2-hop back through node 0)
    assert out[2] == [5.0]  # 5 + 0 + 0


def test_h2gcn_layer_forward_all_blocks():
    # Every block at weight 1 on the 2-node graph: node 0 sees
    # ego 1 + 1-hop 2 + 2-hop 1 = 4; node 1 sees 2 + 1 + 2 = 5.
    g = _two_node_graph()
    layer = _layer_with_weights()
    layer.W = [[1.0, 1.0, 1.0]]
    out, _ = layer.forward(g, g.features)
    assert out[0] == [4.0]
    assert out[1] == [5.0]


# ---------------------------------------------------------------------------
# backward: numeric gradient checks
# ---------------------------------------------------------------------------


def test_h2gcn_layer_gradient_check():
    g = Graph(3, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.set_features(0, [1.0, 0.5])
    g.set_features(1, [0.0, 2.0])
    g.set_features(2, [-1.0, 1.0])
    layer = H2GCNLayer(2, 2, rng=random.Random(1))
    out, cache = layer.forward(g, g.features)
    d_out = [[1.0, -0.5], [0.5, 1.0], [-1.0, 0.5]]
    layer.zero_grad()
    d_H = layer.backward(d_out, cache)

    def loss_fn():
        o, _ = layer.forward(g, g.features)
        return sum(d * x for row_d, row_o in zip(d_out, o)
                   for d, x in zip(row_d, row_o))

    eps = 1e-6
    for i in range(2):
        for j in range(6):
            orig = layer.W[i][j]
            layer.W[i][j] = orig + eps
            l1 = loss_fn()
            layer.W[i][j] = orig - eps
            l2 = loss_fn()
            layer.W[i][j] = orig
            num = (l1 - l2) / (2 * eps)
            assert abs(num - layer.gW[i][j]) < 1e-5, \
                f"W[{i}][{j}]: {num} != {layer.gW[i][j]}"
    for i in range(2):
        orig = layer.b[i]
        layer.b[i] = orig + eps
        l1 = loss_fn()
        layer.b[i] = orig - eps
        l2 = loss_fn()
        layer.b[i] = orig
        num = (l1 - l2) / (2 * eps)
        assert abs(num - layer.gb[i]) < 1e-5, \
            f"b[{i}]: {num} != {layer.gb[i]}"

    # d_H: numeric-check the input gradient too, since the transpose
    # path through A and A^2 is the one place the row normalization
    # (not symmetric, unlike TAGCN) can bite.
    H = [list(row) for row in g.features]
    for v in range(3):
        for j in range(2):
            orig = H[v][j]
            H[v][j] = orig + eps
            o, _ = layer.forward(g, H)
            l1 = sum(d * x for row_d, row_o in zip(d_out, o)
                     for d, x in zip(row_d, row_o))
            H[v][j] = orig - eps
            o, _ = layer.forward(g, H)
            l2 = sum(d * x for row_d, row_o in zip(d_out, o)
                     for d, x in zip(row_d, row_o))
            H[v][j] = orig
            num = (l1 - l2) / (2 * eps)
            assert abs(num - d_H[v][j]) < 1e-5, \
                f"d_H[{v}][{j}]: {num} != {d_H[v][j]}"


def test_h2gcn_model_gradient_check():
    # Full model (embedding + two layers + head), numeric-checked
    # end to end over every parameter, including the embedding
    # weights that feed both layer 1 and the head's r0 block.
    g = Graph(3, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.set_features(0, [1.0, 0.5])
    g.set_features(1, [0.0, 2.0])
    g.set_features(2, [-1.0, 1.0])
    model = GraphH2GCN(2, hidden_dim=3, lr=0.1, seed=0)
    targets = [0, 1, 2]
    labels = {0: 1, 1: 0, 2: 1}

    def loss_fn():
        probs, _ = model.forward(g, targets)
        total = 0.0
        for v in targets:
            p = min(max(probs[v], 1e-7), 1.0 - 1e-7)
            total += -(labels[v] * math.log(p)
                       + (1 - labels[v]) * math.log(1.0 - p))
        return total / len(targets)

    # One manual backward pass to collect analytic gradients: run
    # the same gradient flow train_batch uses, without the update.
    model.layer1.zero_grad()
    model.layer2.zero_grad()
    model.gW_e = [[0.0] * 2 for _ in range(3)]
    model.gb_e = [0.0] * 3
    model.gw_out = [0.0] * 9
    model.gb_out = 0.0
    probs, cache = model.forward(g, targets)
    final = cache["final"]
    d_final = [[0.0] * 9 for _ in range(3)]
    gw_out = [0.0] * 9
    gb_out = 0.0
    for v in targets:
        d_logit = (probs[v] - labels[v]) / len(targets)
        for j in range(9):
            gw_out[j] += d_logit * final[v][j]
        gb_out += d_logit
        d_final[v] = [w * d_logit for w in model.w_out]
    h = 3
    d_r0 = [[0.0] * h for _ in range(3)]
    d_r1 = [[0.0] * h for _ in range(3)]
    d_r2 = [[0.0] * h for _ in range(3)]
    for v in range(3):
        d_r0[v] = list(d_final[v][0:h])
        d_r1[v] = list(d_final[v][h:2 * h])
        d_r2[v] = list(d_final[v][2 * h:3 * h])
    d_r1_from_l2 = model.layer2.backward(d_r2, cache["cache2"])
    for v in range(3):
        for j in range(h):
            d_r1[v][j] += d_r1_from_l2[v][j]
    d_r0_from_l1 = model.layer1.backward(d_r1, cache["cache1"])
    for v in range(3):
        for j in range(h):
            d_r0[v][j] += d_r0_from_l1[v][j]
    for v in range(3):
        dv = d_r0[v]
        rv = cache["r0"][v]
        xv = g.features[v]
        for i in range(h):
            if rv[i] <= 0.0:
                continue
            di = dv[i]
            for j in range(2):
                model.gW_e[i][j] += di * xv[j]
            model.gb_e[i] += di

    eps = 1e-6

    def check_param(get, set_, analytic, name):
        orig = get()
        set_(orig + eps)
        l1 = loss_fn()
        set_(orig - eps)
        l2 = loss_fn()
        set_(orig)
        num = (l1 - l2) / (2 * eps)
        assert abs(num - analytic) < 1e-5, f"{name}: {num} != {analytic}"

    for i in range(3):
        for j in range(2):
            check_param(
                lambda i=i, j=j: model.W_e[i][j],
                lambda val, i=i, j=j: model.W_e[i].__setitem__(j, val),
                model.gW_e[i][j], f"W_e[{i}][{j}]")
    for i in range(3):
        check_param(
            lambda i=i: model.b_e[i],
            lambda val, i=i: model.b_e.__setitem__(i, val),
            model.gb_e[i], f"b_e[{i}]")
    for layer, name in ((model.layer1, "layer1"),
                        (model.layer2, "layer2")):
        for i in range(3):
            for j in range(9):
                check_param(
                    lambda layer=layer, i=i, j=j: layer.W[i][j],
                    lambda val, layer=layer, i=i, j=j:
                        layer.W[i].__setitem__(j, val),
                    layer.gW[i][j], f"{name} W[{i}][{j}]")
            check_param(
                lambda layer=layer, i=i: layer.b[i],
                lambda val, layer=layer, i=i:
                    layer.b.__setitem__(i, val),
                layer.gb[i], f"{name} b[{i}]")
    for j in range(9):
        check_param(
            lambda j=j: model.w_out[j],
            lambda val, j=j: model.w_out.__setitem__(j, val),
            gw_out[j], f"w_out[{j}]")
    check_param(lambda: model.b_out,
                lambda val: setattr(model, "b_out", val),
                gb_out, "b_out")


def test_h2gcn_state_dict_round_trip():
    model = GraphH2GCN(2, hidden_dim=3, lr=0.1, seed=0)
    g = Graph(3, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.set_features(0, [1.0, 0.5])
    g.set_features(1, [0.0, 2.0])
    g.set_features(2, [-1.0, 1.0])
    before = model.predict(g, [0, 1, 2])
    state = model.state_dict()
    model2 = GraphH2GCN(2, hidden_dim=3, lr=0.1, seed=9)
    model2.load_state_dict(state)
    after = model2.predict(g, [0, 1, 2])
    for v in (0, 1, 2):
        approx(before[v], after[v])
    # Mutating the loaded copy must not touch the saved state.
    model2.w_out[0] = 99.0
    model3 = GraphH2GCN(2, hidden_dim=3, lr=0.1, seed=9)
    model3.load_state_dict(state)
    approx(model3.predict(g, [0])[0], before[0])


def test_h2gcn_layer_block_norms_sum():
    # The head's per-block norms are non-negative and sum in
    # quadrature to the full head norm.
    model = GraphH2GCN(5, hidden_dim=4, lr=0.1, seed=0)
    norms = model.layer_block_norms()
    assert len(norms) == 3
    assert all(n >= 0.0 for n in norms)
    approx(sum(n * n for n in norms),
           sum(w * w for w in model.w_out))


# ---------------------------------------------------------------------------
# training dynamics: the dense ring
# ---------------------------------------------------------------------------


def test_h2gcn_trains_to_high_auc_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
    model, hist = train_h2gcn(g, train_ids, val_ids, hidden_dim=16,
                              lr=0.5, batch_size=64, epochs=100,
                              patience=25, seed=7)
    y_true = [g.labels[v] for v in val_ids]
    probs = [model.predict(g, [v])[v] for v in val_ids]
    metrics = evaluate_all(y_true, probs)
    assert metrics["auc"] >= 0.90, f"H2GCN AUC {metrics['auc']:.3f}"
    assert len(hist["train_loss"]) > 1
    assert hist["val_loss"][-1] < hist["val_loss"][0]


def test_h2gcn_headline_vs_gcn_and_mlp():
    # Same protocol as the README headline: H2GCN against the plain
    # GCN it extends and the feature-only baseline. The claim is
    # only that H2GCN clears the bar the GCN family is held to
    # (AUC 0.90+) and beats the MLP; the exact ordering is reported
    # in the example script, not asserted here.
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
    y_true = [g.labels[v] for v in val_ids]

    h2, _ = train_h2gcn(g, train_ids, val_ids, hidden_dim=16, lr=0.5,
                        batch_size=64, epochs=100, patience=25, seed=7)
    gcn, _ = train_gcn(g, train_ids, val_ids, hidden_dim=16, lr=0.5,
                       batch_size=64, epochs=100, patience=25, seed=7)
    mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                       hidden_dim=16, lr=0.3, batch_size=32,
                       epochs=150, patience=20, seed=7)

    auc_h2 = evaluate_all(
        y_true, [h2.predict(g, [v])[v] for v in val_ids])["auc"]
    auc_gcn = evaluate_all(
        y_true, [gcn.predict(g, [v])[v] for v in val_ids])["auc"]
    auc_mlp = evaluate_all(
        y_true, mlp.predict([g.features[v] for v in val_ids]))["auc"]
    assert auc_h2 >= 0.90, f"H2GCN AUC {auc_h2:.3f}"
    assert auc_h2 > auc_mlp, \
        f"H2GCN {auc_h2:.3f} did not beat MLP {auc_mlp:.3f}"
    # H2GCN should not fall far behind the plain GCN it generalizes.
    assert auc_h2 >= auc_gcn - 0.05, \
        f"H2GCN {auc_h2:.3f} vs GCN {auc_gcn:.3f}"

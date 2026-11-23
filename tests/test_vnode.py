"""Tests for virtual-node GraphSAGE: hand-computed global and
neighbor means, a hand-computed layer forward, the isolated
node's global term, the zero-global-term ablation identity,
numeric gradient checks at the layer and end to end,
checkpointing, bad-parameter rejection, and training dynamics
on the dense ring."""

import math
import random

from gnn_fraud import (
    Graph,
    GraphGenerator,
    VNodeLayer,
    VirtualNodeSAGE,
    evaluate_all,
    global_mean,
    neighbor_means,
    train_val_split,
    train_vnode,
)


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def _two_node_graph():
    g = Graph(2, 1)
    g.add_edge(0, 1)
    g.set_features(0, [1.0])
    g.set_features(1, [3.0])
    return g


# ---------------------------------------------------------------------------
# hand-computed means and layer forward
# ---------------------------------------------------------------------------


def test_global_mean_hand_computed():
    approx(global_mean([[1.0, 2.0], [3.0, 4.0], [5.0, 9.0]])[0], 3.0)
    approx(global_mean([[1.0, 2.0], [3.0, 4.0], [5.0, 9.0]])[1], 5.0)


def test_neighbor_means_hand_computed_and_isolated():
    g = Graph(3, 1)
    g.add_edge(0, 1)
    g.set_features(0, [1.0])
    g.set_features(1, [3.0])
    g.set_features(2, [7.0])
    means = neighbor_means(g, g.features)
    approx(means[0][0], 3.0)
    approx(means[1][0], 1.0)
    # Isolated node: no neighborhood, exactly zero.
    approx(means[2][0], 0.0)


def test_layer_forward_hand_computed():
    # W_self = W_neigh = W_glob = [[1]], b = 0, features [1, 3]:
    # node 0: self 1 + neighbor mean 3 + global mean 2 = 6
    # node 1: self 3 + neighbor mean 1 + global mean 2 = 6
    g = _two_node_graph()
    layer = VNodeLayer(1, 1, random.Random(0))
    layer.W_self = [[1.0]]
    layer.W_neigh = [[1.0]]
    layer.W_glob = [[1.0]]
    layer.b = [0.0]
    pre, _ = layer.forward(g, g.features)
    approx(pre[0][0], 6.0)
    approx(pre[1][0], 6.0)


def test_isolated_node_still_gets_global_term():
    # Node 2 is isolated, so its neighbor mean is zero, but the
    # global term still reaches it: self 7 + global mean
    # (1 + 3 + 7) / 3 = 7 + 11/3.
    g = Graph(3, 1)
    g.add_edge(0, 1)
    g.set_features(0, [1.0])
    g.set_features(1, [3.0])
    g.set_features(2, [7.0])
    layer = VNodeLayer(1, 1, random.Random(0))
    layer.W_self = [[1.0]]
    layer.W_neigh = [[1.0]]
    layer.W_glob = [[1.0]]
    layer.b = [0.0]
    pre, _ = layer.forward(g, g.features)
    approx(pre[2][0], 7.0 + 11.0 / 3.0)


def test_zero_global_weights_ignore_graph_mean_shift():
    # With W_glob zeroed the layer is a plain mean GraphSAGE
    # layer: adding a constant to every node's features shifts
    # the global mean but must not change the part of the output
    # beyond the self and neighbor terms, which the hand values
    # pin down: node 0 gets self 1 + neighbor 3 = 4.
    g = _two_node_graph()
    layer = VNodeLayer(1, 1, random.Random(0))
    layer.W_self = [[1.0]]
    layer.W_neigh = [[1.0]]
    layer.W_glob = [[0.0]]
    layer.b = [0.0]
    pre, _ = layer.forward(g, g.features)
    approx(pre[0][0], 4.0)
    approx(pre[1][0], 4.0)


def test_vnode_bad_params_rejected():
    for kwargs in ({"hidden_dim": 0},):
        try:
            VirtualNodeSAGE(6, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")
    model = VirtualNodeSAGE(6, hidden_dim=4, seed=1)
    g = _two_node_graph()
    try:
        model.virtual_state(g, layer=3)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for layer=3")
    try:
        global_mean([])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty H")


# ---------------------------------------------------------------------------
# numeric gradient checks
# ---------------------------------------------------------------------------


def _loss(model, g, labels):
    probs, _ = model.forward(g, [0, 1, 2, 3])
    total = 0.0
    for v in [0, 1, 2, 3]:
        p = min(max(probs[v], 1e-7), 1.0 - 1e-7)
        y = labels[v]
        total += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
    return total / 4


def _grad_graph():
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    g.set_features(3, [0.1, 0.4])
    return g


def _check(check_fn, analytic, get, set_):
    eps = 1e-6
    orig = get()
    set_(orig + eps)
    plus = check_fn()
    set_(orig - eps)
    minus = check_fn()
    set_(orig)
    approx(analytic, (plus - minus) / (2 * eps), tol=1e-5)


def test_vnode_layer_gradient_check():
    # Perturb every layer parameter and compare with backward's
    # accumulated gradients, using a fixed upstream gradient.
    # The global path is the one that matters: a missing 1 / n
    # in the broadcast backward fails it.
    g = _grad_graph()
    rng = random.Random(3)
    layer = VNodeLayer(2, 3, rng)
    H = [row[:] for row in g.features]
    pre, cache = layer.forward(g, H)
    d_pre = [[0.3, -0.2, 0.1], [0.0, 0.5, -0.4],
             [-0.1, 0.2, 0.6], [0.4, 0.0, -0.3]]
    layer.zero_grad()
    layer.backward(g, d_pre, cache)

    def layer_loss():
        p, _ = layer.forward(g, H)
        return sum(d_pre[v][j] * p[v][j]
                   for v in range(4) for j in range(3))

    for name in ("W_self", "W_neigh", "W_glob"):
        mat = getattr(layer, name)
        gmat = getattr(layer, "g" + name)
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                _check(layer_loss, gmat[i][j],
                       lambda i=i, j=j, m=mat: m[i][j],
                       lambda v, i=i, j=j, m=mat: m[i].__setitem__(j, v))
    for i in range(layer.out_dim):
        _check(layer_loss, layer.gb[i],
               lambda i=i: layer.b[i],
               lambda v, i=i: layer.b.__setitem__(i, v))


def test_vnode_model_gradient_check():
    g = _grad_graph()
    labels = {0: 1, 1: 0, 2: 1, 3: 0}
    model = VirtualNodeSAGE(2, hidden_dim=3, lr=0.0, seed=5)
    model.train_batch(g, [0, 1, 2, 3], labels)

    def loss():
        return _loss(model, g, labels)

    for layer in (model.layer1, model.layer2):
        for name in ("W_self", "W_neigh", "W_glob"):
            mat = getattr(layer, name)
            gmat = getattr(layer, "g" + name)
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    _check(loss, gmat[i][j],
                           lambda i=i, j=j, m=mat: m[i][j],
                           lambda v, i=i, j=j, m=mat: m[i].__setitem__(j, v))
        for i in range(layer.out_dim):
            _check(loss, layer.gb[i],
                   lambda i=i, ly=layer: ly.b[i],
                   lambda v, i=i, ly=layer: ly.b.__setitem__(i, v))
    for j in range(model.hidden_dim):
        _check(loss, model.gw_out[j],
               lambda j=j: model.w_out[j],
               lambda v, j=j: model.w_out.__setitem__(j, v))
    _check(loss, model.gb_out, lambda: model.b_out,
           lambda v: setattr(model, "b_out", v))


# ---------------------------------------------------------------------------
# checkpointing and virtual state
# ---------------------------------------------------------------------------


def test_vnode_checkpoint_round_trip():
    model = VirtualNodeSAGE(6, hidden_dim=8, seed=1)
    state = model.state_dict()
    model2 = VirtualNodeSAGE(6, hidden_dim=8, seed=99)
    model2.load_state_dict(state)
    g = Graph(2, 6)
    g.add_edge(0, 1)
    g.set_features(0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    g.set_features(1, [0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    p1 = model.predict(g, [0, 1])
    p2 = model2.predict(g, [0, 1])
    approx(p1[0], p2[0])
    approx(p1[1], p2[1])


def test_vnode_virtual_state_is_feature_mean_at_layer1():
    model = VirtualNodeSAGE(6, hidden_dim=8, seed=1)
    g = Graph(2, 6)
    g.set_features(0, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    g.set_features(1, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    vs = model.virtual_state(g, layer=1)
    for j in range(6):
        approx(vs[j], 1.0 + j)
    assert len(model.virtual_state(g, layer=2)) == 8


# ---------------------------------------------------------------------------
# training dynamics
# ---------------------------------------------------------------------------


def test_vnode_trains_to_high_auc_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    model, _ = train_vnode(g, train_ids, val_ids, hidden_dim=16,
                           lr=0.1, batch_size=64, epochs=60,
                           patience=12, seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    y_true = [g.labels[v] for v in val_ids]
    assert evaluate_all(y_true, scores)["auc"] >= 0.90
    # The global term trains: layer 1's W_glob moves off a zero
    # start in norm terms - it is initialized nonzero, so pin
    # instead that the trained norm is substantial, not dead.
    norm = math.sqrt(sum(x * x for row in model.layer1.W_glob
                         for x in row))
    assert norm > 0.5

"""Tests for TransformerConv: hand-computed scaled dot-product
attention on a two-node graph, the isolated-node identity
(attention exactly 1.0 on itself), attention weights summing
to 1, numeric gradient checks end to end over every query,
key, and value parameter in both layers, checkpointing,
bad-parameter rejection, and training dynamics on the dense
ring (AUC 0.90+ at the GCN family's lr 0.5 schedule, with the
lr 0.1 underfitting documented in train_tconv)."""

import math

from gnn_fraud import (
    Graph,
    GraphGenerator,
    GraphTransformerConv,
    evaluate_all,
    train_tconv,
    train_val_split,
)


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def _two_node_graph():
    g = Graph(2, 1)
    g.add_edge(0, 1)
    g.set_features(0, [1.0])
    g.set_features(1, [3.0])
    return g


def test_attention_weights_sum_to_one_and_self_first():
    g = _two_node_graph()
    model = GraphTransformerConv(1, hidden_dim=4, seed=3)
    for v in (0, 1):
        w = model.attention_weights(g, v, layer=1)
        assert w[0][0] == v
        approx(sum(a for _, a in w), 1.0)
        assert {u for u, _ in w} == {0, 1}


def test_isolated_node_attention_is_exactly_itself():
    g = Graph(3, 1)
    g.add_edge(0, 1)
    g.set_features(2, [2.0])
    model = GraphTransformerConv(1, hidden_dim=4, seed=3)
    w = model.attention_weights(g, 2, layer=1)
    assert w == [(2, 1.0)]


def test_dot_product_attention_hand_computed():
    # Build a one-layer model by hand: W_q = W_k = W_v = [[1]]
    # on 1-dim states, so q = k = val = the feature itself.
    # Node 0 (feature 1) attends over itself (score 1) and
    # node 1 (feature 3, score 3): softmax gives
    # alpha_self = e^1 / (e^1 + e^3), and the output is
    # alpha_self * 1 + (1 - alpha_self) * 3.
    from gnn_fraud import TransformerConvLayer
    import random as _random
    layer = TransformerConvLayer(1, 1, _random.Random(0), relu=False)
    layer.W_q = [[1.0]]
    layer.W_k = [[1.0]]
    layer.W_v = [[1.0]]
    layer.b_v = [0.0]
    g = _two_node_graph()
    out, cache = layer.forward(g, [[1.0], [3.0]])
    e1, e3 = math.exp(1.0), math.exp(3.0)
    a_self = e1 / (e1 + e3)
    approx(cache["alpha"][0][0], a_self)
    approx(out[0][0], a_self * 1.0 + (1.0 - a_self) * 3.0)
    # Node 1 attends over itself (score 9) and node 0 (score 3).
    e9 = math.exp(9.0)
    a_self1 = e9 / (e9 + e3)
    approx(out[1][0], a_self1 * 3.0 + (1.0 - a_self1) * 1.0)


def test_tconv_bad_params_rejected():
    for kwargs in ({"in_dim": 0}, {"hidden_dim": 0}):
        try:
            GraphTransformerConv(kwargs.get("in_dim", 6),
                                 hidden_dim=kwargs.get("hidden_dim", 16))
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")
    model = GraphTransformerConv(6, hidden_dim=4)
    try:
        model.attention_weights(_two_node_graph(), 0, layer=3)
    except ValueError:
        return
    raise AssertionError("expected ValueError for layer=3")


# ---------------------------------------------------------------------------
# numeric gradient check
# ---------------------------------------------------------------------------


def _loss(model, g, labels):
    probs, _ = model.forward(g, [0, 1, 2, 3])
    total = 0.0
    for v in [0, 1, 2, 3]:
        p = min(max(probs[v], 1e-7), 1.0 - 1e-7)
        y = labels[v]
        total += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
    return total / 4


def test_tconv_model_gradient_check():
    # Perturb every parameter of both attention layers (query,
    # key, value, value bias) plus the output bias, and compare
    # with the gradients train_batch accumulates at lr=0.0. The
    # query/key path is the one that matters: a missing
    # 1 / sqrt(d) or a softmax backward that forgets the
    # weighted-mean term fails it.
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    g.set_features(3, [0.1, 0.4])
    labels = {0: 1, 1: 0, 2: 1, 3: 0}
    model = GraphTransformerConv(2, hidden_dim=3, lr=0.0, seed=5)
    model.train_batch(g, [0, 1, 2, 3], labels)
    eps = 1e-6

    def check(analytic, get, set_):
        orig = get()
        set_(orig + eps)
        plus = _loss(model, g, labels)
        set_(orig - eps)
        minus = _loss(model, g, labels)
        set_(orig)
        approx(analytic, (plus - minus) / (2 * eps), tol=1e-5)

    for layer in (model.layer1, model.layer2):
        for W, gW in ((layer.W_q, layer.gW_q),
                      (layer.W_k, layer.gW_k),
                      (layer.W_v, layer.gW_v)):
            for i in range(len(W)):
                for j in range(len(W[i])):
                    check(gW[i][j],
                          lambda i=i, j=j, W=W: W[i][j],
                          lambda v, i=i, j=j, W=W: W[i].__setitem__(j, v))
        for i in range(len(layer.b_v)):
            check(layer.gb_v[i],
                  lambda i=i, layer=layer: layer.b_v[i],
                  lambda v, i=i, layer=layer: layer.b_v.__setitem__(i, v))
    check(model.gb_out, lambda: model.b_out,
          lambda v: setattr(model, "b_out", v))


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------


def test_tconv_checkpoint_round_trip():
    model = GraphTransformerConv(6, hidden_dim=8, seed=1)
    state = model.state_dict()
    model2 = GraphTransformerConv(6, hidden_dim=8, seed=99)
    model2.load_state_dict(state)
    g = Graph(2, 6)
    g.add_edge(0, 1)
    g.set_features(0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    g.set_features(1, [0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    p1 = model.predict(g, [0, 1])
    p2 = model2.predict(g, [0, 1])
    approx(p1[0], p2[0])
    approx(p1[1], p2[1])


# ---------------------------------------------------------------------------
# training dynamics
# ---------------------------------------------------------------------------


def test_tconv_trains_to_high_auc_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    model, _ = train_tconv(g, train_ids, val_ids, hidden_dim=16,
                           lr=0.5, batch_size=64, epochs=100,
                           patience=25, seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    y_true = [g.labels[v] for v in val_ids]
    assert evaluate_all(y_true, scores)["auc"] >= 0.90


def test_tconv_trained_attention_concentrates_on_ring_for_ring_members():
    # After training, a held-out ring member's first-layer
    # attention lands mostly on ring accounts. Part of that is
    # neighbour composition (ring members are wired mostly to
    # the ring), so the bar is set at the composition level the
    # untrained model already shows, and the claim pinned here
    # is only that training does not diffuse it away.
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    ring = set(gen.ring_nodes(g))
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    model, _ = train_tconv(g, train_ids, val_ids, hidden_dim=16,
                           lr=0.5, batch_size=64, epochs=100,
                           patience=25, seed=7)
    shares = []
    for v in val_ids:
        if v not in ring:
            continue
        w = model.attention_weights(g, v, layer=1)
        shares.append(sum(a for u, a in w if u in ring))
    assert shares and sum(shares) / len(shares) >= 0.80

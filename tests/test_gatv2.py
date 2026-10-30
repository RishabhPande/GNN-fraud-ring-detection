"""Tests for the GATv2 dynamic-attention layer: hand-computed
forward/backward, numeric gradient checks, the dynamic-ranking property
that separates GATv2 from GAT, and training dynamics on the dense ring."""

import math
import random

from gnn_fraud import (
    GATv2Layer,
    Graph,
    GraphAttentionV2,
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_gatv2,
    train_val_split,
)


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


# ---------------------------------------------------------------------------
# forward: hand-computable case
# ---------------------------------------------------------------------------


def test_gatv2_layer_forward_hand_computed():
    # W = [[1]], W_att = [[2, -1]], a = [1], b = [0], dim 1.
    # For v = 0, candidates [0, 1], h = {0: [1], 1: [2]}:
    #   pair(0, 0) = [1, 1] -> s = 2*1 - 1*1 = 1 -> e = 1
    #   pair(0, 1) = [1, 2] -> s = 2*1 - 1*2 = 0 -> e = 0
    # alpha = softmax([1, 0]); out = relu(alpha_0 * 1 + alpha_1 * 2).
    layer = GATv2Layer(1, 1, random.Random(0))
    layer.W = [[1.0]]
    layer.W_att = [[2.0, -1.0]]
    layer.a = [1.0]
    layer.b = [0.0]
    h = {0: [1.0], 1: [2.0]}
    out, cache = layer.forward(0, h, [1])

    a0 = math.exp(1.0) / (math.exp(1.0) + 1.0)
    a1 = 1.0 / (math.exp(1.0) + 1.0)
    approx(cache["head_caches"][0]["alphas"][0], a0)
    approx(cache["head_caches"][0]["alphas"][1], a1)
    approx(out[0], a0 * 1.0 + a1 * 2.0)


def test_gatv2_layer_uniform_attention_with_zero_attention_map():
    # With W_att = 0 every score is 0, so attention is uniform over
    # {v} union N(v): the layer reduces to self+mean aggregation.
    layer = GATv2Layer(2, 2, random.Random(0))
    layer.W = [[1.0, 0.0], [0.0, 1.0]]
    layer.W_att = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    layer.a = [0.0, 0.0]
    layer.b = [0.0, 0.0]
    h = {0: [1.0, 0.0], 1: [0.0, 2.0], 2: [3.0, 0.0]}
    out, cache = layer.forward(0, h, [1, 2])
    assert set(cache["head_caches"][0]["alphas"]) == {0, 1, 2}
    for a in cache["head_caches"][0]["alphas"].values():
        approx(a, 1.0 / 3.0)
    approx(out[0], (1.0 + 0.0 + 3.0) / 3.0)
    approx(out[1], (0.0 + 2.0 + 0.0) / 3.0)


def test_gatv2_layer_isolated_node():
    layer = GATv2Layer(1, 1, random.Random(0))
    layer.W = [[2.0]]
    layer.W_att = [[1.0, 1.0]]
    layer.a = [1.0]
    layer.b = [0.5]
    out, cache = layer.forward(0, {0: [1.0]}, [])
    approx(cache["head_caches"][0]["alphas"][0], 1.0)
    approx(out[0], 2.0 * 1.0 + 0.5)


# ---------------------------------------------------------------------------
# dynamic ranking: the property GAT cannot express
# ---------------------------------------------------------------------------


def test_gatv2_ranking_depends_on_query_node():
    # W_att rows [-2, -2] and [-1, -2], a = [-2, 2], in_dim = 1.
    # Neighbors carry h = 1 and h = -1. Hand-computed scores:
    #   query h_v = 1:  e(1) = 0.4, e(-1) = 2.0  -> neighbor -1 ranks first
    #   query h_v = -1: e(1) = -0.4, e(-1) = -2.0 -> neighbor 1 ranks first
    # GAT cannot do this: its a_self term is constant across neighbors,
    # so the neighbor ranking is query-independent. Softmax is monotone
    # in the scores, so the alpha ordering matches the score ordering.
    layer = GATv2Layer(1, 1, random.Random(0))
    layer.W = [[1.0]]
    layer.W_att = [[-2.0, -2.0], [-1.0, -2.0]]
    layer.a = [-2.0, 2.0]
    layer.b = [0.0]

    _, cache_hi = layer.forward(0, {0: [1.0], 1: [1.0], 2: [-1.0]}, [1, 2])
    alphas_hi = cache_hi["head_caches"][0]["alphas"]
    assert alphas_hi[2] > alphas_hi[1]

    _, cache_lo = layer.forward(0, {0: [-1.0], 1: [1.0], 2: [-1.0]}, [1, 2])
    alphas_lo = cache_lo["head_caches"][0]["alphas"]
    assert alphas_lo[1] > alphas_lo[2]


# ---------------------------------------------------------------------------
# backward: numeric gradient check over every parameter
# ---------------------------------------------------------------------------


def test_gatv2_layer_numeric_gradient_check():
    rng = random.Random(3)
    in_dim, out_dim = 3, 2
    layer = GATv2Layer(in_dim, out_dim, rng)
    h = {u: [rng.uniform(-1, 1) for _ in range(in_dim)] for u in (0, 1, 2)}
    out, cache = layer.forward(0, h, [1, 2])

    d_out = [0.7, -0.4]
    layer.zero_grad()
    d_h = layer.backward(d_out, cache)
    analytic = {
        "W": [row[:] for row in layer.gW],
        "W_att": [row[:] for row in layer.gW_att],
        "a": layer.ga[:],
        "b": layer.gb[:],
        "h": {u: d_h[u][:] for u in (0, 1, 2)},
    }

    def loss():
        o, _ = layer.forward(0, h, [1, 2])
        return sum(do * oi for do, oi in zip(d_out, o))

    eps = 1e-6

    def num_grad(getter, setter):
        vals = getter()
        grads = []
        for idx, val in enumerate(vals):
            setter(idx, val + eps)
            lp = loss()
            setter(idx, val - eps)
            lm = loss()
            setter(idx, val)
            grads.append((lp - lm) / (2 * eps))
        return grads

    for name, param, got in [
        ("W", layer.W, analytic["W"]),
        ("W_att", layer.W_att, analytic["W_att"]),
        ("a", layer.a, analytic["a"]),
        ("b", layer.b, analytic["b"]),
    ]:
        if name in ("W", "W_att"):
            num = []
            for i in range(len(param)):
                num.append(num_grad(
                    lambda i=i: param[i][:],
                    lambda j, v, i=i: param[i].__setitem__(j, v),
                ))
        else:
            num = num_grad(lambda: param[:],
                           lambda j, v: param.__setitem__(j, v))
        flat_got = [x for row in got for x in row] if name in ("W", "W_att") else got
        flat_num = [x for row in num for x in row] if name in ("W", "W_att") else num
        for a, n in zip(flat_got, flat_num):
            assert abs(a - n) < 1e-5, f"{name}: {a} vs {n}"

    for u in (0, 1, 2):
        num = num_grad(lambda u=u: h[u][:],
                       lambda j, v, u=u: h[u].__setitem__(j, v))
        for a, n in zip(analytic["h"][u], num):
            assert abs(a - n) < 1e-5, f"h[{u}]: {a} vs {n}"


def test_gatv2_layer_numeric_gradient_check_average_mode():
    # Same check with 2 heads in average mode, exercising the 1/K
    # gradient split on the second combination path.
    rng = random.Random(5)
    layer = GATv2Layer(2, 2, rng, n_heads=2, out_mode="average")
    h = {u: [rng.uniform(-1, 1) for _ in range(2)] for u in (0, 1, 2)}
    d_out = [0.7]
    _, cache = layer.forward(0, h, [1, 2])
    layer.zero_grad()
    d_h = layer.backward(d_out, cache)

    def loss():
        o, _ = layer.forward(0, h, [1, 2])
        return d_out[0] * o[0]

    eps = 1e-6
    for k in range(2):
        head, grad = layer.heads[k], layer.grads[k]
        for name, param, got in [
            ("W", head["W"], grad["W"]),
            ("W_att", head["W_att"], grad["W_att"]),
            ("a", head["a"], grad["a"]),
            ("b", head["b"], grad["b"]),
        ]:
            if name in ("W", "W_att"):
                for i in range(len(param)):
                    for j in range(len(param[i])):
                        old = param[i][j]
                        param[i][j] = old + eps
                        lp = loss()
                        param[i][j] = old - eps
                        lm = loss()
                        param[i][j] = old
                        assert abs(got[i][j] - (lp - lm) / (2 * eps)) < 1e-5, \
                            f"head {k} {name}[{i}][{j}]: {got[i][j]} vs {(lp - lm) / (2 * eps)}"
            else:
                for j in range(len(param)):
                    old = param[j]
                    param[j] = old + eps
                    lp = loss()
                    param[j] = old - eps
                    lm = loss()
                    param[j] = old
                    assert abs(got[j] - (lp - lm) / (2 * eps)) < 1e-5, \
                        f"head {k} {name}[{j}]: {got[j]} vs {(lp - lm) / (2 * eps)}"
    # Input gradients in average mode too.
    for u in (0, 1, 2):
        for j in range(2):
            old = h[u][j]
            h[u][j] = old + eps
            lp = loss()
            h[u][j] = old - eps
            lm = loss()
            h[u][j] = old
            assert abs(d_h[u][j] - (lp - lm) / (2 * eps)) < 1e-5, \
                f"h[{u}][{j}]: {d_h[u][j]} vs {(lp - lm) / (2 * eps)}"


# ---------------------------------------------------------------------------
# model-level behavior
# ---------------------------------------------------------------------------


def test_gatv2_train_loss_decreases():
    gen = GraphGenerator(seed=11)
    g = gen.generate(n=60, ring_size=8)
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=11)
    model, history = train_gatv2(g, train_ids, val_ids, hidden_dim=8,
                                 k1=3, k2=3, lr=0.2, batch_size=32,
                                 epochs=12, patience=12, seed=11)
    assert len(history["train_loss"]) >= 3
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_gatv2_predict_probas_in_range():
    gen = GraphGenerator(seed=12)
    g = gen.generate(n=50, ring_size=8)
    train_ids, val_ids = train_val_split(g.labels, seed=12)
    model, _ = train_gatv2(g, train_ids, val_ids, hidden_dim=8, epochs=4,
                           patience=4, seed=12)
    probs = model.predict(g, val_ids)
    assert set(probs) == set(val_ids)
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_gatv2_state_dict_round_trip():
    gen = GraphGenerator(seed=13)
    g = gen.generate(n=50, ring_size=8)
    train_ids, val_ids = train_val_split(g.labels, seed=13)
    model, _ = train_gatv2(g, train_ids, val_ids, hidden_dim=8, epochs=3,
                           patience=3, seed=13)
    before = model.predict(g, val_ids)
    state = model.state_dict()
    fresh = GraphAttentionV2(g.feature_dim, hidden_dim=8, seed=99)
    fresh.load_state_dict(state)
    after = fresh.predict(g, val_ids)
    for v in val_ids:
        approx(before[v], after[v], tol=1e-12)


def test_gatv2_attention_weights_are_distributions():
    gen = GraphGenerator(seed=14)
    g = gen.generate(n=60, ring_size=8)
    model = GraphAttentionV2(g.feature_dim, hidden_dim=8, seed=14)
    v = 0
    layers = model.attention_weights(g, v)
    assert len(layers) == 2
    for alphas in layers:
        assert set(alphas) == ({v} | set(g.neighbors(v)))
        approx(sum(alphas.values()), 1.0)
        assert all(a >= 0.0 for a in alphas.values())


def test_gatv2_trains_to_auc_090_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
    model, _ = train_gatv2(g, train_ids, val_ids, hidden_dim=16,
                           k1=5, k2=5, lr=0.5, batch_size=64,
                           epochs=60, patience=12, seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    y_true = [g.labels[v] for v in val_ids]
    auc = evaluate_all(y_true, scores)["auc"]
    assert auc >= 0.90, f"GATv2 dense-ring AUC {auc:.3f} below 0.90"

"""Tests for the GAT attention layer: hand-computed forward/backward,
numeric gradient checks, training dynamics, and attention introspection."""

import math
import random

from gnn_fraud import (
    GATLayer,
    Graph,
    GraphAttention,
    GraphGenerator,
    train_gat,
    train_val_split,
)


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def _tiny_line_graph():
    """2 nodes, 0 -- 1."""
    g = Graph(2, 1)
    g.add_edge(0, 1)
    g.set_features(0, [1.0])
    g.set_features(1, [2.0])
    g.set_label(0, 1)
    g.set_label(1, 0)
    return g


# ---------------------------------------------------------------------------
# forward: hand-computable case
# ---------------------------------------------------------------------------


def test_gat_layer_forward_hand_computed():
    # W = [[1]], a_self = [2], a_neigh = [-1], b = [0], dim 1.
    # For v = 0, candidates [0, 1]: Wh = [1], [2].
    #   z_00 = leaky(2*1 + (-1)*1) = leaky(1) = 1
    #   z_01 = leaky(2*1 + (-1)*2) = leaky(0) = 0
    # α = softmax([1, 0]) = [e/(e+1), 1/(e+1)]
    # out_0 = relu(α_0 * 1 + α_1 * 2)
    layer = GATLayer(1, 1, random.Random(0))
    layer.W = [[1.0]]
    layer.a_self = [2.0]
    layer.a_neigh = [-1.0]
    layer.b = [0.0]
    h = {0: [1.0], 1: [2.0]}
    out, cache = layer.forward(0, h, [1])

    a0 = math.exp(1.0) / (math.exp(1.0) + 1.0)
    a1 = 1.0 / (math.exp(1.0) + 1.0)
    approx(cache["head_caches"][0]["alphas"][0], a0)
    approx(cache["head_caches"][0]["alphas"][1], a1)
    approx(out[0], a0 * 1.0 + a1 * 2.0)


def test_gat_layer_uniform_attention_with_zero_attention_weights():
    # With a_self = a_neigh = 0, every logit is 0, so attention is uniform
    # over {v} ∪ N(v): the layer reduces to self+mean aggregation.
    layer = GATLayer(2, 2, random.Random(0))
    layer.W = [[1.0, 0.0], [0.0, 1.0]]
    layer.a_self = [0.0, 0.0]
    layer.a_neigh = [0.0, 0.0]
    layer.b = [0.0, 0.0]
    h = {0: [1.0, 0.0], 1: [0.0, 2.0], 2: [3.0, 0.0]}
    out, cache = layer.forward(0, h, [1, 2])
    assert set(cache["head_caches"][0]["alphas"]) == {0, 1, 2}
    for a in cache["head_caches"][0]["alphas"].values():
        approx(a, 1.0 / 3.0)
    # uniform mean over self + neighbors, then relu
    approx(out[0], (1.0 + 0.0 + 3.0) / 3.0)
    approx(out[1], (0.0 + 2.0 + 0.0) / 3.0)


def test_gat_layer_isolated_node():
    # Empty neighborhood: the only candidate is v itself, α = 1.
    layer = GATLayer(1, 1, random.Random(0))
    layer.W = [[2.0]]
    layer.a_self = [1.0]
    layer.a_neigh = [1.0]
    layer.b = [0.5]
    out, cache = layer.forward(0, {0: [1.0]}, [])
    approx(cache["head_caches"][0]["alphas"][0], 1.0)
    approx(out[0], 2.0 * 1.0 + 0.5)


# ---------------------------------------------------------------------------
# backward: hand-computed + numeric gradient check
# ---------------------------------------------------------------------------


def test_gat_layer_backward_uniform_case_hand_computed():
    # Uniform-attention setup from above: dL/dout = [1, 1], pre > 0.
    # agg = mean over 3 candidates; dL/dα_u = d_pre · Wh_u; W=I so Wh_u = h_u.
    layer = GATLayer(2, 2, random.Random(0))
    layer.W = [[1.0, 0.0], [0.0, 1.0]]
    layer.a_self = [0.0, 0.0]
    layer.a_neigh = [0.0, 0.0]
    layer.b = [0.0, 0.0]
    h = {0: [1.0, 0.0], 1: [0.0, 2.0], 2: [3.0, 0.0]}
    _, cache = layer.forward(0, h, [1, 2])
    d_h = layer.backward([1.0, 1.0], cache)

    # d_pre = [1,1]; gb should accumulate [1,1].
    assert layer.gb == [1.0, 1.0]
    # With uniform α and zero attention vectors, the softmax term cancels:
    # dL/dW accumulates Σ_u α_u·d_pre·h_u.
    # α=1/3: row i=0: (h[0]+h[1]+h[2])/3 = [4/3, 2/3]; row i=1: same.
    approx(layer.gW[0][0], 4.0 / 3.0)
    approx(layer.gW[0][1], 2.0 / 3.0)
    approx(layer.gW[1][0], 4.0 / 3.0)
    approx(layer.gW[1][1], 2.0 / 3.0)
    # d_h[v] gets the weighted-sum share (α=1/3) plus attention-path terms.
    # With a=0 the attention-path terms are 0, so d_h[u] = α·d_pre = [1/3,1/3].
    for u in (0, 1, 2):
        approx(d_h[u][0], 1.0 / 3.0)
        approx(d_h[u][1], 1.0 / 3.0)


def test_gat_layer_numeric_gradient_check():
    # Central-difference check of gW, ga_self, ga_neigh, gb and d_h
    # against the analytic backward pass.
    rng = random.Random(3)
    in_dim, out_dim = 3, 2
    layer = GATLayer(in_dim, out_dim, rng)
    h = {u: [rng.uniform(-1, 1) for _ in range(in_dim)] for u in (0, 1, 2)}
    out, cache = layer.forward(0, h, [1, 2])

    d_out = [0.7, -0.4]
    layer.zero_grad()
    d_h = layer.backward(d_out, cache)
    analytic = {
        "W": [row[:] for row in layer.gW],
        "a_self": layer.ga_self[:],
        "a_neigh": layer.ga_neigh[:],
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
        ("a_self", layer.a_self, analytic["a_self"]),
        ("a_neigh", layer.a_neigh, analytic["a_neigh"]),
        ("b", layer.b, analytic["b"]),
    ]:
        if name == "W":
            num = []
            for i in range(out_dim):
                num.append(num_grad(
                    lambda i=i: param[i][:],
                    lambda j, v, i=i: param[i].__setitem__(j, v),
                ))
        else:
            num = num_grad(lambda: param[:],
                           lambda j, v: param.__setitem__(j, v))
        flat_got = [x for row in got for x in row] if name == "W" else got
        flat_num = [x for row in num for x in row] if name == "W" else num
        for a, n in zip(flat_got, flat_num):
            assert abs(a - n) < 1e-5, f"{name}: {a} vs {n}"

    # Gradient w.r.t. input embeddings.
    for u in (0, 1, 2):
        num = num_grad(lambda u=u: h[u][:],
                       lambda j, v, u=u: h[u].__setitem__(j, v))
        for a, n in zip(analytic["h"][u], num):
            assert abs(a - n) < 1e-5, f"h[{u}]: {a} vs {n}"


# ---------------------------------------------------------------------------
# model-level behavior
# ---------------------------------------------------------------------------


def test_gat_train_loss_decreases():
    gen = GraphGenerator(seed=11)
    g = gen.generate(n=60, ring_size=8)
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=11)
    model, history = train_gat(g, train_ids, val_ids, hidden_dim=8,
                              k1=3, k2=3, lr=0.2, batch_size=32,
                              epochs=12, patience=12, seed=11)
    assert len(history["train_loss"]) >= 3
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_gat_predict_probas_in_range():
    gen = GraphGenerator(seed=12)
    g = gen.generate(n=50, ring_size=8)
    train_ids, val_ids = train_val_split(g.labels, seed=12)
    model, _ = train_gat(g, train_ids, val_ids, hidden_dim=8, epochs=4,
                         patience=4, seed=12)
    probs = model.predict(g, val_ids)
    assert set(probs) == set(val_ids)
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_gat_state_dict_round_trip():
    gen = GraphGenerator(seed=13)
    g = gen.generate(n=50, ring_size=8)
    train_ids, val_ids = train_val_split(g.labels, seed=13)
    model, _ = train_gat(g, train_ids, val_ids, hidden_dim=8, epochs=3,
                         patience=3, seed=13)
    before = model.predict(g, val_ids)
    state = model.state_dict()
    fresh = GraphAttention(g.feature_dim, hidden_dim=8, seed=99)
    fresh.load_state_dict(state)
    after = fresh.predict(g, val_ids)
    for v in val_ids:
        approx(before[v], after[v], tol=1e-12)


def test_gat_attention_weights_are_distributions():
    gen = GraphGenerator(seed=14)
    g = gen.generate(n=60, ring_size=8)
    model = GraphAttention(g.feature_dim, hidden_dim=8, seed=14)
    v = 0
    layers = model.attention_weights(g, v)
    assert len(layers) == 2
    for alphas in layers:
        assert set(alphas) == ({v} | set(g.neighbors(v)))
        approx(sum(alphas.values()), 1.0)
        assert all(a >= 0.0 for a in alphas.values())


# ---------------------------------------------------------------------------
# multi-head GAT
# ---------------------------------------------------------------------------


def _two_head_layer():
    """Deterministic 2-head, head_dim=1 layer for hand computation."""
    layer = GATLayer(1, 2, random.Random(0), n_heads=2, out_mode="concat")
    # Head 0: W=[[1]], a_self=[2], a_neigh=[-1], b=[0.5]
    layer.heads[0]["W"] = [[1.0]]
    layer.heads[0]["a_self"] = [2.0]
    layer.heads[0]["a_neigh"] = [-1.0]
    layer.heads[0]["b"] = [0.5]
    # Head 1: zero attention params -> uniform alphas, W=0, b=-1
    layer.heads[1]["W"] = [[0.0]]
    layer.heads[1]["a_self"] = [0.0]
    layer.heads[1]["a_neigh"] = [0.0]
    layer.heads[1]["b"] = [-1.0]
    return layer


def test_multihead_concat_forward_hand_computed():
    # h = {0: [1], 1: [2]}, v = 0, neighbors = [1].
    # Head 0: Wh = [1], [2]; z_00 = leaky(2*1 - 1*1) = 1;
    #   z_01 = leaky(2*1 - 1*2) = 0; alphas = softmax([1, 0]).
    #   pre_0 = a0*1 + a1*2 + 0.5.
    # Head 1: zero params -> uniform alphas, Wh = 0 -> pre_1 = -1.
    # concat pre = [pre_0, -1]; relu kills the negative head-1 slot.
    layer = _two_head_layer()
    out, cache = layer.forward(0, {0: [1.0], 1: [2.0]}, [1])

    a0 = math.exp(1.0) / (math.exp(1.0) + 1.0)
    a1 = 1.0 / (math.exp(1.0) + 1.0)
    expected_pre0 = a0 * 1.0 + a1 * 2.0 + 0.5
    approx(out[0], expected_pre0)
    approx(out[1], 0.0)
    # Per-head caches are retained: head 1 really is uniform.
    hcs = cache["head_caches"]
    approx(hcs[1]["alphas"][0], 0.5)
    approx(hcs[1]["alphas"][1], 0.5)
    approx(hcs[0]["alphas"][0], a0)


def test_multihead_average_forward_hand_computed():
    # Same weights, average mode: pre = (pre_0 + pre_1) / 2, head_dim = 1.
    layer = _two_head_layer()
    layer.out_mode = "average"
    out, _ = layer.forward(0, {0: [1.0], 1: [2.0]}, [1])

    a0 = math.exp(1.0) / (math.exp(1.0) + 1.0)
    a1 = 1.0 / (math.exp(1.0) + 1.0)
    pre0 = a0 * 1.0 + a1 * 2.0 + 0.5
    pre1 = -1.0
    approx(out[0], (pre0 + pre1) / 2.0 if (pre0 + pre1) / 2.0 > 0 else 0.0)


def test_multihead_single_head_equivalence():
    # n_heads=1 must reproduce the legacy single-head math bit-for-bit.
    rng_a, rng_b = random.Random(5), random.Random(5)
    legacy = GATLayer(3, 4, rng_a)          # n_heads=1, concat: legacy path
    multi = GATLayer(3, 4, rng_b, n_heads=1, out_mode="concat")
    h = {u: [rng_a.uniform(-1, 1) for _ in range(3)] for u in (0, 1, 2)}
    out_l, cache_l = legacy.forward(0, h, [1, 2])
    out_m, cache_m = multi.forward(0, h, [1, 2])
    for x, y in zip(out_l, out_m):
        approx(x, y)
    d_out = [0.3, -0.7, 0.1, 0.9]
    d_l = legacy.backward(d_out, cache_l)
    d_m = multi.backward(d_out, cache_m)
    for u in (0, 1, 2):
        for x, y in zip(d_l[u], d_m[u]):
            approx(x, y)
    for name in ("gW", "ga_self", "ga_neigh", "gb"):
        gl, gm = getattr(legacy, name), getattr(multi, name)
        flat_l = [x for row in gl for x in row] if name == "gW" else gl
        flat_m = [x for row in gm for x in row] if name == "gW" else gm
        for x, y in zip(flat_l, flat_m):
            approx(x, y)


def test_multihead_bad_config_raises():
    rng = random.Random(0)
    try:
        GATLayer(4, 5, rng, n_heads=2)
        raise AssertionError("expected ValueError for non-divisible out_dim")
    except ValueError:
        pass
    try:
        GATLayer(4, 4, rng, n_heads=2, out_mode="sum")
        raise AssertionError("expected ValueError for bad out_mode")
    except ValueError:
        pass


def _numeric_check_layer(layer, in_dim, n_heads):
    rng = random.Random(21)
    h = {u: [rng.uniform(-1, 1) for _ in range(in_dim)] for u in (0, 1, 2)}
    out, cache = layer.forward(0, h, [1, 2])
    d_out = [rng.uniform(-1, 1) for _ in range(layer.out_dim)]
    layer.zero_grad()
    d_h = layer.backward(d_out, cache)
    analytic = {
        "h": {u: d_h[u][:] for u in (0, 1, 2)},
        "heads": [
            {
                "W": [row[:] for row in g["W"]],
                "a_self": g["a_self"][:],
                "a_neigh": g["a_neigh"][:],
                "b": g["b"][:],
            }
            for g in layer.grads
        ],
    }

    def loss():
        o, _ = layer.forward(0, h, [1, 2])
        return sum(do * oi for do, oi in zip(d_out, o))

    eps = 1e-6

    def num_grad(params):
        vals = list(params)
        grads = []
        for j, v in enumerate(vals):
            params[j] = v + eps
            lp = loss()
            params[j] = v - eps
            lm = loss()
            params[j] = v
            grads.append((lp - lm) / (2 * eps))
        return grads

    # Input embeddings.
    for u in (0, 1, 2):
        for a, n in zip(analytic["h"][u], num_grad(h[u])):
            assert abs(a - n) < 1e-5, f"h[{u}]: {a} vs {n}"
    # Per-head parameters.
    for k in range(n_heads):
        head = layer.heads[k]
        ag = analytic["heads"][k]
        for name, dim2 in [("W", True), ("a_self", False),
                           ("a_neigh", False), ("b", False)]:
            p = head[name]
            if dim2:
                for i in range(len(p)):
                    got = ag[name][i]
                    num = num_grad(p[i])
                    for a, n in zip(got, num):
                        assert abs(a - n) < 1e-5, f"head {k} {name}: {a} vs {n}"
            else:
                for a, n in zip(ag[name], num_grad(p)):
                    assert abs(a - n) < 1e-5, f"head {k} {name}: {a} vs {n}"


def test_multihead_numeric_gradient_check_concat():
    layer = GATLayer(3, 4, random.Random(21), n_heads=2, out_mode="concat")
    _numeric_check_layer(layer, 3, 2)


def test_multihead_numeric_gradient_check_average():
    layer = GATLayer(3, 4, random.Random(22), n_heads=2, out_mode="average")
    _numeric_check_layer(layer, 3, 2)


def test_multihead_model_trains_and_predicts():
    gen = GraphGenerator(seed=31)
    g = gen.generate(n=60, ring_size=8)
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=31)
    model, history = train_gat(g, train_ids, val_ids, hidden_dim=8,
                              k1=3, k2=3, lr=0.2, batch_size=32,
                              epochs=12, patience=12, seed=31, n_heads=2)
    assert model.n_heads == 2
    assert len(history["train_loss"]) >= 3
    assert history["train_loss"][-1] < history["train_loss"][0]
    probs = model.predict(g, val_ids)
    assert set(probs) == set(val_ids)
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_multihead_attention_per_head_are_distributions():
    gen = GraphGenerator(seed=32)
    g = gen.generate(n=50, ring_size=8)
    model = GraphAttention(g.feature_dim, hidden_dim=8, seed=32, n_heads=2)
    v = 0
    layers = model.attention_weights_per_head(g, v)
    assert len(layers) == 2
    for heads in layers:
        assert len(heads) == 2
        for alphas in heads:
            assert set(alphas) == ({v} | set(g.neighbors(v)))
            approx(sum(alphas.values()), 1.0)
            assert all(a >= 0.0 for a in alphas.values())
    # The averaged view stays a distribution too (gat_example contract).
    for alphas in model.attention_weights(g, v):
        approx(sum(alphas.values()), 1.0)


def test_multihead_state_dict_round_trip():
    gen = GraphGenerator(seed=33)
    g = gen.generate(n=50, ring_size=8)
    train_ids, val_ids = train_val_split(g.labels, seed=33)
    model, _ = train_gat(g, train_ids, val_ids, hidden_dim=8, epochs=3,
                         patience=3, seed=33, n_heads=2)
    before = model.predict(g, val_ids)
    fresh = GraphAttention(g.feature_dim, hidden_dim=8, seed=99, n_heads=2)
    fresh.load_state_dict(model.state_dict())
    after = fresh.predict(g, val_ids)
    for v in val_ids:
        approx(before[v], after[v], tol=1e-12)


def test_legacy_checkpoint_loads_into_single_head_layer():
    # Old-format checkpoints (flat W/a_self/a_neigh/b) still load when
    # n_heads=1, so models saved before multi-head support keep working.
    legacy_state = {
        "W": [[1.0]],
        "a_self": [2.0],
        "a_neigh": [-1.0],
        "b": [0.0],
        "negative_slope": 0.2,
    }
    layer = GATLayer(1, 1, random.Random(0))
    layer.load_state_dict(legacy_state)
    out, cache = layer.forward(0, {0: [1.0], 1: [2.0]}, [1])
    a0 = math.exp(1.0) / (math.exp(1.0) + 1.0)
    a1 = 1.0 / (math.exp(1.0) + 1.0)
    approx(out[0], a0 * 1.0 + a1 * 2.0)

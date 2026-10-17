"""Tests for heterogeneous graphs: typed graph structure, the ring
generator's cross-type signals, relational message passing (hand-computed
+ numeric gradient check), and training dynamics."""

import math
import random

from gnn_fraud import (
    ACCOUNT,
    DEVICE,
    MERCHANT,
    HeteroGenerator,
    HeteroGraph,
    RelSAGELayer,
    RelationalGraphSAGE,
    roc_auc,
    train_hetero,
    train_mlp,
    train_val_split,
)
from gnn_fraud.hetero import (
    PAID_BY,
    PAYS,
    TRANSFERS,
    USED_BY,
    USES,
)


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def _dims():
    return {ACCOUNT: 2, DEVICE: 1, MERCHANT: 1}


def _tiny_hetero():
    """1 account (0), 1 device (1), 1 merchant (2); 0 uses 1, 0 pays 2."""
    g = HeteroGraph(_dims())
    a = g.add_node(ACCOUNT)
    d = g.add_node(DEVICE)
    m = g.add_node(MERCHANT)
    g.set_features(a, [1.0, 0.0])
    g.set_features(d, [2.0])
    g.set_features(m, [3.0])
    g.set_label(a, 1)
    g.add_uses(a, d)
    g.add_pays(a, m)
    return g


# ---------------------------------------------------------------------------
# graph data structure
# ---------------------------------------------------------------------------


def test_hetero_node_types_and_dims():
    g = _tiny_hetero()
    assert g.n == 3
    assert g.node_type == [ACCOUNT, DEVICE, MERCHANT]
    assert len(g.features[0]) == 2
    assert len(g.features[1]) == 1


def test_hetero_wrong_feature_dim_rejected():
    g = HeteroGraph(_dims())
    a = g.add_node(ACCOUNT)
    try:
        g.set_features(a, [1.0])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_hetero_relations_symmetric_pairs():
    g = _tiny_hetero()
    assert g.rel_neighbors(0, USES) == [1]
    assert g.rel_neighbors(1, USED_BY) == [0]
    assert g.rel_neighbors(0, PAYS) == [2]
    assert g.rel_neighbors(2, PAID_BY) == [0]
    # The wrong direction is empty.
    assert g.rel_neighbors(0, USED_BY) == []
    assert g.rel_neighbors(1, USES) == []


def test_hetero_labels_only_on_accounts():
    g = _tiny_hetero()
    assert g.labels[0] == 1
    assert g.labels[1] == -1
    assert g.labels[2] == -1
    try:
        g.set_label(1, 0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_hetero_transfers_symmetric():
    g = HeteroGraph(_dims())
    a0 = g.add_node(ACCOUNT)
    a1 = g.add_node(ACCOUNT)
    g.add_transfer(a0, a1)
    assert g.rel_neighbors(a0, TRANSFERS) == [a1]
    assert g.rel_neighbors(a1, TRANSFERS) == [a0]
    assert g.edge_count(TRANSFERS) == 2  # directed count


def test_hetero_type_counts():
    g = _tiny_hetero()
    assert g.type_counts() == {ACCOUNT: 1, DEVICE: 1, MERCHANT: 1}


# ---------------------------------------------------------------------------
# generator: the ring is planted and shares devices/merchants
# ---------------------------------------------------------------------------


def test_hetero_generator_plants_ring():
    gen = HeteroGenerator(seed=7)
    g = gen.generate(n_accounts=200, ring_size=18)
    ring = gen.ring_nodes(g)
    assert len(ring) == 18
    assert sum(g.labels[v] for v in g.account_ids()) == 18
    # Devices and merchants are unlabeled.
    assert all(g.labels[v] == -1 for v in range(g.n)
               if g.node_type[v] != ACCOUNT)


def test_hetero_generator_shared_device_signal():
    # The ring's structural tell: a shared device pool, vs benign
    # accounts that mostly use a private device.
    gen = HeteroGenerator(seed=7)
    g = gen.generate(n_accounts=200, ring_size=18)
    shared = gen.ring_devices(g)
    assert len(shared) >= 2, "ring should share at least 2 devices"
    ring_set = set(gen.ring_nodes(g))
    for d in shared:
        users = g.rel_neighbors(d, USED_BY)
        ring_users = [u for u in users if u in ring_set]
        assert len(ring_users) >= 3, (d, len(ring_users))


def test_hetero_generator_seedable():
    g1 = HeteroGenerator(seed=9).generate(n_accounts=100, ring_size=10)
    g2 = HeteroGenerator(seed=9).generate(n_accounts=100, ring_size=10)
    assert g1.labels == g2.labels
    assert g1.features == g2.features
    assert g1.node_type == g2.node_type
    for v in range(g1.n):
        for r in (USES, USED_BY, PAYS, PAID_BY, TRANSFERS):
            assert g1.rel_neighbors(v, r) == g2.rel_neighbors(v, r)


# ---------------------------------------------------------------------------
# relational layer: hand-computed forward
# ---------------------------------------------------------------------------


def test_relsage_layer_forward_hand_computed():
    # node type 0 (dim 1), one USES neighbor (device, dim 1).
    # W_self[0] = [[1]], W_rel[USES] = [[2]], b = [0.5].
    # pre = 1*2 + 2*mean([4]) + 0.5 = 10.5 -> relu keeps it.
    layer = RelSAGELayer({ACCOUNT: 1, DEVICE: 1, MERCHANT: 1}, 1,
                         random.Random(0))
    layer.W_self[ACCOUNT] = [[1.0]]
    layer.W_self[DEVICE] = [[1.0]]
    layer.W_self[MERCHANT] = [[1.0]]
    for r in (USES, USED_BY, PAYS, PAID_BY, TRANSFERS):
        layer.W_rel[r] = [[2.0]]
    layer.b = [0.5]
    h_by_rel = {r: ([[4.0]] if r == USES else []) for r in
                (USES, USED_BY, PAYS, PAID_BY, TRANSFERS)}
    out, cache = layer.forward(ACCOUNT, [2.0], h_by_rel)
    approx(out[0], 10.5)


def test_relsage_layer_forward_empty_relations():
    # No neighbors anywhere: only the self path + bias contribute.
    layer = RelSAGELayer({ACCOUNT: 1, DEVICE: 1, MERCHANT: 1}, 1,
                         random.Random(0))
    layer.W_self[ACCOUNT] = [[3.0]]
    layer.W_self[DEVICE] = [[1.0]]
    layer.W_self[MERCHANT] = [[1.0]]
    for r in (USES, USED_BY, PAYS, PAID_BY, TRANSFERS):
        layer.W_rel[r] = [[5.0]]
    layer.b = [-1.0]
    out, _ = layer.forward(ACCOUNT, [2.0], {})
    approx(out[0], 5.0)  # relu(3*2 - 1)


def test_relsage_layer_backward_hand_computed():
    # Same setup as the forward test: dL/dout = [1], pre > 0 so d_pre = [1].
    layer = RelSAGELayer({ACCOUNT: 1, DEVICE: 1, MERCHANT: 1}, 1,
                         random.Random(0))
    layer.W_self[ACCOUNT] = [[1.0]]
    layer.W_self[DEVICE] = [[1.0]]
    layer.W_self[MERCHANT] = [[1.0]]
    for r in (USES, USED_BY, PAYS, PAID_BY, TRANSFERS):
        layer.W_rel[r] = [[2.0]]
    layer.b = [0.5]
    h_by_rel = {r: ([[4.0]] if r == USES else []) for r in
                (USES, USED_BY, PAYS, PAID_BY, TRANSFERS)}
    _, cache = layer.forward(ACCOUNT, [2.0], h_by_rel)
    d_h_self, d_agg = layer.backward([1.0], cache)
    # gW_self[ACCOUNT] = d_pre * h_self = 1 * 2 = 2
    approx(layer.gW_self[ACCOUNT][0][0], 2.0)
    # gW_rel[USES] = d_pre * agg = 1 * 4 = 4; other relations got zero input.
    approx(layer.gW_rel[USES][0][0], 4.0)
    approx(layer.gW_rel[PAYS][0][0], 0.0)
    approx(layer.gb[0], 1.0)
    # d_h_self = W_self^T @ [1] = 1; d_agg[USES] = W_rel^T @ [1] = 2.
    approx(d_h_self[0], 1.0)
    approx(d_agg[USES][0], 2.0)
    approx(d_agg[PAYS][0], 2.0)  # W_rel[PAYS] is [[2]] regardless of input


def test_relsage_layer_numeric_gradient_check():
    # Central-difference check of gW_self, gW_rel, gb, d_h_self and the
    # per-relation d_agg against the analytic backward pass.
    rng = random.Random(5)
    in_dims = {ACCOUNT: 3, DEVICE: 2, MERCHANT: 2}
    layer = RelSAGELayer(in_dims, 2, rng)
    h_self = [rng.uniform(-1, 1) for _ in range(3)]
    h_by_rel = {
        USES: [[rng.uniform(-1, 1) for _ in range(2)] for _ in range(2)],
        USED_BY: [[rng.uniform(-1, 1) for _ in range(3)]],
        PAYS: [],
        PAID_BY: [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(2)],
        TRANSFERS: [[rng.uniform(-1, 1) for _ in range(3)]],
    }

    d_out = [0.7, -0.4]
    layer.zero_grad()
    out, cache = layer.forward(ACCOUNT, h_self, h_by_rel)
    d_h_self, d_agg = layer.backward(d_out, cache)
    analytic = {
        "W_self": {t: [row[:] for row in layer.gW_self[t]]
                   for t in in_dims},
        "W_rel": {r: [row[:] for row in layer.gW_rel[r]]
                  for r in (USES, USED_BY, PAYS, PAID_BY, TRANSFERS)},
        "b": layer.gb[:],
        "h_self": d_h_self[:],
        "agg": {r: d_agg[r][:] for r in
                (USES, USED_BY, PAYS, PAID_BY, TRANSFERS)},
    }

    def loss():
        o, _ = layer.forward(ACCOUNT, h_self, h_by_rel)
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

    def check_matrix(name, param, got, out_dim):
        in_dim = len(param[0])
        num = [num_grad(lambda i=i: param[i][:],
                        lambda j, v, i=i: param[i].__setitem__(j, v))
               for i in range(out_dim)]
        for i in range(out_dim):
            for j in range(in_dim):
                assert abs(got[i][j] - num[i][j]) < 1e-5, \
                    f"{name}[{i}][{j}]: {got[i][j]} vs {num[i][j]}"

    check_matrix("W_self[ACCOUNT]", layer.W_self[ACCOUNT],
                 analytic["W_self"][ACCOUNT], 2)
    check_matrix("W_self[DEVICE]", layer.W_self[DEVICE],
                 analytic["W_self"][DEVICE], 2)
    check_matrix("W_rel[USES]", layer.W_rel[USES],
                 analytic["W_rel"][USES], 2)
    check_matrix("W_rel[PAYS]", layer.W_rel[PAYS],
                 analytic["W_rel"][PAYS], 2)
    check_matrix("W_rel[TRANSFERS]", layer.W_rel[TRANSFERS],
                 analytic["W_rel"][TRANSFERS], 2)
    num_b = num_grad(lambda: layer.b[:],
                     lambda j, v: layer.b.__setitem__(j, v))
    for a, n in zip(analytic["b"], num_b):
        assert abs(a - n) < 1e-5, f"b: {a} vs {n}"

    # Gradients w.r.t. the center embedding and the per-relation means.
    num_h = num_grad(lambda: h_self[:],
                     lambda j, v: h_self.__setitem__(j, v))
    for a, n in zip(analytic["h_self"], num_h):
        assert abs(a - n) < 1e-5, f"h_self: {a} vs {n}"
    for r, vecs in h_by_rel.items():
        if not vecs:
            continue
        # Perturb the first neighbor; d(mean)/d(neighbor) = 1/m.
        m = len(vecs)
        num = num_grad(lambda: vecs[0][:],
                       lambda j, v: vecs[0].__setitem__(j, v))
        for j, n in enumerate(num):
            assert abs(analytic["agg"][r][j] / m - n) < 1e-5, \
                f"agg[{r}][{j}]: {analytic['agg'][r][j] / m} vs {n}"


# ---------------------------------------------------------------------------
# model-level behavior
# ---------------------------------------------------------------------------


def _small_hetero(seed=21):
    gen = HeteroGenerator(seed=seed)
    return gen.generate(n_accounts=60, ring_size=8, n_merchants=20)


def test_hetero_train_loss_decreases():
    g = _small_hetero()
    accounts = g.account_ids()
    labels = [g.labels[v] for v in accounts]
    tr, va = train_val_split(labels, val_frac=0.25, seed=21)
    train_ids = [accounts[i] for i in tr]
    val_ids = [accounts[i] for i in va]
    _, hist = train_hetero(g, train_ids, val_ids, hidden_dim=8,
                           k1=3, k2=3, lr=0.1, batch_size=32,
                           epochs=8, patience=8, seed=21)
    assert hist["train_loss"][-1] < hist["train_loss"][0]


def test_hetero_sampling_respects_per_relation_k():
    g = _small_hetero()
    model = RelationalGraphSAGE(g.type_dims, hidden_dim=4, seed=0)
    target = g.account_ids()[0]
    s2, _ = model._expand_batch(g, [target], k1=2, k2=2)
    for r in (USES, USED_BY, PAYS, PAID_BY, TRANSFERS):
        assert len(s2[target][r]) <= 2, (r, len(s2[target][r]))
    # Full neighborhood with k=None.
    s2f, _ = model._expand_batch(g, [target], k1=None, k2=None)
    for r in (USES, USED_BY, PAYS, PAID_BY, TRANSFERS):
        assert len(s2f[target][r]) == g.degree(target, r)


def test_hetero_predict_deterministic():
    g = _small_hetero()
    model, _ = train_hetero(g, hidden_dim=8, epochs=2, patience=2, seed=21)
    va = g.account_ids()
    p1 = model.predict(g, va)
    p2 = model.predict(g, va)
    assert p1 == p2
    assert all(0.0 <= p <= 1.0 for p in p1.values())


def test_hetero_state_dict_round_trip():
    g = _small_hetero()
    model, _ = train_hetero(g, hidden_dim=8, epochs=2, patience=2, seed=21)
    va = g.account_ids()
    before = model.predict(g, va)
    fresh = RelationalGraphSAGE(g.type_dims, hidden_dim=8, seed=99)
    fresh.load_state_dict(model.state_dict())
    after = fresh.predict(g, va)
    for v in va:
        approx(before[v], after[v], tol=1e-12)


def test_hetero_beats_feature_baseline():
    # The cross-type story, pinned by seed: a shared device or merchant is
    # a far stronger signal than camouflaged account features alone, so
    # the relational model must beat the feature-only MLP by a wide margin.
    gen = HeteroGenerator(seed=7)
    g = gen.generate(n_accounts=200, ring_size=18)
    accounts = g.account_ids()
    labels = [g.labels[v] for v in accounts]
    tr, va = train_val_split(labels, val_frac=0.25, seed=7)
    train_ids = [accounts[i] for i in tr]
    val_ids = [accounts[i] for i in va]

    hetero, _ = train_hetero(g, train_ids, val_ids, hidden_dim=16,
                             k1=5, k2=5, lr=0.1, batch_size=64,
                             epochs=40, patience=40, seed=7)
    mlp, _ = train_mlp([g.features[v] for v in accounts], labels, tr, va,
                       hidden_dim=16, lr=0.3, batch_size=32,
                       epochs=100, patience=100, seed=7)
    y_true = [g.labels[v] for v in val_ids]
    hetero_auc = roc_auc(y_true, [hetero.predict(g, [v])[v] for v in val_ids])
    mlp_auc = roc_auc(y_true, mlp.predict([g.features[v] for v in val_ids]))
    assert hetero_auc > 0.85, f"hetero auc {hetero_auc}"
    assert hetero_auc > mlp_auc + 0.20, f"hetero {hetero_auc} vs mlp {mlp_auc}"

"""Tests for the money-mule-chain fraud pattern (generate_mule_chain).

Money flows originator -> parallel mule chains -> consolidator -> exit:
long thin paths with almost no triangles, a structural counterpoint to
the dense pocket in generate(). Tests are pinned by seed where they
assert realized values.
"""

import pytest

from gnn_fraud import GraphGenerator, roc_auc, train_mlp, train_sage, train_val_split
from gnn_fraud.generator import FEATURE_NAMES


def _layout(g):
    return g.mule_chain


def test_mule_chain_topology():
    gen = GraphGenerator(seed=7)
    g = gen.generate_mule_chain(n=200, n_chains=4, chain_len=4)
    mc = _layout(g)
    assert len(mc["chains"]) == 4
    assert all(len(c) == 4 for c in mc["chains"])

    fraud = set(gen.ring_nodes(g))
    # fraud = originator + 16 mules + consolidator + exit = 19
    assert len(fraud) == 19
    expected = {mc["originator"], mc["consolidator"], mc["exit"]}
    for c in mc["chains"]:
        expected.update(c)
    assert fraud == expected

    # Wiring: originator -> chain heads -> ... -> consolidator -> exit.
    for c in mc["chains"]:
        assert mc["originator"] in g.adj[c[0]]
        for a, b in zip(c, c[1:]):
            assert b in g.adj[a]
        assert mc["consolidator"] in g.adj[c[-1]]
    assert mc["exit"] in g.adj[mc["consolidator"]]

    # Exact fraud-fraud edge count: per chain chain_len+1 edges,
    # plus the single consolidator->exit edge. Bridges only touch benign.
    ff = sum(1 for u in fraud for v in g.adj[u]
             if v in fraud and u < v)
    assert ff == 4 * (4 + 1) + 1


def test_mule_chain_sparse_not_dense():
    # The whole point: chains are a sparse pattern, not a dense pocket.
    gen = GraphGenerator(seed=7)
    g = gen.generate_mule_chain(n=200, n_chains=4, chain_len=4)
    fraud = gen.ring_nodes(g)
    assert g.density(fraud) < 0.25
    # No triangles at all among the fraud network on this seed.
    for a in fraud:
        for b in g.adj[a]:
            if b in set(fraud) and b > a:
                common = (set(g.adj[a]) & set(g.adj[b])) & set(fraud)
                assert not common


def test_mule_chain_timestamps_ordered():
    gen = GraphGenerator(seed=7)
    g = gen.generate_mule_chain(n=200, n_chains=4, chain_len=4,
                                timeline_days=30.0, burst_days=3.0)
    mc = _layout(g)
    for c in mc["chains"]:
        ts = [g.edge_time(mc["originator"], c[0])]
        ts += [g.edge_time(c[i], c[i + 1]) for i in range(len(c) - 1)]
        ts += [g.edge_time(c[-1], mc["consolidator"]),
               g.edge_time(mc["consolidator"], mc["exit"])]
        # Money moves downstream: monotone hops, all inside the late burst.
        assert all(b >= a for a, b in zip(ts, ts[1:])), ts
        assert all(t >= 27.0 for t in ts), ts
    # Benign edges spread over the whole timeline, not just the burst.
    benign = [u for u in range(g.n) if g.labels[u] == 0]
    times = [g.edge_time(u, v) for u in benign[:20] for v in g.adj[u]
             if v in set(benign) and g.edge_time(u, v) is not None]
    assert min(times) < 27.0


def test_mule_chain_deterministic():
    a = GraphGenerator(seed=7).generate_mule_chain()
    b = GraphGenerator(seed=7).generate_mule_chain()
    assert a.labels == b.labels
    assert a.mule_chain == b.mule_chain
    assert [sorted(a.adj[u]) for u in range(a.n)] == \
           [sorted(b.adj[u]) for u in range(b.n)]
    # A different seed plants a different network.
    c = GraphGenerator(seed=8).generate_mule_chain()
    assert c.mule_chain != a.mule_chain


def test_mule_chain_size_guard():
    with pytest.raises(ValueError):
        # 3 + 4*4 = 19 fraud nodes will not fit in n=18
        GraphGenerator(seed=7).generate_mule_chain(n=18, n_chains=4, chain_len=4)


def test_mule_chain_near_camouflage():
    # Same discipline as the dense ring: tells weak individually.
    gen = GraphGenerator(seed=42)
    g = gen.generate_mule_chain(n=300, n_chains=5, chain_len=5)
    fraud = gen.ring_nodes(g)
    benign = [u for u in range(g.n) if g.labels[u] == 0]

    def means(nodes):
        return [sum(g.features[u][j] for u in nodes) / len(nodes)
                for j in range(g.feature_dim)]

    mf, mb = means(fraud), means(benign)
    idx = {name: j for j, name in enumerate(FEATURE_NAMES)}
    for name in ["avg_txn_amount"]:
        assert abs(mf[idx[name]] - mb[idx[name]]) < 0.15, name
    for name in ["txn_count", "night_txn_ratio", "unique_counterparties"]:
        shift = mf[idx[name]] - mb[idx[name]]
        assert 0.0 <= shift < 0.20, (name, shift)


def test_gnn_beats_baseline_on_mule_chains():
    # Thin chains are a slow pattern to learn (mule neighborhoods mix
    # chain and benign neighbors), so the test budgets the long schedule
    # the example script uses. Pinned by seed.
    gen = GraphGenerator(seed=7)
    g = gen.generate_mule_chain(n=200, n_chains=4, chain_len=4)
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
    y_true = [g.labels[v] for v in val_ids]

    sage, _ = train_sage(g, train_ids, val_ids, hidden_dim=16,
                         k1=5, k2=5, lr=0.1, batch_size=64,
                         epochs=120, patience=120, seed=7)
    mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                       hidden_dim=16, lr=0.3, batch_size=32,
                       epochs=100, patience=100, seed=7)
    sage_auc = roc_auc(y_true, [sage.predict(g, [v])[v] for v in val_ids])
    mlp_auc = roc_auc(y_true, mlp.predict([g.features[v] for v in val_ids]))
    assert sage_auc > 0.85, f"sage auc {sage_auc}"
    # The margin over features is narrower than on the dense ring: the
    # organizer/consolidator/exit roles carry stronger individual tells
    # (orchestration volume, emptied cash-out balance) than camouflaged
    # ring members do. Message passing still adds clear value on top.
    assert sage_auc > mlp_auc + 0.10, f"sage {sage_auc} vs mlp {mlp_auc}"

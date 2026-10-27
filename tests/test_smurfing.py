"""Tests for the smurfing (structuring) fraud pattern (generate_smurfing).

Money flows source -> smurf intermediaries -> sink: a fan-out/fan-in
network, not a dense pocket and not a chain. The source and the sink
are hubs, every smurf hangs off exactly those two, and smurfs never
transact with each other -- so the network has no triangles at all.
Tests are pinned by seed where they assert realized values.
"""

import pytest

from gnn_fraud import GraphGenerator, roc_auc, train_mlp, train_sage, train_val_split
from gnn_fraud.generator import FEATURE_NAMES


def test_smurfing_topology():
    gen = GraphGenerator(seed=7)
    g = gen.generate_smurfing(n=200, n_smurfs=16)
    sm = g.smurfing
    assert len(sm["smurfs"]) == 16

    fraud = set(gen.ring_nodes(g))
    # fraud = source + 16 smurfs + sink = 18
    assert len(fraud) == 18
    assert fraud == {sm["source"], sm["sink"]} | set(sm["smurfs"])

    # Wiring: source -> every smurf -> sink, and nothing else inside
    # the network: no smurf-smurf edges, no direct source-sink edge.
    for s in sm["smurfs"]:
        assert s in g.adj[sm["source"]]
        assert sm["sink"] in g.adj[s]
    assert sm["sink"] not in g.adj[sm["source"]]
    for i, a in enumerate(sm["smurfs"]):
        for b in sm["smurfs"][i + 1:]:
            assert b not in g.adj[a]

    # Exact fraud-fraud edge count: source-smurf + smurf-sink.
    ff = sum(1 for u in fraud for v in g.adj[u]
             if v in fraud and u < v)
    assert ff == 2 * 16


def test_smurfing_fan_out_fan_in_not_dense():
    # The whole point: hubs + spokes, not a dense pocket and not a path.
    gen = GraphGenerator(seed=7)
    g = gen.generate_smurfing(n=200, n_smurfs=16)
    sm = g.smurfing
    fraud = gen.ring_nodes(g)
    # Induced density 32/153 ~ 0.21: far below the ring's ~0.7, far
    # above the chains' thin paths -- but density is the wrong lens
    # here; the signature is the degree split.
    assert g.density(fraud) < 0.30
    # Hubs: source and sink touch every smurf; smurfs touch only the
    # two hubs (plus any bridge edges to benign accounts).
    fraud_neighbors = lambda u: [v for v in g.adj[u] if v in set(fraud)]
    assert len(fraud_neighbors(sm["source"])) == 16
    assert len(fraud_neighbors(sm["sink"])) == 16
    assert all(len(fraud_neighbors(s)) == 2 for s in sm["smurfs"])
    # No triangles at all among the fraud network on this seed: a
    # smurf pair shares both hubs but the pair itself is unlinked.
    for a in fraud:
        for b in g.adj[a]:
            if b in set(fraud) and b > a:
                common = (set(g.adj[a]) & set(g.adj[b])) & set(fraud)
                assert not common


def test_smurfing_timestamps_ordered():
    gen = GraphGenerator(seed=7)
    g = gen.generate_smurfing(n=200, n_smurfs=16,
                              timeline_days=30.0, burst_days=3.0)
    sm = g.smurfing
    for s in sm["smurfs"]:
        t_in = g.edge_time(sm["source"], s)
        t_out = g.edge_time(s, sm["sink"])
        # Money arrives before it leaves, both inside the late burst.
        assert 27.0 <= t_in <= t_out <= 30.0, (t_in, t_out)
    # Benign edges spread over the whole timeline, not just the burst.
    benign = [u for u in range(g.n) if g.labels[u] == 0]
    times = [g.edge_time(u, v) for u in benign[:20] for v in g.adj[u]
             if v in set(benign) and g.edge_time(u, v) is not None]
    assert min(times) < 27.0


def test_smurfing_deterministic():
    a = GraphGenerator(seed=7).generate_smurfing()
    b = GraphGenerator(seed=7).generate_smurfing()
    assert a.labels == b.labels
    assert a.smurfing == b.smurfing
    assert [sorted(a.adj[u]) for u in range(a.n)] == \
           [sorted(b.adj[u]) for u in range(b.n)]
    # A different seed plants a different network.
    c = GraphGenerator(seed=8).generate_smurfing()
    assert c.smurfing != a.smurfing


def test_smurfing_size_guard():
    with pytest.raises(ValueError):
        # 16 + 2 = 18 fraud nodes will not fit in n=17
        GraphGenerator(seed=7).generate_smurfing(n=17, n_smurfs=16)


def test_smurfing_near_camouflage():
    # Same discipline as the dense ring and the mule chains: tells weak
    # individually. Smurfing's distinctive tell is on the amount side:
    # structured amounts run *below* the benign mean, not above it.
    gen = GraphGenerator(seed=42)
    g = gen.generate_smurfing(n=300, n_smurfs=20)
    fraud = gen.ring_nodes(g)
    benign = [u for u in range(g.n) if g.labels[u] == 0]

    def means(nodes):
        return [sum(g.features[u][j] for u in nodes) / len(nodes)
                for j in range(g.feature_dim)]

    mf, mb = means(fraud), means(benign)
    idx = {name: j for j, name in enumerate(FEATURE_NAMES)}
    shift = mf[idx["avg_txn_amount"]] - mb[idx["avg_txn_amount"]]
    assert -0.15 < shift < 0.0, shift
    for name in ["txn_count", "night_txn_ratio"]:
        shift = mf[idx[name]] - mb[idx[name]]
        assert 0.0 <= shift < 0.20, (name, shift)


def test_gnn_beats_baseline_on_smurfing():
    # Fan-out/fan-in is a slow pattern to learn (a smurf's 2-hop
    # neighborhood mixes the two hubs with benign bridge neighbors),
    # so the test budgets the long schedule the example script uses.
    # Pinned by seed.
    gen = GraphGenerator(seed=7)
    g = gen.generate_smurfing(n=200, n_smurfs=16)
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
    y_true = [g.labels[v] for v in val_ids]

    sage, _ = train_sage(g, train_ids, val_ids, hidden_dim=16,
                         k1=5, k2=5, lr=0.1, batch_size=64,
                         epochs=120, patience=20, seed=7)
    mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids,
                       hidden_dim=16, lr=0.3, batch_size=32,
                       epochs=150, patience=20, seed=7)
    sage_auc = roc_auc(y_true, [sage.predict(g, [v])[v] for v in val_ids])
    mlp_auc = roc_auc(y_true, mlp.predict([g.features[v] for v in val_ids]))
    assert sage_auc > 0.85, f"sage auc {sage_auc}"
    # The graph signal's premium is real here: the hubs' degree tell
    # is invisible to a feature-only model, which sees only the smurfs'
    # weak per-account tells.
    assert sage_auc > mlp_auc + 0.10, f"sage {sage_auc} vs mlp {mlp_auc}"

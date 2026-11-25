"""Tests for LightGCN: hand-computed no-self-loop propagation
powers and their average, the isolated-node dilution, numeric
gradient checks end to end, feature independence (the model is
structure-only), checkpointing, bad-parameter rejection, and
training dynamics on the dense ring."""

import math

from gnn_fraud import (
    Graph,
    GraphGenerator,
    LightGCN,
    averaged_embeddings,
    embedding_powers,
    evaluate_all,
    train_lightgcn,
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


# ---------------------------------------------------------------------------
# hand-computed powers and average
# ---------------------------------------------------------------------------


def test_lightgcn_powers_two_node_hand_computed():
    g = _two_node_graph()
    # Degrees are 1 and 1, so S (no self-loops) swaps the rows:
    # the powers alternate between E and its swap.
    powers = embedding_powers(g, [[1.0], [3.0]], 2)
    approx(powers[0][0][0], 1.0)
    approx(powers[1][0][0], 3.0)
    approx(powers[1][1][0], 1.0)
    approx(powers[2][0][0], 1.0)
    approx(powers[2][1][0], 3.0)


def test_lightgcn_average_two_node_hand_computed():
    g = _two_node_graph()
    # Mean of the powers above: node 0 gets (1 + 3 + 1) / 3,
    # node 1 gets (3 + 1 + 3) / 3.
    avg = averaged_embeddings(g, [[1.0], [3.0]], 2)
    approx(avg[0][0], 5.0 / 3.0)
    approx(avg[1][0], 7.0 / 3.0)


def test_lightgcn_isolated_node_diluted_to_own_embedding():
    g = Graph(3, 1)
    g.add_edge(0, 1)
    # Node 2 is isolated: with no self-loop its propagated state
    # is zero at every power k >= 1, so the average is exactly
    # its own embedding at weight 1 / (K + 1) - where the
    # self-looped operators keep it whole.
    avg = averaged_embeddings(g, [[1.0], [3.0], [7.0]], 4)
    approx(avg[2][0], 7.0 / 5.0)


def test_lightgcn_K_zero_returns_copy():
    g = _two_node_graph()
    e = [[1.0], [3.0]]
    avg = averaged_embeddings(g, e, 0)
    assert avg == e
    avg[0][0] = 99.0
    assert e[0][0] == 1.0


def test_lightgcn_K_zero_model_ignores_graph():
    # K = 0: Z = E exactly, so edges change nothing.
    g = _two_node_graph()
    g2 = Graph(2, 1)
    g2.set_features(0, [1.0])
    g2.set_features(1, [3.0])
    model = LightGCN(2, emb_dim=4, K=0, seed=3)
    p1 = model.predict(g, [0, 1])
    p2 = model.predict(g2, [0, 1])
    approx(p1[0], p2[0])
    approx(p1[1], p2[1])


def test_lightgcn_model_ignores_features():
    # The structure-only claim: features are never read, so
    # rewriting every feature vector changes no prediction.
    g = _two_node_graph()
    g2 = _two_node_graph()
    g2.set_features(0, [-100.0])
    g2.set_features(1, [55.0])
    model = LightGCN(2, emb_dim=4, K=3, seed=3)
    p1 = model.predict(g, [0, 1])
    p2 = model.predict(g2, [0, 1])
    approx(p1[0], p2[0])
    approx(p1[1], p2[1])


def test_lightgcn_bad_params_rejected():
    for kwargs in ({"n_nodes": 0}, {"emb_dim": 0}, {"K": -1}):
        args = {"n_nodes": 4}
        args.update(kwargs)
        try:
            LightGCN(**args)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")
    for fn in (lambda: averaged_embeddings(_two_node_graph(),
                                           [[1.0], [2.0]], -1),
               lambda: embedding_powers(_two_node_graph(),
                                        [[1.0]], 1)):
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")
    model = LightGCN(2, emb_dim=4, seed=0)
    try:
        model.predict(Graph(3, 1), [0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for wrong graph size")


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


def test_lightgcn_model_gradient_check():
    # Perturb every parameter (embeddings and head; the
    # propagation has none) and compare with the gradients
    # train_batch accumulates at lr=0.0. The embedding check is
    # the one that matters: its gradient must run the same layer
    # average backwards, and a missing 1 / (K + 1) or a
    # last-power-only backward fails it. Embeddings of nodes
    # with no target within K hops get exactly zero gradient,
    # which the check covers too (both sides are zero).
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    labels = {0: 1, 1: 0, 2: 1, 3: 0}
    model = LightGCN(4, emb_dim=3, K=3, lr=0.0, seed=5)
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

    for v in range(4):
        for j in range(3):
            check(model.gE[v][j],
                  lambda v=v, j=j: model.embeddings[v][j],
                  lambda val, v=v, j=j: model.embeddings[v].__setitem__(j, val))
    for j in range(model.emb_dim):
        check(model.gw_out[j],
              lambda j=j: model.w_out[j],
              lambda v, j=j: model.w_out.__setitem__(j, v))
    check(model.gb_out, lambda: model.b_out,
          lambda v: setattr(model, "b_out", v))


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------


def test_lightgcn_checkpoint_round_trip():
    model = LightGCN(6, emb_dim=8, K=3, seed=1)
    state = model.state_dict()
    model2 = LightGCN(6, emb_dim=8, K=3, seed=99)
    model2.load_state_dict(state)
    g = Graph(6, 6)
    g.add_edge(0, 1)
    p1 = model.predict(g, [0, 1])
    p2 = model2.predict(g, [0, 1])
    approx(p1[0], p2[0])
    approx(p1[1], p2[1])


# ---------------------------------------------------------------------------
# training dynamics
# ---------------------------------------------------------------------------


def test_lightgcn_trains_to_high_auc_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    model, _ = train_lightgcn(g, train_ids, val_ids, emb_dim=16,
                              K=3, lr=0.5, batch_size=64, epochs=100,
                              patience=25, seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    y_true = [g.labels[v] for v in val_ids]
    assert evaluate_all(y_true, scores)["auc"] >= 0.90


def test_lightgcn_propagation_beats_no_propagation_on_dense_ring():
    # The mechanism claim: at K = 0 a held-out node's embedding
    # is never trained (no labeled node is within reach), so it
    # ranks far below the propagated model, while K = 3 lets
    # labeled neighbors train it through the propagation. Pin
    # the ordering, not the exact numbers.
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    y_true = [g.labels[v] for v in val_ids]
    aucs = {}
    for K in (0, 3):
        model, _ = train_lightgcn(g, train_ids, val_ids, emb_dim=16,
                                  K=K, lr=0.5, batch_size=64, epochs=100,
                                  patience=25, seed=7)
        scores = [model.predict(g, [v])[v] for v in val_ids]
        aucs[K] = evaluate_all(y_true, scores)["auc"]
    assert aucs[3] > aucs[0] + 0.2

"""Tests for LINKX: hand-computed adjacency rows, the isolated
node's one-hot row, the two branches' separation (features-only
and structure-only ablations by construction), numeric gradient
checks end to end over both branches, checkpointing, bad
parameter and wrong-size rejection, and training dynamics on
the dense ring."""

import math

from gnn_fraud import (
    LINKX,
    Graph,
    GraphGenerator,
    adjacency_row,
    adjacency_rows,
    evaluate_all,
    train_linkx,
    train_val_split,
)


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def _path_graph():
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    g.set_features(3, [0.1, 0.4])
    return g


# ---------------------------------------------------------------------------
# hand-computed adjacency rows
# ---------------------------------------------------------------------------


def test_linkx_adjacency_row_hand_computed():
    g = _path_graph()
    # Node 1 has degree 2, so its row is itself and both
    # neighbours at 1/3 each, self first.
    row = adjacency_row(g, 1)
    assert [u for u, _ in row] == [1, 0, 2]
    for _, c in row:
        approx(c, 1.0 / 3.0)
    approx(sum(c for _, c in row), 1.0)
    # An endpoint has degree 1: self and its one neighbour at 1/2.
    row0 = adjacency_row(g, 0)
    assert [u for u, _ in row0] == [0, 1]
    approx(row0[0][1], 0.5)


def test_linkx_adjacency_row_isolated_is_one_hot():
    g = Graph(3, 1)
    g.add_edge(0, 1)
    row = adjacency_row(g, 2)
    assert row == [(2, 1.0)]


def test_linkx_adjacency_rows_all_sum_to_one():
    g = _path_graph()
    rows = adjacency_rows(g)
    assert len(rows) == 4
    for row in rows:
        approx(sum(c for _, c in row), 1.0)


def test_linkx_adjacency_row_bad_node_rejected():
    try:
        adjacency_row(_path_graph(), 9)
    except IndexError:
        return
    raise AssertionError("expected IndexError")


# ---------------------------------------------------------------------------
# branch separation
# ---------------------------------------------------------------------------


def test_linkx_structure_branch_ignores_features():
    # Rewriting every feature changes only the feature branch;
    # with W_feat zeroed the predictions must not move at all.
    g = _path_graph()
    model = LINKX(2, 4, hidden_dim=3, seed=3)
    model.W_feat = [[0.0, 0.0] for _ in range(3)]
    p1 = model.predict(g, [0, 1, 2, 3])
    g2 = _path_graph()
    for v in range(4):
        g2.set_features(v, [9.0, -9.0])
    p2 = model.predict(g2, [0, 1, 2, 3])
    for v in range(4):
        approx(p1[v], p2[v])


def test_linkx_feature_branch_ignores_wiring():
    # With W_adj zeroed, adding an edge changes no prediction:
    # the feature branch never sees the graph.
    g = _path_graph()
    model = LINKX(2, 4, hidden_dim=3, seed=3)
    model.W_adj = [[0.0] * 4 for _ in range(3)]
    p1 = model.predict(g, [0, 1, 2, 3])
    g.add_edge(0, 3)
    p2 = model.predict(g, [0, 1, 2, 3])
    for v in range(4):
        approx(p1[v], p2[v])


def test_linkx_wrong_graph_size_rejected():
    model = LINKX(2, 4, hidden_dim=3, seed=3)
    try:
        model.predict(Graph(5, 2), [0])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_linkx_bad_params_rejected():
    for kwargs in ({"in_dim": 0}, {"n_nodes": 0}, {"hidden_dim": 0}):
        args = {"in_dim": 6, "n_nodes": 10}
        args.update(kwargs)
        try:
            LINKX(**args)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


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


def test_linkx_model_gradient_check():
    # Perturb every parameter in both branches and the head.
    # The structure branch is the one that matters: its gradient
    # must land on exactly the columns each row names, scaled by
    # the row weight, and a dense or unscaled version fails here.
    g = _path_graph()
    labels = {0: 1, 1: 0, 2: 1, 3: 0}
    model = LINKX(2, 4, hidden_dim=3, lr=0.0, seed=5)
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

    for i in range(model.hidden_dim):
        for j in range(model.in_dim):
            check(model.gW_feat[i][j],
                  lambda i=i, j=j: model.W_feat[i][j],
                  lambda v, i=i, j=j: model.W_feat[i].__setitem__(j, v))
        for u in range(model.n_nodes):
            check(model.gW_adj[i][u],
                  lambda i=i, u=u: model.W_adj[i][u],
                  lambda v, i=i, u=u: model.W_adj[i].__setitem__(u, v))
        check(model.gb_feat[i],
              lambda i=i: model.b_feat[i],
              lambda v, i=i: model.b_feat.__setitem__(i, v))
        check(model.gb_adj[i],
              lambda i=i: model.b_adj[i],
              lambda v, i=i: model.b_adj.__setitem__(i, v))
    for j in range(2 * model.hidden_dim):
        check(model.gw_out[j],
              lambda j=j: model.w_out[j],
              lambda v, j=j: model.w_out.__setitem__(j, v))
    check(model.gb_out, lambda: model.b_out,
          lambda v: setattr(model, "b_out", v))


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------


def test_linkx_checkpoint_round_trip():
    model = LINKX(6, 10, hidden_dim=8, seed=1)
    state = model.state_dict()
    model2 = LINKX(6, 10, hidden_dim=8, seed=99)
    model2.load_state_dict(state)
    g = Graph(10, 6)
    g.add_edge(0, 1)
    for v in range(10):
        g.set_features(v, [0.1 * (v + 1), 0.2, 0.3, 0.4, 0.5, 0.6])
    p1 = model.predict(g, [0, 1])
    p2 = model2.predict(g, [0, 1])
    approx(p1[0], p2[0])
    approx(p1[1], p2[1])


def test_linkx_branch_norms_split_head():
    model = LINKX(6, 10, hidden_dim=4, seed=1)
    feat, adj = model.branch_norms()
    approx(feat, math.sqrt(sum(w * w for w in model.w_out[:4])))
    approx(adj, math.sqrt(sum(w * w for w in model.w_out[4:])))


# ---------------------------------------------------------------------------
# training dynamics
# ---------------------------------------------------------------------------


def test_linkx_trains_to_high_auc_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    model, _ = train_linkx(g, train_ids, val_ids, hidden_dim=16,
                           lr=0.5, batch_size=64, epochs=100,
                           patience=25, seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    y_true = [g.labels[v] for v in val_ids]
    assert evaluate_all(y_true, scores)["auc"] >= 0.90

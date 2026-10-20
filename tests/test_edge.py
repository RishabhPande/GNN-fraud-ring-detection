"""Tests for edge classification: edge labels and features, the edge head's
hand-computed forward pass and numeric gradient check, the edge train/val
split, training dynamics, and the headline claim that a head over frozen
GNN embeddings separates ring-internal flows from background edges."""

import math
import random

from gnn_fraud import (
    EdgeHead,
    Graph,
    GraphGenerator,
    edge_labels,
    edge_phi,
    edge_train_val_split,
    full_embeddings,
    roc_auc,
    score_edges,
    train_edge_cls,
    train_sage,
    train_val_split,
)
from gnn_fraud.edge_cls import _bce, _sigmoid


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


# ---------------------------------------------------------------------------
# edge features and labels
# ---------------------------------------------------------------------------


def test_edge_phi_hand_computed():
    h_u, h_v = [1.0, -2.0], [0.5, 3.0]
    phi = edge_phi(h_u, h_v)
    assert len(phi) == 8  # 4 * embedding_dim
    assert phi == [1.0, -2.0, 0.5, 3.0, 0.5, -6.0, 0.5, 5.0]


def test_edge_labels_ring_bridge_benign():
    # 4 nodes: 0,1 ring (edge), 2,3 benign (edge), bridge 1-2.
    g = Graph(4, 1)
    g.set_label(0, 1)
    g.set_label(1, 1)
    g.add_edge(0, 1)
    g.add_edge(2, 3)
    g.add_edge(1, 2)
    labels = edge_labels(g)
    assert labels == {(0, 1): 1, (1, 2): 0, (2, 3): 0}
    # undirected key canonicalization: add (3,2) order doesn't matter
    assert len(labels) == 3


def test_edge_train_val_split_stratified():
    edges = [(i, i + 1) for i in range(100)]
    labels = {e: (1 if i < 30 else 0) for i, e in enumerate(edges)}
    tr, va = edge_train_val_split(edges, labels, val_frac=0.2, seed=3)
    assert len(tr) + len(va) == 100
    assert set(tr).isdisjoint(set(va))
    n_pos_va = sum(labels[e] for e in va)
    n_neg_va = len(va) - n_pos_va
    assert n_pos_va == 6 and n_neg_va == 14  # stratified 20% of each class
    # deterministic
    assert edge_train_val_split(edges, labels, val_frac=0.2, seed=3) == (tr, va)


# ---------------------------------------------------------------------------
# the edge head: hand-computed forward + numeric gradient check
# ---------------------------------------------------------------------------


def _fixed_head():
    # 2-dim embeddings -> phi dim 8; hidden dim 2. Fixed weights so the
    # forward pass is hand-computable.
    head = EdgeHead(8, hidden_dim=2, lr=0.1, seed=0)
    head.W1 = [[0.5, -0.5, 0.25, 0.0, 1.0, 0.0, 0.0, 0.5],
               [0.0, 0.25, -0.5, 0.5, 0.0, 1.0, -0.25, 0.0]]
    head.b1 = [0.1, -0.1]
    head.w2 = [1.0, -0.5]
    head.b2 = 0.2
    return head


def test_edge_head_forward_hand_computed():
    head = _fixed_head()
    phi = [1.0, 0.5, 0.0, 1.0, 0.5, -1.0, 0.25, 0.0]
    p, cache = head.forward(phi)
    # pre1[0] = 0.5*1 + (-0.5)*0.5 + 0.25*0 + 0*1 + 1*0.5 + 0*(-1)
    #           + 0*0.25 + 0.5*0 + 0.1 = 0.85
    # pre1[1] = 0*1 + 0.25*0.5 + (-0.5)*0 + 0.5*1 + 0*0.5 + 1*(-1)
    #           + (-0.25)*0.25 + 0*0 - 0.1 = -0.5375 -> relu 0
    approx(cache[2][0], 0.85)
    approx(cache[2][1], 0.0)
    approx(p, _sigmoid(1.0 * 0.85 + 0.2))
    # probability is monotone in the positive unit's activation
    p2, _ = head.forward([2.0, 0.5, 0.0, 1.0, 0.5, -1.0, 0.25, 0.0])
    assert p2 > p


def test_edge_head_numeric_gradient():
    # All-positive weights/inputs/bias: relu never kinks, so the numeric
    # check sees a smooth function.
    head = EdgeHead(8, hidden_dim=3, lr=0.1, seed=1)
    for j in range(3):
        head.W1[j] = [0.1 * (j + k + 1) for k in range(8)]
        head.b1[j] = 0.1
    head.w2 = [0.2, 0.3, 0.4]
    head.b2 = 0.1
    phi = [0.5, 1.0, 0.25, 0.75, 0.1, 0.9, 0.4, 0.6]
    y = 1
    w, B = 2.0, 4.0  # pos_weight and batch size, as in the train loop

    def loss():
        p, _ = head.forward(phi)
        return w * _bce(p, y) / B

    base = loss()
    eps = 1e-6

    def numeric(get, set_):
        old = get()
        set_(old + eps)
        up = loss()
        set_(old - eps)
        dn = loss()
        set_(old)
        return (up - dn) / (2 * eps)

    # analytic grads for d_z = w*(p - y)/B
    p, cache = head.forward(phi)
    head.zero_grad()
    head.backward(w * (p - y) / B, cache)
    assert abs(base - loss()) < 1e-12  # loss() is pure

    approx(head.gw2[1], numeric(lambda: head.w2[1],
                                lambda x: head.w2.__setitem__(1, x)),
           tol=1e-5)
    approx(head.gb2, numeric(lambda: head.b2,
                             lambda x: setattr(head, "b2", x)),
           tol=1e-5)
    approx(head.gW1[2][5], numeric(lambda: head.W1[2][5],
                                   lambda x: head.W1[2].__setitem__(5, x)),
           tol=1e-5)
    approx(head.gb1[0], numeric(lambda: head.b1[0],
                                lambda x: head.b1.__setitem__(0, x)),
           tol=1e-5)


def test_edge_head_step_and_checkpoint():
    head = _fixed_head()
    phi = [1.0, 0.5, 0.0, 1.0, 0.5, -1.0, 0.25, 0.0]
    p, cache = head.forward(phi)
    head.zero_grad()
    head.backward((p - 1), cache)  # push toward y=1
    before = (list(head.w2), head.b2, [r[:] for r in head.W1])
    head.step(0.1)
    # the output layer moved along the negative gradient
    for j in range(head.hidden_dim):
        approx(head.w2[j], before[0][j] - 0.1 * head.gw2[j])
    approx(head.b2, before[1] - 0.1 * head.gb2)
    state = head.state_dict()
    head2 = EdgeHead(8, hidden_dim=2, seed=9)
    head2.load_state_dict(state)
    assert head2.W1 == head.W1 and head2.w2 == head.w2
    assert head2.b1 == head.b1 and head2.b2 == head.b2
    # the restored checkpoint scores exactly like the stepped head
    p_after, _ = head.forward(phi)
    p3, _ = head2.forward(phi)
    approx(p3, p_after)


def test_edge_head_sigmoid_bce_derivative_cancel():
    # Sanity on the algebra the training loop relies on:
    # d/dp [BCE(p, y)] * p(1-p) == p - y exactly.
    for p, y in [(0.2, 1), (0.8, 0), (0.99, 1), (0.01, 0)]:
        p = min(max(p, 1e-7), 1.0 - 1e-7)
        d_bce_dp = (p - y) / (p * (1 - p))
        approx(d_bce_dp * p * (1 - p), p - y)


# ---------------------------------------------------------------------------
# training dynamics + headline claim
# ---------------------------------------------------------------------------


def _small_graph(seed=123):
    return GraphGenerator(seed=seed).generate(n=60, ring_size=8)


def test_train_edge_head_loss_decreases():
    g = _small_graph()
    tr, va = train_val_split(g.labels, val_frac=0.25, seed=1)
    enc, _ = train_sage(g, tr, va, hidden_dim=8, epochs=6, patience=6,
                        seed=1)
    _, _, hist = train_edge_cls(g, enc, hidden_dim=8, epochs=8,
                                patience=8, seed=1)
    assert hist["train_loss"][-1] < hist["train_loss"][0]
    assert hist["val_loss"][-1] <= hist["val_loss"][0]


def test_full_embeddings_deterministic():
    g = _small_graph()
    tr, va = train_val_split(g.labels, val_frac=0.25, seed=1)
    enc, _ = train_sage(g, tr, va, hidden_dim=8, epochs=2, patience=2,
                        seed=1)
    nodes = list(range(g.n))
    e1 = full_embeddings(enc, g, nodes)
    e2 = full_embeddings(enc, g, nodes)
    assert e1 == e2  # full neighborhood: no sampling noise
    assert set(e1) == set(nodes)
    assert all(len(v) == 8 for v in e1.values())


def test_score_edges_deterministic():
    g = _small_graph()
    tr, va = train_val_split(g.labels, val_frac=0.25, seed=1)
    enc, _ = train_sage(g, tr, va, hidden_dim=8, epochs=3, patience=3,
                        seed=1)
    head, emb, _ = train_edge_cls(g, enc, hidden_dim=8, epochs=4,
                                  patience=4, seed=1)
    edges = sorted(edge_labels(g))[:20]
    assert score_edges(head, emb, edges) == score_edges(head, emb, edges)


def test_edge_headline_separates_ring_flows():
    # Same pinned protocol as the other headline tests: the edge head over
    # frozen GraphSAGE embeddings must separate ring-internal edges from
    # background edges on held-out edges.
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    tr, va = train_val_split(g.labels, val_frac=0.25, seed=7)
    enc, _ = train_sage(g, tr, va, hidden_dim=16, k1=5, k2=5, lr=0.1,
                        batch_size=64, epochs=60, patience=12, seed=7)
    head, emb, _ = train_edge_cls(g, enc, hidden_dim=16, lr=0.1,
                                  batch_size=64, epochs=60, patience=12,
                                  seed=7)
    labels = edge_labels(g)
    _, val_edges = edge_train_val_split(sorted(labels), labels, seed=7)
    scores = score_edges(head, emb, val_edges)
    y = [labels[e] for e in val_edges]
    s = [scores[e] for e in val_edges]
    auc = roc_auc(y, s)
    assert auc > 0.85, f"edge auc {auc}"
    # ring-internal edges score above background on average
    pos = [scores[e] for e in val_edges if labels[e] == 1]
    neg = [scores[e] for e in val_edges if labels[e] == 0]
    assert sum(pos) / len(pos) > sum(neg) / len(neg)

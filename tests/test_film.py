"""Tests for the FiLM-GNN: hand-computed layer forward, the zero-FiLM
GraphSAGE-mean equivalence, the isolated-node case, modulation
introspection, numeric gradient checks, checkpointing, and training
dynamics on the dense ring."""

import math
import random

from gnn_fraud import (
    FiLMLayer,
    Graph,
    GraphFiLM,
    GraphGenerator,
    evaluate_all,
    train_film,
    train_val_split,
)


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def _hand_layer():
    """1-dim in, 1-dim out layer with every weight set by hand."""
    layer = FiLMLayer(1, 1, random.Random(0))
    layer.W_self = [[1.0]]
    layer.W_neigh = [[2.0]]
    layer.W_gamma = [[0.5]]
    layer.b_gamma = [0.0]
    layer.W_beta = [[-1.0]]
    layer.b_beta = [0.25]
    layer.b = [0.1]
    return layer


def _two_node_graph():
    g = Graph(2, 1)
    g.add_edge(0, 1)
    g.set_features(0, [1.0])
    g.set_features(1, [3.0])
    return g


# ---------------------------------------------------------------------------
# hand-computed forward
# ---------------------------------------------------------------------------


def test_film_layer_hand_computed():
    g = _two_node_graph()
    layer = _hand_layer()
    out, _ = layer.forward(g, g.features)
    # Node 0: scale 1 + 0.5*1 = 1.5, shift -1 + 0.25 = -0.75,
    # message mean = 2*3 = 6, agg = 1.5*6 - 0.75 = 8.25,
    # pre = 1*1 + 8.25 + 0.1 = 9.35.
    approx(out[0][0], 9.35)
    # Node 1: scale 1 + 0.5*3 = 2.5, shift -3 + 0.25 = -2.75,
    # message mean = 2*1 = 2, agg = 2.5*2 - 2.75 = 2.25,
    # pre = 3 + 2.25 + 0.1 = 5.35.
    approx(out[1][0], 5.35)


def test_film_modulation_hand_computed():
    layer = _hand_layer()
    scale, shift = layer.modulation([1.0])
    approx(scale[0], 1.5)
    approx(shift[0], -0.75)
    scale, shift = layer.modulation([3.0])
    approx(scale[0], 2.5)
    approx(shift[0], -2.75)


def test_film_zero_branch_is_unmodulated():
    # With the FiLM branch zeroed, the scale is exactly 1 and the
    # shift exactly 0 for every input, so the layer reduces to the
    # GraphSAGE-mean combination self + mean neighbor message.
    layer = _hand_layer()
    layer.W_gamma = [[0.0]]
    layer.W_beta = [[0.0]]
    layer.b_gamma = [0.0]
    layer.b_beta = [0.0]
    for x in ([0.0], [1.0], [-4.5], [100.0]):
        scale, shift = layer.modulation(x)
        approx(scale[0], 1.0)
        approx(shift[0], 0.0)
    g = _two_node_graph()
    out, _ = layer.forward(g, g.features)
    # Node 0: self 1 + mean message 6 + bias 0.1 = 7.1 (shift is 0,
    # but the layer bias remains).
    approx(out[0][0], 7.1)
    # Node 1: self 3 + mean message 2 + 0.1 = 5.1.
    approx(out[1][0], 5.1)


def test_film_isolated_node_gets_no_phantom_message():
    # An isolated node has no neighbors, so its output is the self
    # transform plus bias only, however large the shift its profile
    # generates: the FiLM branch modulates a message, it does not
    # invent one.
    g = Graph(3, 1)
    g.add_edge(0, 1)
    g.set_features(0, [1.0])
    g.set_features(1, [3.0])
    g.set_features(2, [2.0])
    layer = _hand_layer()
    out, _ = layer.forward(g, g.features)
    # Node 2: relu(1*2 + 0.1) = 2.1, not 2 + (2*2 - 1.75) + 0.1.
    approx(out[2][0], 2.1)


def test_film_bad_params_rejected():
    for kwargs in ({"in_dim": 0, "out_dim": 2}, {"in_dim": 2, "out_dim": 0}):
        try:
            FiLMLayer(rng=random.Random(0), **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")
    try:
        GraphFiLM(6, hidden_dim=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for hidden_dim=0")


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


def test_film_model_gradient_check():
    # Perturb every parameter of both layers (self, neighbor, gamma,
    # beta matrices and every bias) plus the head, and compare with
    # the gradients train_batch accumulates at lr=0.0. This is the
    # check that validates the bilinear split: scale takes message
    # times output gradient, message takes scale times output
    # gradient, and both FiLM matrices take outer products against
    # the node's own input.
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    g.set_features(3, [0.1, 0.4])
    labels = {0: 1, 1: 0, 2: 1, 3: 0}
    model = GraphFiLM(2, hidden_dim=4, lr=0.0, seed=5)
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
        for W, gW in (
            (layer.W_self, layer.gW_self),
            (layer.W_neigh, layer.gW_neigh),
            (layer.W_gamma, layer.gW_gamma),
            (layer.W_beta, layer.gW_beta),
        ):
            for i in range(layer.out_dim):
                for j in range(layer.in_dim):
                    check(gW[i][j],
                          lambda W=W, i=i, j=j: W[i][j],
                          lambda v, W=W, i=i, j=j: W[i].__setitem__(j, v))
        for b, gb in (
            (layer.b_gamma, layer.gb_gamma),
            (layer.b_beta, layer.gb_beta),
            (layer.b, layer.gb),
        ):
            for i in range(layer.out_dim):
                check(gb[i],
                      lambda b=b, i=i: b[i],
                      lambda v, b=b, i=i: b.__setitem__(i, v))
    for j in range(model.hidden_dim):
        check(model.gw_out[j],
              lambda j=j: model.w_out[j],
              lambda v, j=j: model.w_out.__setitem__(j, v))
    check(model.gb_out, lambda: model.b_out,
          lambda v: setattr(model, "b_out", v))


def test_film_layer_gradient_check_isolated_node():
    # Node 3 is isolated: its FiLM branch must carry no gradient at
    # all (no message, no modulation), while its self transform
    # still does. Checked through the full model so a phantom-message
    # bug in forward or backward both fail here.
    g = Graph(4, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.set_features(0, [0.5, -1.0])
    g.set_features(1, [1.5, 0.25])
    g.set_features(2, [-0.75, 2.0])
    g.set_features(3, [0.3, 0.9])
    labels = {0: 1, 1: 0, 2: 1, 3: 1}
    model = GraphFiLM(2, hidden_dim=3, lr=0.0, seed=3)
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

    layer = model.layer1
    for W, gW in ((layer.W_gamma, layer.gW_gamma),
                  (layer.W_beta, layer.gW_beta),
                  (layer.W_self, layer.gW_self)):
        for i in range(layer.out_dim):
            for j in range(layer.in_dim):
                check(gW[i][j],
                      lambda W=W, i=i, j=j: W[i][j],
                      lambda v, W=W, i=i, j=j: W[i].__setitem__(j, v))


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------


def test_film_checkpoint_round_trip():
    model = GraphFiLM(6, hidden_dim=8, seed=1)
    state = model.state_dict()
    model2 = GraphFiLM(6, hidden_dim=8, seed=99)
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


def test_film_trains_to_high_auc_on_dense_ring():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    model, _ = train_film(g, train_ids, val_ids, hidden_dim=16,
                          lr=0.1, batch_size=64, epochs=60,
                          patience=12, seed=7)
    scores = [model.predict(g, [v])[v] for v in val_ids]
    y_true = [g.labels[v] for v in val_ids]
    assert evaluate_all(y_true, scores)["auc"] >= 0.90


def test_film_modulation_moves_off_identity():
    # The conditioning is the finding, so pin that training actually
    # moves the scales off their near-1 start (a dead FiLM branch
    # would leave every scale at 1.0 and the model a plain SAGE).
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    train_ids, val_ids = train_val_split(g.labels, seed=7)
    model, _ = train_film(g, train_ids, val_ids, hidden_dim=16,
                          lr=0.1, batch_size=64, epochs=60,
                          patience=12, seed=7)
    scales = []
    for v in range(g.n):
        scale, _ = model.modulation(g, v)
        scales.extend(scale)
    spread = max(scales) - min(scales)
    assert spread > 0.1

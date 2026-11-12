"""GGNN: Gated Graph Neural Networks (Li et al., 2016), from scratch.

Every other model in this repo decides in advance how many hops of
aggregation happen and then stops: two SAGE/GCN layers read two hops,
the polynomial models read K hops of a fixed-shape filter. GGNN makes
propagation itself recurrent. The hidden state is rewritten ``steps``
times, and at every step a GRU decides, per node and per dimension,
how much of the incoming neighborhood message to take and how much of
the node's own history to keep::

    h_v^(0) = relu(W_in @ x_v + b_in)                (encoder)
    a_v^(t) = mean_{u in N(v)} h_u^(t-1)             (message, 0 if isolated)
    z = sigmoid(W_z a + U_z h^(t-1) + b_z)           (update gate)
    r = sigmoid(W_r a + U_r h^(t-1) + b_r)           (reset gate)
    n = tanh(W_h a + U_h (r * h^(t-1)) + b_h)        (candidate)
    h_v^(t) = (1 - z) * h_v^(t-1) + z * n            (gated update)
    logit   = w_out . h_v^(steps) + b_out

The gate parameters are shared across steps, so ``steps`` is a
receptive-field knob, not a parameter-count knob: a node whose
neighborhood turns noisy after one hop can learn to close its update
gate and keep its step-1 state, where a plain stacked GCN has no way
to refuse the later aggregations. That is the mechanism this module
tests on the dense ring. It is the same reason oversmoothing is less
automatic here than in a deep GCN, and the same reason GCNII needs
its hand-wired residual: the residual here is learned, per dimension.

Two simplifications against the paper, both named. First, the paper
propagates with a separate weight matrix per edge type; this graph
has one edge type, and the message is the plain mean of neighbor
states, so the adjacency weighting is the only propagation matrix.
Second, the paper pads raw features to the hidden width as the
initial state; here a learned ReLU encoder makes the initial state,
like the other models in this repo.

Backprop is full backprop through time over the shared GRU: each
step's gradient splits into the gate branches (whose Jacobians are
the usual sigmoid/tanh derivatives), the message branch (which routes
``d a_v`` back to every neighbor that contributed to the mean), and
the direct ``(1 - z)`` carry into the previous state. All of it is
numeric-gradient-checked end to end, including every gate matrix.
All linear algebra is hand-rolled lists-of-floats -- no numpy, no
torch.
"""

from __future__ import annotations

import copy
import math
import random


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _bce(p: float, y: int) -> float:
    p = min(max(p, 1e-7), 1.0 - 1e-7)
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


def _matvec(M, x):
    return [sum(w * xi for w, xi in zip(row, x)) for row in M]


def _matTvec(M, x):
    """M^T @ x for M stored as a list of rows."""
    width = len(M[0])
    out = [0.0] * width
    for i, row in enumerate(M):
        xi = x[i]
        if xi == 0.0:
            continue
        for j in range(width):
            out[j] += row[j] * xi
    return out


def _outer_add(G, a, b) -> None:
    """G += a (x) b, in place."""
    for i, ai in enumerate(a):
        if ai == 0.0:
            continue
        row = G[i]
        for j, bj in enumerate(b):
            row[j] += ai * bj


class GGNN:
    """Gated Graph Neural Network: a ReLU encoder, ``steps`` rounds
    of mean aggregation rewritten by a shared GRU, and a linear head.

    ``predict`` takes a graph and target node ids and returns
    ``{v: probability}``, matching the other models' interface, so
    the shared training/evaluation protocol applies unchanged.
    ``steps=0`` skips propagation entirely: the model is exactly its
    encoder plus the head.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 16,
        steps: int = 3,
        lr: float = 0.5,
        seed: int = 0,
    ):
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        rng = random.Random(seed)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.steps = steps
        self.lr = lr

        def mat(rows, cols, scale):
            return [
                [rng.uniform(-scale, scale) for _ in range(cols)]
                for _ in range(rows)
            ]

        enc_scale = math.sqrt(2.0 / (in_dim + hidden_dim))
        self.W_in = mat(hidden_dim, in_dim, enc_scale)
        self.b_in = [0.0] * hidden_dim
        gate_scale = math.sqrt(2.0 / (hidden_dim + hidden_dim))
        self.W_z = mat(hidden_dim, hidden_dim, gate_scale)
        self.U_z = mat(hidden_dim, hidden_dim, gate_scale)
        self.b_z = [0.0] * hidden_dim
        self.W_r = mat(hidden_dim, hidden_dim, gate_scale)
        self.U_r = mat(hidden_dim, hidden_dim, gate_scale)
        self.b_r = [0.0] * hidden_dim
        self.W_h = mat(hidden_dim, hidden_dim, gate_scale)
        self.U_h = mat(hidden_dim, hidden_dim, gate_scale)
        self.b_h = [0.0] * hidden_dim
        out_scale = math.sqrt(2.0 / (hidden_dim + 1))
        self.w_out = [rng.uniform(-out_scale, out_scale)
                      for _ in range(hidden_dim)]
        self.b_out = 0.0
        self.zero_grad()

    def zero_grad(self) -> None:
        h, d = self.hidden_dim, self.in_dim

        def zeros(rows, cols):
            return [[0.0] * cols for _ in range(rows)]

        self.gW_in = zeros(h, d)
        self.gb_in = [0.0] * h
        self.gW_z = zeros(h, h)
        self.gU_z = zeros(h, h)
        self.gb_z = [0.0] * h
        self.gW_r = zeros(h, h)
        self.gU_r = zeros(h, h)
        self.gb_r = [0.0] * h
        self.gW_h = zeros(h, h)
        self.gU_h = zeros(h, h)
        self.gb_h = [0.0] * h
        self.gw_out = [0.0] * h
        self.gb_out = 0.0

    # -- forward ----------------------------------------------------------

    def _step(self, graph, h_prev):
        """One gated propagation step. Returns (h_new, step_cache)."""
        hd = self.hidden_dim
        h_new = []
        cache = []
        for v in range(graph.n):
            nbrs = graph.neighbors(v)
            if nbrs:
                a = [0.0] * hd
                for u in nbrs:
                    hu = h_prev[u]
                    for j in range(hd):
                        a[j] += hu[j]
                a = [x / len(nbrs) for x in a]
            else:
                a = [0.0] * hd
            hp = h_prev[v]
            wz = _matvec(self.W_z, a)
            uz = _matvec(self.U_z, hp)
            z = [_sigmoid(wz[j] + uz[j] + self.b_z[j]) for j in range(hd)]
            wr = _matvec(self.W_r, a)
            ur = _matvec(self.U_r, hp)
            r = [_sigmoid(wr[j] + ur[j] + self.b_r[j]) for j in range(hd)]
            q = [r[j] * hp[j] for j in range(hd)]
            wh = _matvec(self.W_h, a)
            uh = _matvec(self.U_h, q)
            cand = [math.tanh(wh[j] + uh[j] + self.b_h[j])
                    for j in range(hd)]
            h_new.append([(1.0 - z[j]) * hp[j] + z[j] * cand[j]
                          for j in range(hd)])
            cache.append({"a": a, "z": z, "r": r, "q": q, "cand": cand})
        return h_new, cache

    def forward(self, graph, targets: list[int]):
        """Forward pass over the whole graph. Returns (probs dict,
        cache) with probs only for the requested targets. The cache
        holds the encoder pre-activations, every step's hidden states
        and gate values, backprop's raw material."""
        pre = []
        h = []
        for v in range(graph.n):
            x = graph.features[v]
            p = [sum(w * xi for w, xi in zip(row, x)) + b
                 for row, b in zip(self.W_in, self.b_in)]
            pre.append(p)
            h.append([t if t > 0.0 else 0.0 for t in p])
        states = [h]
        step_caches = []
        for _ in range(self.steps):
            h, sc = self._step(graph, h)
            states.append(h)
            step_caches.append(sc)
        probs: dict[int, float] = {}
        for v in targets:
            logit = (sum(w * hv for w, hv in zip(self.w_out, h[v]))
                     + self.b_out)
            probs[v] = _sigmoid(logit)
        cache = {
            "pre": pre,
            "states": states,
            "steps": step_caches,
            "targets": list(targets),
        }
        return probs, cache

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        """Inference -- propagation is deterministic, so this is exact."""
        probs, _ = self.forward(graph, targets)
        return probs

    def gate_means(self, graph) -> dict[str, float]:
        """Mean update/reset gate values over all nodes and steps, and
        the mean candidate magnitude: a quick read on whether the
        gates are doing anything (a z stuck at its sigmoid(0) = 0.5
        start on both gates means the recurrence is a plain average
        of state and message, not a learned gate)."""
        _, cache = self.forward(graph, [])
        zs, rs = [], []
        for sc in cache["steps"]:
            for entry in sc:
                zs.extend(entry["z"])
                rs.extend(entry["r"])
        if not zs:
            return {"update": 0.5, "reset": 0.5}
        return {
            "update": sum(zs) / len(zs),
            "reset": sum(rs) / len(rs),
        }

    # -- training step -----------------------------------------------------

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    pos_weight: float = 1.0) -> float:
        """One SGD step on ``targets``. Returns mean weighted BCE loss.

        The forward pass embeds every node (a target's gated state
        depends on its neighbors' states), but the loss and its seed
        gradient cover only ``targets``. Backprop runs the steps in
        reverse: at each step the incoming gradient splits into the
        (1 - z) carry, the candidate branch, and the two gate
        branches, and the message gradient is scattered back to the
        neighbors that formed the mean. ``lr=0.0`` accumulates
        gradients without moving the weights, which the numeric
        gradient check relies on.
        """
        self.zero_grad()
        probs, cache = self.forward(graph, targets)
        pre = cache["pre"]
        states = cache["states"]
        step_caches = cache["steps"]
        hd = self.hidden_dim

        B = len(targets)
        wts = {v: (pos_weight if labels[v] == 1 else 1.0)
               for v in targets}
        loss = sum(wts[v] * _bce(probs[v], labels[v])
                   for v in targets) / B

        # Seed gradient on the final states, through the head.
        d_h = [[0.0] * hd for _ in range(graph.n)]
        final = states[-1]
        for v in targets:
            dl = wts[v] * (probs[v] - labels[v]) / B
            if dl == 0.0:
                continue
            for j in range(hd):
                d_h[v][j] = dl * self.w_out[j]
                self.gw_out[j] += dl * final[v][j]
            self.gb_out += dl

        # Backprop through time, last step first.
        for t in range(self.steps - 1, -1, -1):
            h_prev = states[t]
            sc = step_caches[t]
            d_prev = [[0.0] * hd for _ in range(graph.n)]
            d_a_all = []
            for v in range(graph.n):
                dh = d_h[v]
                if not any(dh):
                    d_a_all.append([0.0] * hd)
                    continue
                entry = sc[v]
                a, z, r, q, cand = (entry["a"], entry["z"], entry["r"],
                                    entry["q"], entry["cand"])
                hp = h_prev[v]
                # h_new = (1 - z) * hp + z * cand
                d_z = [dh[j] * (cand[j] - hp[j]) for j in range(hd)]
                d_cand = [dh[j] * z[j] for j in range(hd)]
                d_hp = [dh[j] * (1.0 - z[j]) for j in range(hd)]
                # Candidate branch: cand = tanh(W_h a + U_h q + b_h)
                d_pre_h = [d_cand[j] * (1.0 - cand[j] * cand[j])
                           for j in range(hd)]
                _outer_add(self.gW_h, d_pre_h, a)
                _outer_add(self.gU_h, d_pre_h, q)
                for j in range(hd):
                    self.gb_h[j] += d_pre_h[j]
                d_a = _matTvec(self.W_h, d_pre_h)
                d_q = _matTvec(self.U_h, d_pre_h)
                d_r = [d_q[j] * hp[j] for j in range(hd)]
                for j in range(hd):
                    d_hp[j] += d_q[j] * r[j]
                # Reset gate: r = sigmoid(W_r a + U_r hp + b_r)
                d_pre_r = [d_r[j] * r[j] * (1.0 - r[j]) for j in range(hd)]
                _outer_add(self.gW_r, d_pre_r, a)
                _outer_add(self.gU_r, d_pre_r, hp)
                for j in range(hd):
                    self.gb_r[j] += d_pre_r[j]
                da_r = _matTvec(self.W_r, d_pre_r)
                dhp_r = _matTvec(self.U_r, d_pre_r)
                for j in range(hd):
                    d_a[j] += da_r[j]
                    d_hp[j] += dhp_r[j]
                # Update gate: z = sigmoid(W_z a + U_z hp + b_z)
                d_pre_z = [d_z[j] * z[j] * (1.0 - z[j]) for j in range(hd)]
                _outer_add(self.gW_z, d_pre_z, a)
                _outer_add(self.gU_z, d_pre_z, hp)
                for j in range(hd):
                    self.gb_z[j] += d_pre_z[j]
                da_z = _matTvec(self.W_z, d_pre_z)
                dhp_z = _matTvec(self.U_z, d_pre_z)
                for j in range(hd):
                    d_a[j] += da_z[j]
                    d_hp[j] += dhp_z[j]
                d_a_all.append(d_a)
                for j in range(hd):
                    d_prev[v][j] += d_hp[j]
            # Scatter the message gradient: a_v is the mean of its
            # neighbors' previous states.
            for v in range(graph.n):
                da = d_a_all[v]
                if not any(da):
                    continue
                nbrs = graph.neighbors(v)
                if not nbrs:
                    continue
                share = 1.0 / len(nbrs)
                for u in nbrs:
                    row = d_prev[u]
                    for j in range(hd):
                        row[j] += da[j] * share
            d_h = d_prev

        # Encoder backward: h^(0) = relu(W_in x + b_in).
        for v in range(graph.n):
            dh = d_h[v]
            if not any(dh):
                continue
            x = graph.features[v]
            for j in range(hd):
                d_pre = dh[j] * (1.0 if pre[v][j] > 0.0 else 0.0)
                if d_pre == 0.0:
                    continue
                for k in range(self.in_dim):
                    self.gW_in[j][k] += d_pre * x[k]
                self.gb_in[j] += d_pre

        # SGD update.
        def step_mat(P, G):
            for i in range(len(P)):
                for j in range(len(P[i])):
                    P[i][j] -= self.lr * G[i][j]

        step_mat(self.W_in, self.gW_in)
        step_mat(self.W_z, self.gW_z)
        step_mat(self.U_z, self.gU_z)
        step_mat(self.W_r, self.gW_r)
        step_mat(self.U_r, self.gU_r)
        step_mat(self.W_h, self.gW_h)
        step_mat(self.U_h, self.gU_h)
        for j in range(hd):
            self.b_in[j] -= self.lr * self.gb_in[j]
            self.b_z[j] -= self.lr * self.gb_z[j]
            self.b_r[j] -= self.lr * self.gb_r[j]
            self.b_h[j] -= self.lr * self.gb_h[j]
            self.w_out[j] -= self.lr * self.gw_out[j]
        self.b_out -= self.lr * self.gb_out
        return loss

    # -- checkpointing ------------------------------------------------------

    def state_dict(self):
        return {
            "W_in": copy.deepcopy(self.W_in),
            "b_in": list(self.b_in),
            "W_z": copy.deepcopy(self.W_z),
            "U_z": copy.deepcopy(self.U_z),
            "b_z": list(self.b_z),
            "W_r": copy.deepcopy(self.W_r),
            "U_r": copy.deepcopy(self.U_r),
            "b_r": list(self.b_r),
            "W_h": copy.deepcopy(self.W_h),
            "U_h": copy.deepcopy(self.U_h),
            "b_h": list(self.b_h),
            "w_out": list(self.w_out),
            "b_out": self.b_out,
        }

    def load_state_dict(self, state) -> None:
        self.W_in = copy.deepcopy(state["W_in"])
        self.b_in = list(state["b_in"])
        self.W_z = copy.deepcopy(state["W_z"])
        self.U_z = copy.deepcopy(state["U_z"])
        self.b_z = list(state["b_z"])
        self.W_r = copy.deepcopy(state["W_r"])
        self.U_r = list(state["U_r"])
        self.b_r = list(state["b_r"])
        self.W_h = copy.deepcopy(state["W_h"])
        self.U_h = copy.deepcopy(state["U_h"])
        self.b_h = list(state["b_h"])
        self.w_out = list(state["w_out"])
        self.b_out = state["b_out"]
        self.zero_grad()

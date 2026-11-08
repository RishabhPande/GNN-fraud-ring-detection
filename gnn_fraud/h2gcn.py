"""H2GCN: beyond homophily in graph neural networks (Zhu et al.,
NeurIPS 2020), from scratch.

The GCN family in this repo all mixes a node's own features with its
neighbors' inside one normalized sum, so the head can never tell
whether a signal came from the account itself or from its pocket.
H2GCN is built on the opposite bet, and it is the one architecture
here designed for graphs where neighbors are *not* like the node
itself -- which is exactly a fraud graph, where a ring member's
neighbors are a mix of fellow ring members and ordinary accounts.
Its three design choices, all from the paper:

* **Ego- and neighbor-embedding separation.** At each layer the
  node's own embedding and its neighborhood aggregate are kept in
  separate columns of the concatenated input and transformed by one
  shared matrix, never summed first. The model can weight "I look
  odd" differently from "my counterparties look odd".
* **Higher-order neighborhoods.** The aggregate reads both the 1-hop
  and the 2-hop neighbors (two separate blocks), so a node two hops
  from a ring hub still hears the pocket directly instead of through
  one intermediate that may wash the signal out.
* **Combination of intermediate representations.** The final
  representation concatenates the output of *every* layer (the
  embedding layer plus both message-passing layers), so the head
  sees the raw-feature view, the 1-hop view, and the 2-hop view side
  by side instead of only the deepest one -- the same instinct as
  the JK-GCN in this repo, but with the separation above.

Concretely, with ``H`` the input embeddings of one layer::

    agg1 = A H,  agg2 = A (A H),  H' = relu([H || agg1 || agg2] W + b)

where ``A`` is the row-normalized adjacency *without* self-loops
(the ego block already carries the node itself) and ``||`` is
concatenation along the feature axis. The embedding layer is a
plain ``relu(X W_e + b_e)``. Two message-passing layers with
``hidden_dim = 16`` read up to 4 hops, the same receptive field as
the TAGCN / ChebNet / MixHop stacks, and the head scores the
``3 * hidden_dim`` concatenated final representation.

Like the rest of the GCN family there is no neighbor sampling: the
normalization needs true degrees, so training runs full-batch over
the whole graph. All linear algebra is hand-rolled lists-of-floats,
no numpy, no torch.
"""

from __future__ import annotations

import copy
import math
import random

from gnn_fraud.gcn import _bce, _relu, _sigmoid


def _row_norm_weights(graph) -> list[list[tuple[int, float]]]:
    """Row-normalized adjacency without self-loops: for node ``v``,
    the list of ``(u, 1 / d_v)`` over ``u in N(v)``.

    The ego block carries the node itself, so unlike MixHop's
    weights there is no self-loop term here. Each row sums to 1
    when the node has neighbors; an isolated node gets an empty
    row, so its aggregate is the zero vector, not a NaN.
    """
    weights: list[list[tuple[int, float]]] = []
    for v in range(graph.n):
        nbrs = sorted(set(graph.neighbors(v)))
        if not nbrs:
            weights.append([])
            continue
        c = 1.0 / len(nbrs)
        weights.append([(u, c) for u in nbrs])
    return weights


def _row_norm_weights_T(graph) -> list[list[tuple[int, float]]]:
    """Transpose of :func:`_row_norm_weights`, as the same sparse-row
    format. ``(A^T G)[v]`` sums ``G[u] / d_u`` over ``u`` in ``N(v)``,
    because the weight on the ``u`` row is ``1 / d_u`` and the graph
    is undirected.
    """
    weights: list[list[tuple[int, float]]] = []
    for v in range(graph.n):
        row = [(u, 1.0 / len(set(graph.neighbors(u))))
               for u in sorted(set(graph.neighbors(v)))]
        weights.append(row)
    return weights


def _apply(weights, H: list[list[float]]) -> list[list[float]]:
    """One sparse-matrix multiplication: ``(A H)[v] = sum_{u, c in
    weights[v]} c * H[u]``."""
    out = []
    for v in range(len(weights)):
        acc = [0.0] * len(H[0])
        for u, c in weights[v]:
            hu = H[u]
            for j in range(len(hu)):
                acc[j] += c * hu[j]
        out.append(acc)
    return out


# ---------------------------------------------------------------------------
# one H2GCN layer
# ---------------------------------------------------------------------------


class H2GCNLayer:
    """Single H2GCN layer: ``H' = relu([H || A H || A^2 H] W + b)``.

    ``W`` is ``out_dim x 3 * in_dim``: columns ``0 .. in_dim - 1``
    read the ego block, ``in_dim .. 2 * in_dim - 1`` the 1-hop
    aggregate, ``2 * in_dim .. 3 * in_dim - 1`` the 2-hop aggregate.
    The separation is the point: unlike the GCN, the ego embedding
    is never averaged into its neighbors before the transform.
    """

    def __init__(self, in_dim: int, out_dim: int, rng: random.Random):
        # Xavier-ish init over the full 3 * in_dim input width.
        scale = math.sqrt(2.0 / (3 * in_dim + out_dim))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = [[rng.uniform(-scale, scale) for _ in range(3 * in_dim)]
                  for _ in range(out_dim)]
        self.b = [0.0] * out_dim
        self.zero_grad()

    def zero_grad(self) -> None:
        self.gW = [[0.0] * (3 * self.in_dim)
                   for _ in range(self.out_dim)]
        self.gb = [0.0] * self.out_dim

    def forward(self, graph, H: list[list[float]]):
        """Returns (H_out, cache). Cache holds everything backprop
        needs: (norm weights, transpose weights, input H, the two
        aggregates, the concatenated rows, pre-activations)."""
        weights = _row_norm_weights(graph)
        weights_T = _row_norm_weights_T(graph)
        agg1 = _apply(weights, H)
        agg2 = _apply(weights, agg1)
        concat = []
        for v in range(graph.n):
            concat.append(list(H[v]) + list(agg1[v]) + list(agg2[v]))
        pre = []
        for v in range(graph.n):
            row = list(self.b)
            cv = concat[v]
            for i in range(self.out_dim):
                wi = self.W[i]
                s = 0.0
                for j in range(3 * self.in_dim):
                    s += wi[j] * cv[j]
                row[i] += s
            pre.append(row)
        out = [_relu(row) for row in pre]
        return out, (weights, weights_T, H, agg1, agg2, concat, pre)

    def backward(self, d_out: list[list[float]], cache
                 ) -> list[list[float]]:
        """Backprop one layer. Returns ``d_H``, the gradient w.r.t.
        the input embeddings, indexed by node id.

        The gradient splits back into the three blocks: the ego
        block goes straight to ``d_H``, the 1-hop block goes back
        through ``A^T`` once, the 2-hop block through ``(A^T)^2``.
        ``A`` is row-normalized, not symmetric, so the forward
        weights cannot be reused here.
        """
        weights, weights_T, H, agg1, agg2, concat, pre = cache
        n = len(weights)
        in_dim = self.in_dim
        d_pre = []
        for v in range(n):
            dv = d_out[v]
            pv = pre[v]
            d_pre.append([d * (1.0 if p > 0.0 else 0.0)
                          for d, p in zip(dv, pv)])
        for v in range(n):
            dv = d_pre[v]
            if not any(dv):
                continue
            cv = concat[v]
            for i in range(self.out_dim):
                di = dv[i]
                if di == 0.0:
                    continue
                gWi = self.gW[i]
                for j in range(3 * in_dim):
                    gWi[j] += di * cv[j]
            for i in range(self.out_dim):
                self.gb[i] += dv[i]
        # d_concat = W^T d_pre, split into the three blocks.
        d_ego = [[0.0] * in_dim for _ in range(n)]
        d_agg1 = [[0.0] * in_dim for _ in range(n)]
        d_agg2 = [[0.0] * in_dim for _ in range(n)]
        for v in range(n):
            dv = d_pre[v]
            for i in range(self.out_dim):
                di = dv[i]
                if di == 0.0:
                    continue
                wi = self.W[i]
                re = d_ego[v]
                r1 = d_agg1[v]
                r2 = d_agg2[v]
                for j in range(in_dim):
                    re[j] += wi[j] * di
                    r1[j] += wi[in_dim + j] * di
                    r2[j] += wi[2 * in_dim + j] * di
        # d_H = d_ego + A^T d_agg1 + (A^T)^2 d_agg2.
        d_H = [list(row) for row in d_ego]
        back1 = _apply(weights_T, d_agg1)
        back2 = _apply(weights_T, _apply(weights_T, d_agg2))
        for v in range(n):
            row = d_H[v]
            b1 = back1[v]
            b2 = back2[v]
            for j in range(in_dim):
                row[j] += b1[j] + b2[j]
        return d_H

    def step(self, lr: float) -> None:
        for i in range(self.out_dim):
            for j in range(3 * self.in_dim):
                self.W[i][j] -= lr * self.gW[i][j]
            self.b[i] -= lr * self.gb[i]

    def state_dict(self):
        return {"W": copy.deepcopy(self.W), "b": copy.deepcopy(self.b)}

    def load_state_dict(self, state) -> None:
        self.W = copy.deepcopy(state["W"])
        self.b = copy.deepcopy(state["b"])
        self.zero_grad()


# ---------------------------------------------------------------------------
# two-layer model
# ---------------------------------------------------------------------------


class GraphH2GCN:
    """Two-layer H2GCN for binary node classification.

    An embedding layer maps raw features to ``hidden_dim``
    (``r0``), two H2GCN layers produce ``r1`` and ``r2``, and a
    linear + sigmoid head scores the concatenated ``[r0 || r1 ||
    r2]`` final representation -- every layer's view of the node,
    ego and neighborhood kept separate at each step. Full-graph
    forward (no sampling, see the module docstring), weighted BCE
    loss like :class:`gnn_fraud.gcn.GraphConvolution`.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 16,
        lr: float = 0.5,
        seed: int = 0,
    ):
        self.rng = random.Random(seed)
        self.hidden_dim = hidden_dim
        self.lr = lr
        scale_e = math.sqrt(2.0 / (in_dim + hidden_dim))
        self.W_e = [[self.rng.uniform(-scale_e, scale_e)
                     for _ in range(in_dim)]
                    for _ in range(hidden_dim)]
        self.b_e = [0.0] * hidden_dim
        self.gW_e = [[0.0] * in_dim for _ in range(hidden_dim)]
        self.gb_e = [0.0] * hidden_dim
        self.layer1 = H2GCNLayer(hidden_dim, hidden_dim, self.rng)
        self.layer2 = H2GCNLayer(hidden_dim, hidden_dim, self.rng)
        out_dim = 3 * hidden_dim
        scale = math.sqrt(2.0 / (out_dim + 1))
        self.w_out = [self.rng.uniform(-scale, scale)
                      for _ in range(out_dim)]
        self.b_out = 0.0
        self.gw_out = [0.0] * out_dim
        self.gb_out = 0.0

    def _embed(self, features):
        """Embedding layer: ``r0 = relu(X W_e + b_e)``."""
        r0 = []
        for v in range(len(features)):
            row = list(self.b_e)
            xv = features[v]
            for i in range(self.hidden_dim):
                wi = self.W_e[i]
                s = 0.0
                for j in range(len(xv)):
                    s += wi[j] * xv[j]
                row[i] += s
            r0.append(row)
        return [_relu(row) for row in r0]

    # -- forward ----------------------------------------------------------

    def forward(self, graph, targets: list[int]):
        """Forward pass over the whole graph. Returns (probs dict,
        cache) with probs only for the requested targets."""
        r0 = self._embed(graph.features)
        r1, cache1 = self.layer1.forward(graph, r0)
        r2, cache2 = self.layer2.forward(graph, r1)
        final = []
        for v in range(graph.n):
            final.append(list(r0[v]) + list(r1[v]) + list(r2[v]))

        probs: dict[int, float] = {}
        for v in targets:
            z = sum(w * x for w, x in zip(self.w_out, final[v])) \
                + self.b_out
            probs[v] = _sigmoid(z)

        cache = {"r0": r0, "cache1": cache1, "r1": r1,
                 "r2": r2, "cache2": cache2, "final": final,
                 "targets": list(targets)}
        return probs, cache

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        """Inference, deterministic, no sampling anywhere in H2GCN."""
        probs, _ = self.forward(graph, targets)
        return probs

    def layer_block_norms(self) -> list[float]:
        """L2 norm of the head weights over each layer's block of
        the final representation: how much the head leans on the
        embedding view (r0), the 1-hop view (r1), and the 2-hop
        view (r2)."""
        h = self.hidden_dim
        norms = []
        for k in range(3):
            block = self.w_out[k * h:(k + 1) * h]
            norms.append(math.sqrt(sum(w * w for w in block)))
        return norms

    # -- training step -----------------------------------------------------

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    pos_weight: float = 1.0) -> float:
        """One full-batch SGD step scored on ``targets``. Returns mean
        weighted BCE loss. The forward pass still embeds every node:
        targets' embeddings depend on their 4-hop neighborhoods."""
        self.layer1.zero_grad()
        self.layer2.zero_grad()
        self.gW_e = [[0.0] * len(self.W_e[0])
                     for _ in range(self.hidden_dim)]
        self.gb_e = [0.0] * self.hidden_dim
        self.gw_out = [0.0] * len(self.w_out)
        self.gb_out = 0.0

        probs, cache = self.forward(graph, targets)
        final, cache2 = cache["final"], cache["cache2"]
        h = self.hidden_dim

        B = len(targets)
        weights = {v: (pos_weight if labels[v] == 1 else 1.0)
                   for v in targets}
        loss = sum(weights[v] * _bce(probs[v], labels[v])
                   for v in targets) / B

        # Output layer grads; d_final splits into the three blocks.
        d_final = [[0.0] * (3 * h) for _ in range(graph.n)]
        for v in targets:
            d_logit = weights[v] * (probs[v] - labels[v]) / B
            for j in range(3 * h):
                self.gw_out[j] += d_logit * final[v][j]
            self.gb_out += d_logit
            d_final[v] = [w * d_logit for w in self.w_out]
        d_r0 = [[0.0] * h for _ in range(graph.n)]
        d_r1 = [[0.0] * h for _ in range(graph.n)]
        d_r2 = [[0.0] * h for _ in range(graph.n)]
        for v in range(graph.n):
            dv = d_final[v]
            d_r0[v] = list(dv[0:h])
            d_r1[v] = list(dv[h:2 * h])
            d_r2[v] = list(dv[2 * h:3 * h])

        # Backprop through layer 2, then layer 1, then the embedding.
        d_r1_from_l2 = self.layer2.backward(d_r2, cache2)
        for v in range(graph.n):
            for j in range(h):
                d_r1[v][j] += d_r1_from_l2[v][j]
        d_r0_from_l1 = self.layer1.backward(d_r1, cache["cache1"])
        for v in range(graph.n):
            for j in range(h):
                d_r0[v][j] += d_r0_from_l1[v][j]
        # Embedding layer: r0 = relu(X W_e + b_e). r0 is zero exactly
        # where the pre-activation was non-positive, so the ReLU mask
        # reads off the output itself.
        for v in range(graph.n):
            dv = d_r0[v]
            if not any(dv):
                continue
            rv = cache["r0"][v]
            xv = graph.features[v]
            for i in range(h):
                if rv[i] <= 0.0:
                    continue
                di = dv[i]
                if di == 0.0:
                    continue
                gWi = self.gW_e[i]
                for j in range(len(xv)):
                    gWi[j] += di * xv[j]
                self.gb_e[i] += di

        # SGD update.
        self.layer1.step(self.lr)
        self.layer2.step(self.lr)
        for i in range(h):
            for j in range(len(self.W_e[0])):
                self.W_e[i][j] -= self.lr * self.gW_e[i][j]
            self.b_e[i] -= self.lr * self.gb_e[i]
        for j in range(len(self.w_out)):
            self.w_out[j] -= self.lr * self.gw_out[j]
        self.b_out -= self.lr * self.gb_out
        return loss

    # -- checkpointing ------------------------------------------------------

    def state_dict(self):
        return {
            "W_e": copy.deepcopy(self.W_e),
            "b_e": copy.deepcopy(self.b_e),
            "layer1": self.layer1.state_dict(),
            "layer2": self.layer2.state_dict(),
            "w_out": copy.deepcopy(self.w_out),
            "b_out": self.b_out,
        }

    def load_state_dict(self, state) -> None:
        self.W_e = copy.deepcopy(state["W_e"])
        self.b_e = copy.deepcopy(state["b_e"])
        self.layer1.load_state_dict(state["layer1"])
        self.layer2.load_state_dict(state["layer2"])
        self.w_out = copy.deepcopy(state["w_out"])
        self.b_out = state["b_out"]

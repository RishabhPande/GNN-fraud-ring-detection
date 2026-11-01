"""Jumping Knowledge GCN (Xu et al., 2018), from scratch.

A two-layer GCN reads only its last layer: by layer 2 every node's
embedding is a 2-hop neighborhood summary, and the 1-hop summary is
thrown away. That is a problem when the two hops carry different
signals. On a fraud graph, layer 1 says "my direct counterparties look
like this" and layer 2 says "my counterparties' counterparties look
like this"; a ring member's own dense pocket and its wider
neighborhood are not the same evidence, and forcing one embedding to
carry both washes the sharper one out.

Jumping Knowledge keeps both. The model reuses the GCN layers in
:mod:`gnn_fraud.gcn`, concatenates the layer-1 and layer-2 embeddings
for each node, and scores the concatenation::

    h1 = relu(S X W1),  h2 = relu(S h1 W2),  p_v = sigmoid(w @ [h1_v || h2_v] + b)

where ``S`` is the symmetric-normalized adjacency with self-loops.
The concatenation variant is the simplest JK aggregator: no learned
attention over layers, no extra parameters beyond a wider output head
(``2 * hidden_dim`` weights instead of ``hidden_dim``). Backprop splits
at the concatenation: the head's first half of the gradient flows
straight to layer 1, and its second half flows through layer 2 and then
adds to layer 1's gradient, so layer 1 learns from both its direct
contribution to the score and its role as layer 2's input.

Why it belongs here: the repo's APPNP finding is that deep propagation
holds up on the dense ring, and its SGC finding is that propagated
features alone nearly match the GCN. JK asks the complementary
question: does keeping the shallow representation alongside the deep
one buy anything on a graph where the signal is a dense 1-hop pocket
sitting inside a sparser 2-hop neighborhood? If it matches the plain
GCN, the last layer already carries what matters here. If it beats it,
the shallow signal was being washed out. All linear algebra is
hand-rolled lists-of-floats, no numpy, no torch.
"""

from __future__ import annotations

import copy
import math
import random

from gnn_fraud.gcn import GCNLayer, _bce, _sigmoid


class JumpingKnowledgeGCN:
    """Two-layer GCN whose output head reads [layer-1 || layer-2].

    Full-graph forward (no sampling, like :class:`gnn_fraud.gcn.GraphConvolution`:
    the normalization needs true degrees), linear + sigmoid output head over
    the concatenated embeddings, weighted BCE loss, full backprop through
    both convolutional layers and the head.
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
        self.layer1 = GCNLayer(in_dim, hidden_dim, self.rng)
        self.layer2 = GCNLayer(hidden_dim, hidden_dim, self.rng)
        scale = math.sqrt(2.0 / (2 * hidden_dim + 1))
        self.w_out = [
            self.rng.uniform(-scale, scale) for _ in range(2 * hidden_dim)
        ]
        self.b_out = 0.0
        self.gw_out = [0.0] * (2 * hidden_dim)
        self.gb_out = 0.0

    # -- forward ----------------------------------------------------------

    def forward(self, graph, targets: list[int]):
        """Forward pass over the whole graph. Returns (probs dict, cache)
        with probs only for the requested targets."""
        h1, cache1 = self.layer1.forward(graph, graph.features)
        h2, cache2 = self.layer2.forward(graph, h1)

        probs: dict[int, float] = {}
        for v in targets:
            z = sum(
                w * h
                for w, h in zip(self.w_out, h1[v] + h2[v])
            ) + self.b_out
            probs[v] = _sigmoid(z)

        cache = {
            "h1": h1,
            "cache1": cache1,
            "h2": h2,
            "cache2": cache2,
            "targets": list(targets),
        }
        return probs, cache

    def predict(self, graph, targets: list[int]) -> dict[int, float]:
        """Inference: deterministic, no sampling anywhere in the model."""
        probs, _ = self.forward(graph, targets)
        return probs

    # -- training step -----------------------------------------------------

    def train_batch(self, graph, targets: list[int], labels: dict[int, int],
                    pos_weight: float = 1.0) -> float:
        """One full-batch SGD step scored on ``targets``. Returns mean
        weighted BCE loss. The forward pass still embeds every node:
        targets' embeddings depend on their 2-hop neighborhoods."""
        self.layer1.zero_grad()
        self.layer2.zero_grad()
        self.gw_out = [0.0] * (2 * self.hidden_dim)
        self.gb_out = 0.0

        probs, cache = self.forward(graph, targets)
        h1, h2 = cache["h1"], cache["h2"]

        B = len(targets)
        weights = {v: (pos_weight if labels[v] == 1 else 1.0)
                   for v in targets}
        loss = sum(weights[v] * _bce(probs[v], labels[v])
                   for v in targets) / B

        # Output layer grads, and the gradient split at the concatenation:
        # the first hidden_dim entries belong to h1, the rest to h2.
        dh1 = [[0.0] * self.hidden_dim for _ in range(graph.n)]
        dh2 = [[0.0] * self.hidden_dim for _ in range(graph.n)]
        for v in targets:
            d_logit = weights[v] * (probs[v] - labels[v]) / B
            for j in range(self.hidden_dim):
                self.gw_out[j] += d_logit * h1[v][j]
                self.gw_out[self.hidden_dim + j] += d_logit * h2[v][j]
            self.gb_out += d_logit
            dh1[v] = [self.w_out[j] * d_logit
                      for j in range(self.hidden_dim)]
            dh2[v] = [self.w_out[self.hidden_dim + j] * d_logit
                      for j in range(self.hidden_dim)]

        # Backprop through layer 2, then add its gradient into layer 1's
        # direct gradient before backprop through layer 1: layer 1 is
        # scored directly AND feeds layer 2, so it gets both gradients.
        dh1_via_layer2 = self.layer2.backward(dh2, cache["cache2"])
        dh1_total = [
            [a + b for a, b in zip(dh1[v], dh1_via_layer2[v])]
            for v in range(graph.n)
        ]
        self.layer1.backward(dh1_total, cache["cache1"])

        # SGD update.
        self.layer1.step(self.lr)
        self.layer2.step(self.lr)
        for j in range(2 * self.hidden_dim):
            self.w_out[j] -= self.lr * self.gw_out[j]
        self.b_out -= self.lr * self.gb_out
        return loss

    # -- checkpointing ------------------------------------------------------

    def state_dict(self):
        return {
            "layer1": self.layer1.state_dict(),
            "layer2": self.layer2.state_dict(),
            "w_out": copy.deepcopy(self.w_out),
            "b_out": self.b_out,
        }

    def load_state_dict(self, state) -> None:
        self.layer1.load_state_dict(state["layer1"])
        self.layer2.load_state_dict(state["layer2"])
        self.w_out = copy.deepcopy(state["w_out"])
        self.b_out = state["b_out"]
        self.gw_out = [0.0] * (2 * self.hidden_dim)
        self.gb_out = 0.0

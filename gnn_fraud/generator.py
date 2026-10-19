"""Synthetic transaction-graph generator with a planted fraud ring.

Every account node carries intrinsic features::

    [account_age, avg_txn_amount, txn_count, balance, night_txn_ratio,
     unique_counterparties]                      # all normalized to [0, 1]

Benign accounts link up sparsely at random. A fraud ring is a set of
accounts wired into a dense subgraph -- with *near-camouflaged* features:
ring members draw most features from the same distributions as benign
accounts, but carry subtle behavioral tells (slightly higher night-time
activity, burst transaction counts, more counterparties). Each tell is
weak on its own -- a single account looks normal -- yet when message
passing pools features over the ring's dense neighborhood, the tells
*denoise* into a strong signal. That is the whole story: 1-hop features
miss the ring; 2-hop aggregation catches it.

All data is synthetic. No real accounts, no real transactions, no PII.
"""

from __future__ import annotations

import random

from gnn_fraud.graph import Graph

FEATURE_NAMES = [
    "account_age",
    "avg_txn_amount",
    "txn_count",
    "balance",
    "night_txn_ratio",
    "unique_counterparties",
]

# Per-feature (mean, std) for benign accounts vs ring members. The ring's
# tells are deliberately weak (Cohen's d ~ 0.5): individually deniable,
# collectively damning once pooled over a dense neighborhood.
BENIGN_DIST = {
    "account_age": (0.55, 0.25),
    "avg_txn_amount": (0.35, 0.20),
    "txn_count": (0.40, 0.20),
    "balance": (0.50, 0.25),
    "night_txn_ratio": (0.15, 0.10),
    "unique_counterparties": (0.45, 0.20),
}
RING_DIST = {
    "account_age": (0.55, 0.25),          # camouflaged: identical
    "avg_txn_amount": (0.35, 0.20),      # camouflaged: identical
    "txn_count": (0.50, 0.18),           # tell: burst activity
    "balance": (0.50, 0.25),             # camouflaged: identical
    "night_txn_ratio": (0.21, 0.10),     # tell: night activity
    "unique_counterparties": (0.55, 0.18),  # tell: many counterparties
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class GraphGenerator:
    """Generates a transaction graph with one planted fraud ring.

    Fraud rings form fast: ring edges land inside a short burst window late
    in the timeline, while benign edges spread uniformly over it. That
    temporal signature is what :mod:`gnn_fraud.temporal` exploits.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        # Separate stream for timestamps, so turning them on or off never
        # changes the planted structure (pinned seeds stay reproducible).
        self.trng = random.Random(seed ^ 0x9E3779B9)

    def _burst_time(self, burst_start: float, timeline_days: float) -> float:
        return self.trng.uniform(burst_start, timeline_days)

    def _background_time(self, timeline_days: float) -> float:
        return self.trng.uniform(0.0, timeline_days)

    def _sample_features(self, fraud: bool) -> list[float]:
        dist = RING_DIST if fraud else BENIGN_DIST
        return [
            _clamp01(self.rng.gauss(*dist[name])) for name in FEATURE_NAMES
        ]

    def generate(
        self,
        n: int = 200,
        ring_size: int = 18,
        edge_prob: float = 0.008,
        ring_edge_prob: float = 0.70,
        ring_bridge_edges: int = 6,
        timestamps: bool = True,
        timeline_days: float = 30.0,
        ring_burst_days: float = 3.0,
    ) -> Graph:
        """Build the graph.

        Args:
            n: total account nodes.
            ring_size: number of accounts in the planted fraud ring.
            edge_prob: probability of a benign edge between any two
                non-ring accounts.
            ring_edge_prob: probability of an edge between any two ring
                members (the dense subgraph).
            ring_bridge_edges: a few edges from ring members to benign
                accounts, so the ring is not trivially isolated.
            timestamps: stamp every edge with a time. Ring edges land in a
                short burst late in the timeline (rings form fast); benign
                and bridge edges spread uniformly over the whole timeline.
            timeline_days: length of the observation window ("now" is its
                end).
            ring_burst_days: width of the ring's formation burst window.
        """
        if ring_size >= n:
            raise ValueError("ring_size must be smaller than n")
        g = Graph(n, len(FEATURE_NAMES))
        burst_start = timeline_days - ring_burst_days

        ring = self.rng.sample(range(n), ring_size)
        ring_set = set(ring)

        # Features: ring members near-camouflage -- most features match
        # benign, with only subtle behavioral tells (see RING_DIST).
        for u in range(n):
            g.set_features(u, self._sample_features(fraud=(u in ring_set)))

        # Labels.
        for u in ring:
            g.set_label(u, 1)

        # Benign edges: sparse random wiring between non-ring accounts,
        # spread uniformly over the timeline.
        benign = [u for u in range(n) if u not in ring_set]
        for i, u in enumerate(benign):
            for v in benign[i + 1:]:
                if self.rng.random() < edge_prob:
                    g.add_edge(u, v, time=(
                        self._background_time(timeline_days)
                        if timestamps else None))

        # Ring edges: dense subgraph among ring members, formed in a fast
        # late burst -- the temporal signature of a real ring.
        for i, u in enumerate(ring):
            for v in ring[i + 1:]:
                if self.rng.random() < ring_edge_prob:
                    g.add_edge(u, v, time=(
                        self._burst_time(burst_start, timeline_days)
                        if timestamps else None))

        # Bridge edges: a few links from the ring to the outside world.
        for _ in range(ring_bridge_edges):
            u = self.rng.choice(ring)
            v = self.rng.choice(benign)
            g.add_edge(u, v, time=(
                self._background_time(timeline_days)
                if timestamps else None))

        return g

    def ring_nodes(self, g: Graph) -> list[int]:
        """Node ids labeled as fraud (the planted ring)."""
        return [u for u in range(g.n) if g.labels[u] == 1]

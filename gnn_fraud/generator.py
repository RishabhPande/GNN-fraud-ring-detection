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

# Mule-chain roles. Money flows originator -> mule chains -> consolidator
# -> exit. Mules are pass-through accounts: money doesn't sit, it moves.
# Same discipline as RING_DIST: weak per-feature tells, near-camouflaged.
MULE_DIST = {
    "account_age": (0.45, 0.22),         # tell: fresh accounts
    "avg_txn_amount": (0.35, 0.20),      # camouflaged: identical
    "txn_count": (0.55, 0.18),           # tell: burst velocity
    "balance": (0.40, 0.20),             # tell: pass-through, low retention
    "night_txn_ratio": (0.21, 0.10),     # tell: night activity
    "unique_counterparties": (0.55, 0.18),  # tell: fan-in/fan-out
}
ORGANIZER_DIST = {
    "account_age": (0.60, 0.22),
    "avg_txn_amount": (0.35, 0.20),
    "txn_count": (0.60, 0.18),           # tell: orchestrates many chains
    "balance": (0.50, 0.25),
    "night_txn_ratio": (0.20, 0.10),
    "unique_counterparties": (0.65, 0.15),  # tell: touches every chain
}
EXIT_DIST = {
    "account_age": (0.45, 0.22),
    "avg_txn_amount": (0.45, 0.20),      # tell: larger cash-out amounts
    "txn_count": (0.45, 0.20),
    "balance": (0.30, 0.18),             # tell: emptied on cash-out
    "night_txn_ratio": (0.15, 0.10),
    "unique_counterparties": (0.30, 0.15),  # tell: talks to almost nobody
}

# Smurfing roles. One source splits a large sum into many small amounts
# -- each just below a reporting threshold -- routed through smurf
# intermediaries to a single collector. Smurfs are pass-through accounts
# like mules, but the signature is fan-out/fan-in, not a chain: the
# source and the collector are hubs, every smurf hangs off both.
SMURF_DIST = {
    "account_age": (0.45, 0.22),         # tell: fresh accounts
    "avg_txn_amount": (0.28, 0.15),      # tell: small structured amounts
    "txn_count": (0.55, 0.18),           # tell: many small transactions
    "balance": (0.40, 0.20),             # tell: pass-through, low retention
    "night_txn_ratio": (0.20, 0.10),     # tell: night activity
    "unique_counterparties": (0.45, 0.20),  # camouflaged: source + sink only
}
SOURCE_DIST = {
    "account_age": (0.60, 0.22),
    "avg_txn_amount": (0.28, 0.15),      # tell: sends the small amounts
    "txn_count": (0.65, 0.15),           # tell: fans out to every smurf
    "balance": (0.50, 0.25),
    "night_txn_ratio": (0.20, 0.10),
    "unique_counterparties": (0.70, 0.15),  # tell: touches every smurf
}
SINK_DIST = {
    "account_age": (0.50, 0.22),
    "avg_txn_amount": (0.45, 0.20),      # tell: consolidated cash-out
    "txn_count": (0.50, 0.18),
    "balance": (0.30, 0.18),             # tell: emptied on cash-out
    "night_txn_ratio": (0.20, 0.10),
    "unique_counterparties": (0.65, 0.15),  # tell: fans in from every smurf
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

    def _sample_from(self, dist: dict) -> list[float]:
        return [_clamp01(self.rng.gauss(*dist[name])) for name in FEATURE_NAMES]

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

    def generate_mule_chain(
        self,
        n: int = 200,
        n_chains: int = 4,
        chain_len: int = 4,
        edge_prob: float = 0.008,
        bridge_edges: int = 6,
        timestamps: bool = True,
        timeline_days: float = 30.0,
        burst_days: float = 3.0,
    ) -> Graph:
        """Build a graph with a planted money-mule chain network.

        A second fraud pattern alongside the dense pocket in
        :meth:`generate`. Money flows::

            originator -> [mule chain x n_chains] -> consolidator -> exit

        each chain a path of ``chain_len`` pass-through accounts. The
        signature is structural, not density-based: long thin paths with
        low triangle counts -- nothing like a dense subgraph. Hops along
        each chain are fast and late in the timeline (money moves
        downstream within the burst window), so the temporal model sees a
        sequential burst instead of a simultaneous one.

        Args:
            n: total account nodes.
            n_chains: number of parallel mule chains.
            chain_len: accounts per chain.
            edge_prob: probability of a benign edge between any two
                non-fraud accounts.
            bridge_edges: a few edges from chain accounts to benign
                accounts, so the network is not trivially isolated.
            timestamps: stamp every edge with a time. Chain hops land in
                a fast late burst, ordered along the chain (monotone
                downstream); benign and bridge edges spread uniformly.
            timeline_days: length of the observation window ("now" is its
                end).
            burst_days: width of the late burst window the hops land in.

        Returns:
            The graph. The planted layout is attached as
            ``graph.mule_chain`` -- a dict with keys ``originator``,
            ``chains`` (list of node-id lists, money flows in list order),
            ``consolidator``, and ``exit``. All fraud nodes are labeled 1
            (so :meth:`ring_nodes` lists them too).
        """
        fraud_n = 3 + n_chains * chain_len  # originator, consolidator, exit + mules
        if fraud_n >= n:
            raise ValueError("mule-chain network must be smaller than n")
        g = Graph(n, len(FEATURE_NAMES))
        burst_start = timeline_days - burst_days

        fraud_ids = self.rng.sample(range(n), fraud_n)
        originator, rest = fraud_ids[0], fraud_ids[1:]
        exit_node = rest[-1]
        consolidator = rest[-2]
        chain_ids = rest[:-2]
        chains = [
            chain_ids[k * chain_len:(k + 1) * chain_len]
            for k in range(n_chains)
        ]
        fraud_set = set(fraud_ids)

        # Features: mules are near-camouflaged pass-through accounts;
        # the organizer fans out, the exit node cashes out quietly.
        for u in range(n):
            if u == originator:
                dist = ORGANIZER_DIST
            elif u == exit_node:
                dist = EXIT_DIST
            elif u == consolidator:
                dist = ORGANIZER_DIST  # consolidator also fans in/out
            elif u in fraud_set:
                dist = MULE_DIST
            else:
                dist = BENIGN_DIST
            g.set_features(u, self._sample_from(dist))

        # Labels: the whole laundering network is fraud.
        for u in fraud_ids:
            g.set_label(u, 1)

        # Benign edges between non-fraud accounts.
        benign = [u for u in range(n) if u not in fraud_set]
        for i, u in enumerate(benign):
            for v in benign[i + 1:]:
                if self.rng.random() < edge_prob:
                    g.add_edge(u, v, time=(
                        self._background_time(timeline_days)
                        if timestamps else None))

        # Chain edges: originator -> chain -> consolidator -> exit.
        # Hops are fast and late: each chain starts at a random point in
        # the burst window and moves downstream with short gaps, so money
        # visibly travels the chain before the window closes.
        def _ts(t: float) -> float | None:
            return t if timestamps else None

        # The exit hop is shared by all chains: money leaves after the
        # chains complete, stamped once at the end of the timeline (not
        # re-stamped per chain, which would scramble per-chain ordering).
        g.add_edge(consolidator, exit_node, time=_ts(timeline_days))

        for chain in chains:
            t = self.trng.uniform(burst_start, timeline_days - 1.0)
            prev = originator
            for u in chain + [consolidator]:
                g.add_edge(prev, u, time=_ts(t))
                t = min(t + self.trng.uniform(0.2, 0.8), timeline_days)
                prev = u

        # Bridge edges from chain accounts to benign accounts.
        mules = [u for chain in chains for u in chain]
        for _ in range(bridge_edges):
            u = self.rng.choice(mules)
            v = self.rng.choice(benign)
            g.add_edge(u, v, time=(
                self._background_time(timeline_days)
                if timestamps else None))

        g.mule_chain = {
            "originator": originator,
            "chains": chains,
            "consolidator": consolidator,
            "exit": exit_node,
        }
        return g

    def generate_smurfing(
        self,
        n: int = 200,
        n_smurfs: int = 16,
        edge_prob: float = 0.008,
        bridge_edges: int = 6,
        timestamps: bool = True,
        timeline_days: float = 30.0,
        burst_days: float = 3.0,
    ) -> Graph:
        """Build a graph with a planted smurfing (structuring) network.

        A third fraud pattern alongside the dense pocket in
        :meth:`generate` and the mule chains in
        :meth:`generate_mule_chain`. One source splits a large sum into
        many small amounts -- each just below a reporting threshold --
        and routes them through ``n_smurfs`` intermediaries to a single
        collector::

            source -> [smurf x n_smurfs] -> sink

        The signature is fan-out/fan-in, not density and not a path:
        the source and the sink are hubs (degree ``n_smurfs`` inside the
        network), every smurf hangs off exactly those two, and smurfs
        never transact with each other, so the network has no triangles
        at all. Each smurf's hop pair is fast and late: money arrives
        from the source and leaves for the sink inside the burst window,
        in that order, so the temporal model sees a fan of near-simultaneous
        structured hops rather than a sequential chain.

        Args:
            n: total account nodes.
            n_smurfs: number of smurf intermediaries.
            edge_prob: probability of a benign edge between any two
                non-fraud accounts.
            bridge_edges: a few edges from smurfs to benign accounts, so
                the network is not trivially isolated.
            timestamps: stamp every edge with a time. Smurf hops land
                in a fast late burst, in-flow before out-flow per smurf;
                benign and bridge edges spread uniformly.
            timeline_days: length of the observation window ("now" is its
                end).
            burst_days: width of the late burst window the hops land in.

        Returns:
            The graph. The planted layout is attached as
            ``graph.smurfing`` -- a dict with keys ``source``,
            ``smurfs`` (list of node ids), and ``sink``. All fraud nodes
            are labeled 1 (so :meth:`ring_nodes` lists them too).
        """
        fraud_n = n_smurfs + 2  # source + smurfs + sink
        if fraud_n >= n:
            raise ValueError("smurfing network must be smaller than n")
        g = Graph(n, len(FEATURE_NAMES))
        burst_start = timeline_days - burst_days

        fraud_ids = self.rng.sample(range(n), fraud_n)
        source = fraud_ids[0]
        sink = fraud_ids[-1]
        smurfs = fraud_ids[1:-1]
        fraud_set = set(fraud_ids)

        # Features: smurfs are near-camouflaged pass-through accounts
        # moving small amounts; the source fans out, the sink cashes out.
        for u in range(n):
            if u == source:
                dist = SOURCE_DIST
            elif u == sink:
                dist = SINK_DIST
            elif u in fraud_set:
                dist = SMURF_DIST
            else:
                dist = BENIGN_DIST
            g.set_features(u, self._sample_from(dist))

        # Labels: the whole structuring network is fraud.
        for u in fraud_ids:
            g.set_label(u, 1)

        # Benign edges between non-fraud accounts.
        benign = [u for u in range(n) if u not in fraud_set]
        for i, u in enumerate(benign):
            for v in benign[i + 1:]:
                if self.rng.random() < edge_prob:
                    g.add_edge(u, v, time=(
                        self._background_time(timeline_days)
                        if timestamps else None))

        # Smurf hops: source -> smurf -> sink, in-flow before out-flow,
        # both inside the late burst window.
        def _ts(t: float) -> float | None:
            return t if timestamps else None

        for s in smurfs:
            t_in = self.trng.uniform(burst_start, timeline_days - 1.0)
            t_out = min(t_in + self.trng.uniform(0.2, 1.0), timeline_days)
            g.add_edge(source, s, time=_ts(t_in))
            g.add_edge(s, sink, time=_ts(t_out))

        # Bridge edges from smurfs to benign accounts.
        for _ in range(bridge_edges):
            u = self.rng.choice(smurfs)
            v = self.rng.choice(benign)
            g.add_edge(u, v, time=(
                self._background_time(timeline_days)
                if timestamps else None))

        g.smurfing = {
            "source": source,
            "smurfs": smurfs,
            "sink": sink,
        }
        return g

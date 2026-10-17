"""Synthetic heterogeneous transaction graph with a planted fraud ring.

Accounts, devices, and merchants are distinct node types wired by typed
relations (see :mod:`gnn_fraud.hetero`). The planted ring carries two
*structural* tells beyond its camouflaged account features:

1. **Shared devices** -- ring members log in from a small pool of shared
   devices, while benign accounts mostly use their own device.
2. **Shared merchants** -- ring members funnel payments through a handful
   of shared merchants (the mule-merchant pattern).

Each tell is individually deniable (households share devices too), but a
relation-aware model learns which *kind* of neighbor matters: a shared
device is damning, a one-off counterparty is not. That is exactly where
the homogeneous models fall short -- and, per the README, where a
heterogeneous model earns its keep.

All data is synthetic. No real accounts, no real transactions, no PII.
"""

from __future__ import annotations

import random

from gnn_fraud.generator import BENIGN_DIST, FEATURE_NAMES, RING_DIST
from gnn_fraud.hetero import (
    ACCOUNT,
    DEVICE,
    MERCHANT,
    HeteroGraph,
    PAYS,
    TRANSFERS,
    USED_BY,
    USES,
)

DEVICE_FEATURE_NAMES = ["device_age", "login_freq", "login_hour_variance"]

# Benign vs ring-shared device behavior. Shared ring devices skew young
# (fresh devices bought for the operation); everything else matches.
BENIGN_DEVICE_DIST = {
    "device_age": (0.60, 0.25),
    "login_freq": (0.40, 0.20),
    "login_hour_variance": (0.30, 0.15),
}
RING_DEVICE_DIST = {
    "device_age": (0.45, 0.22),          # tell: younger devices
    "login_freq": (0.48, 0.20),          # tell: slightly busier
    "login_hour_variance": (0.30, 0.15),  # camouflaged: identical
}

MERCHANT_FEATURE_NAMES = ["merchant_age", "avg_ticket", "category_risk"]

BENIGN_MERCHANT_DIST = {
    "merchant_age": (0.55, 0.25),
    "avg_ticket": (0.35, 0.20),
    "category_risk": (0.25, 0.15),
}
RING_MERCHANT_DIST = {
    "merchant_age": (0.50, 0.25),        # camouflaged: near-identical
    "avg_ticket": (0.35, 0.20),          # camouflaged: identical
    "category_risk": (0.33, 0.15),       # tell: slightly riskier category
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class HeteroGenerator:
    """Generates a heterogeneous graph with one planted fraud ring."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _sample(self, dist: dict[str, tuple[float, float]],
                names: list[str]) -> list[float]:
        return [_clamp01(self.rng.gauss(*dist[name])) for name in names]

    def generate(
        self,
        n_accounts: int = 200,
        ring_size: int = 18,
        n_devices: int | None = None,
        n_merchants: int = 50,
        ring_device_count: int = 3,
        ring_merchant_count: int = 4,
        transfer_prob: float = 0.008,
        ring_transfer_prob: float = 0.70,
        ring_bridge_transfers: int = 6,
    ) -> HeteroGraph:
        """Build the heterogeneous graph.

        Args:
            n_accounts: total account nodes (added first, ids 0..n-1).
            ring_size: number of ring members.
            n_devices: total device nodes; defaults to one per account.
            n_merchants: total merchant nodes.
            ring_device_count: size of the shared device pool for the ring.
            ring_merchant_count: size of the shared merchant pool for the ring.
            transfer_prob: benign account-to-account transfer probability.
            ring_transfer_prob: ring-internal transfer probability.
            ring_bridge_transfers: ring-to-benign transfer edges.
        """
        if ring_size >= n_accounts:
            raise ValueError("ring_size must be smaller than n_accounts")
        n_devices = n_devices if n_devices is not None else n_accounts

        g = HeteroGraph({
            ACCOUNT: len(FEATURE_NAMES),
            DEVICE: len(DEVICE_FEATURE_NAMES),
            MERCHANT: len(MERCHANT_FEATURE_NAMES),
        })

        ring_ids = set(self.rng.sample(range(n_accounts), ring_size))
        accounts = [g.add_node(ACCOUNT) for _ in range(n_accounts)]
        benign = [a for a in accounts if a not in ring_ids]
        ring = [a for a in accounts if a in ring_ids]

        devices = [g.add_node(DEVICE) for _ in range(n_devices)]
        merchants = [g.add_node(MERCHANT) for _ in range(n_merchants)]

        # -- features ----------------------------------------------------
        for a in accounts:
            dist = RING_DIST if a in ring_ids else BENIGN_DIST
            g.set_features(a, self._sample(dist, FEATURE_NAMES))
        for a in ring:
            g.set_label(a, 1)
        for a in benign:
            g.set_label(a, 0)

        ring_devices = self.rng.sample(devices, ring_device_count)
        ring_device_set = set(ring_devices)
        for d in devices:
            dist = RING_DEVICE_DIST if d in ring_device_set else BENIGN_DEVICE_DIST
            g.set_features(d, self._sample(dist, DEVICE_FEATURE_NAMES))

        ring_merchants = self.rng.sample(merchants, ring_merchant_count)
        ring_merchant_set = set(ring_merchants)
        for m in merchants:
            dist = (RING_MERCHANT_DIST if m in ring_merchant_set
                    else BENIGN_MERCHANT_DIST)
            g.set_features(m, self._sample(dist, MERCHANT_FEATURE_NAMES))

        # -- wiring ------------------------------------------------------
        # Devices: ring members share a small pool; benign accounts mostly
        # get a private device, occasionally sharing (households do too).
        free_devices = [d for d in devices if d not in ring_device_set]
        for a in ring:
            for d in self.rng.sample(ring_devices,
                                     self.rng.randint(1, min(2, ring_device_count))):
                g.add_uses(a, d)
        for a in benign:
            if self.rng.random() < 0.75 and free_devices:
                g.add_uses(a, free_devices.pop())
            else:
                g.add_uses(a, self.rng.choice(devices))

        # Merchants: ring funnels through shared merchants; benign pays
        # 2-5 random merchants.
        for a in ring:
            for m in self.rng.sample(ring_merchants,
                                     self.rng.randint(1, min(3, ring_merchant_count))):
                g.add_pays(a, m)
        for a in benign:
            for m in self.rng.sample(merchants, self.rng.randint(2, 5)):
                g.add_pays(a, m)

        # Transfers: dense ring subgraph, sparse benign wiring, a few
        # bridges -- same recipe as the homogeneous generator.
        for i, a in enumerate(benign):
            for b in benign[i + 1:]:
                if self.rng.random() < transfer_prob:
                    g.add_transfer(a, b)
        for i, a in enumerate(ring):
            for b in ring[i + 1:]:
                if self.rng.random() < ring_transfer_prob:
                    g.add_transfer(a, b)
        for _ in range(ring_bridge_transfers):
            g.add_transfer(self.rng.choice(ring), self.rng.choice(benign))

        return g

    def ring_nodes(self, g: HeteroGraph) -> list[int]:
        """Account ids labeled as fraud (the planted ring)."""
        return [v for v in g.account_ids() if g.labels[v] == 1]

    def ring_devices(self, g: HeteroGraph, ring: list[int] | None = None) -> list[int]:
        """Devices used by at least 3 ring members (the shared pool)."""
        ring = ring if ring is not None else self.ring_nodes(g)
        ring_set = set(ring)
        out = []
        for v in range(g.n):
            if g.node_type[v] != DEVICE:
                continue
            users = [u for u in g.rel_neighbors(v, USED_BY) if u in ring_set]
            if len(users) >= 3:
                out.append(v)
        return out

"""Heterogeneous transaction graph: accounts, devices, and merchants.

A fraud ring is not just accounts wiring to each other. In practice the
wiring shows up *across* entity types: several accounts logging in from the
same device, or a cluster of accounts all paying the same handful of
mule merchants. Collapsing everything to one node type (the homogeneous
graph in ``graph.py``) throws that signal away. This module keeps the
types distinct and models every edge with an explicit relation:

    account --USES-->     device     (reverse: USED_BY)
    account --PAYS-->     merchant   (reverse: PAID_BY)
    account --TRANSFERS--> account    (symmetric)

All data is synthetic. No real accounts, no real transactions, no PII.
"""

from __future__ import annotations

# Node types.
ACCOUNT = 0
DEVICE = 1
MERCHANT = 2
NODE_TYPE_NAMES = {ACCOUNT: "account", DEVICE: "device", MERCHANT: "merchant"}

# Directed relation ids. Reverse relations get their own id so the
# relation-aware model can learn asymmetric message weights
# (a device tells you more about an account than the account's
# neighbors tell you about the device, or vice versa).
USES = 0        # account -> device
USED_BY = 1     # device -> account
PAYS = 2        # account -> merchant
PAID_BY = 3     # merchant -> account
TRANSFERS = 4   # account -> account

RELATION_NAMES = {
    USES: "uses",
    USED_BY: "used_by",
    PAYS: "pays",
    PAID_BY: "paid_by",
    TRANSFERS: "transfers",
}

# The node type found at the far end of each relation: with this fixed,
# every relation's message matrix has a known input dimension.
RELATION_NBR_TYPE = {
    USES: DEVICE,
    USED_BY: ACCOUNT,
    PAYS: MERCHANT,
    PAID_BY: ACCOUNT,
    TRANSFERS: ACCOUNT,
}

RELATIONS = [USES, USED_BY, PAYS, PAID_BY, TRANSFERS]


class HeteroGraph:
    """Graph with typed nodes and typed (directed) edges.

    Node ids are global (``0..n-1``); every node carries exactly one type,
    and each type has its own feature dimension. Labels live only on
    accounts: ``1`` for ring members, ``0`` for benign, ``-1`` for every
    device and merchant.
    """

    def __init__(self, type_dims: dict[int, int]):
        if set(type_dims) != {ACCOUNT, DEVICE, MERCHANT}:
            raise ValueError(
                "type_dims must cover account, device, and merchant types"
            )
        if any(d <= 0 for d in type_dims.values()):
            raise ValueError("feature dims must be positive")
        self.type_dims = dict(type_dims)
        self.node_type: list[int] = []
        self.features: list[list[float]] = []
        self.labels: list[int] = []
        # rel_adj[v][r] = set of neighbor ids of v under relation r.
        self.rel_adj: list[dict[int, set[int]]] = []

    @property
    def n(self) -> int:
        return len(self.node_type)

    # -- construction ----------------------------------------------------

    def add_node(self, node_type: int) -> int:
        """Add a node of `node_type`; returns its global id."""
        if node_type not in self.type_dims:
            raise ValueError(f"unknown node type: {node_type}")
        vid = self.n
        self.node_type.append(node_type)
        self.features.append([0.0] * self.type_dims[node_type])
        self.labels.append(-1)
        self.rel_adj.append({r: set() for r in RELATIONS})
        return vid

    def set_features(self, v: int, features: list[float]) -> None:
        dim = self.type_dims[self.node_type[v]]
        if len(features) != dim:
            raise ValueError(
                f"node {v} is {NODE_TYPE_NAMES[self.node_type[v]]}: "
                f"expected {dim} features, got {len(features)}"
            )
        self.features[v] = [float(x) for x in features]

    def set_label(self, v: int, label: int) -> None:
        if self.node_type[v] != ACCOUNT:
            raise ValueError("only accounts carry labels")
        self.labels[v] = int(label)

    def _add_directed(self, u: int, v: int, r: int) -> None:
        if u == v:
            return
        self.rel_adj[u][r].add(v)

    def add_uses(self, account: int, device: int) -> None:
        """Account logs in from a device (stores USES + USED_BY)."""
        self._add_directed(account, device, USES)
        self._add_directed(device, account, USED_BY)

    def add_pays(self, account: int, merchant: int) -> None:
        """Account pays a merchant (stores PAYS + PAID_BY)."""
        self._add_directed(account, merchant, PAYS)
        self._add_directed(merchant, account, PAID_BY)

    def add_transfer(self, a: int, b: int) -> None:
        """Account-to-account transfer (TRANSFERS, both directions)."""
        self._add_directed(a, b, TRANSFERS)
        self._add_directed(b, a, TRANSFERS)

    # -- queries ----------------------------------------------------------

    def rel_neighbors(self, v: int, r: int) -> list[int]:
        """Sorted neighbor ids of `v` under relation `r` (deterministic)."""
        return sorted(self.rel_adj[v][r])

    def neighbors(self, v: int) -> list[int]:
        """All neighbor ids of `v` across every relation, sorted."""
        out: set[int] = set()
        for r in RELATIONS:
            out.update(self.rel_adj[v][r])
        return sorted(out)

    def degree(self, v: int, r: int | None = None) -> int:
        if r is None:
            return sum(len(s) for s in self.rel_adj[v].values())
        return len(self.rel_adj[v][r])

    def edge_count(self, r: int | None = None) -> int:
        if r is None:
            return sum(
                len(s) for adj in self.rel_adj for s in adj.values()
            )
        return sum(len(adj[r]) for adj in self.rel_adj)

    def account_ids(self) -> list[int]:
        return [v for v in range(self.n) if self.node_type[v] == ACCOUNT]

    def type_counts(self) -> dict[int, int]:
        return {
            t: sum(1 for nt in self.node_type if nt == t)
            for t in (ACCOUNT, DEVICE, MERCHANT)
        }

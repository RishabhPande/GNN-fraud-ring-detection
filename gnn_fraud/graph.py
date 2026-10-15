"""Lightweight graph data structure for GNN experiments.

Nodes are integer ids ``0..n-1``. Edges are undirected, stored as adjacency
sets. Node features are plain lists of floats -- pure stdlib, no numpy.
"""

from __future__ import annotations


class Graph:
    """Undirected graph with float feature vectors on every node."""

    def __init__(self, n: int, feature_dim: int):
        if n <= 0:
            raise ValueError("n must be positive")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.n = n
        self.feature_dim = feature_dim
        self.adj: list[set[int]] = [set() for _ in range(n)]
        self.features: list[list[float]] = [[0.0] * feature_dim for _ in range(n)]
        self.labels: list[int] = [0] * n

    def add_edge(self, u: int, v: int) -> None:
        """Add an undirected edge. Self-loops are ignored."""
        if not (0 <= u < self.n and 0 <= v < self.n):
            raise IndexError(f"node id out of range: {u}, {v}")
        if u == v:
            return
        self.adj[u].add(v)
        self.adj[v].add(u)

    def set_features(self, u: int, features: list[float]) -> None:
        if len(features) != self.feature_dim:
            raise ValueError(
                f"expected {self.feature_dim} features, got {len(features)}"
            )
        self.features[u] = [float(x) for x in features]

    def set_label(self, u: int, label: int) -> None:
        self.labels[u] = int(label)

    def neighbors(self, u: int) -> list[int]:
        """Sorted neighbor ids (sorted for deterministic iteration)."""
        return sorted(self.adj[u])

    def degree(self, u: int) -> int:
        return len(self.adj[u])

    def edge_count(self) -> int:
        return sum(len(s) for s in self.adj) // 2

    def density(self, nodes: list[int]) -> float:
        """Edge density of the induced subgraph on `nodes`."""
        nodes = list(nodes)
        m = len(nodes)
        if m < 2:
            return 0.0
        s = set(nodes)
        e = sum(1 for u in nodes for v in self.adj[u] if v in s) // 2
        return (2.0 * e) / (m * (m - 1))

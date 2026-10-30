"""Plot the quickstart: the planted ring in the graph, and what each model found.

Same seeds and settings as examples/quickstart.py. Left: the synthetic
transaction graph with the planted ring highlighted. Right: validation
metrics for GraphSAGE and the features-only MLP.

Needs numpy and matplotlib (optional, plotting only; the library itself has
zero dependencies): pip install numpy matplotlib

    python examples/plot_ring.py      # writes docs/ring.png

Every account is synthetic.
"""

import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_mlp,
    train_sage,
    train_val_split,
)

BLUE, RED, GREEN, GREY = "#2e86ab", "#e4572e", "#59a14f", "#555555"


def layout(g, ring, iters=200, seed=3):
    """Small deterministic layout so networkx is not needed.

    Ring members sit on a small circle; every other account is placed by a
    simple spring pass (edges attract, all nodes repel) with the ring fixed.
    """
    rng = np.random.default_rng(seed)
    ring = sorted(ring)
    pos = rng.uniform(-1, 1, (g.n, 2))
    ang = np.linspace(0, 2 * np.pi, len(ring), endpoint=False)
    pos[ring] = 0.25 * np.c_[np.cos(ang), np.sin(ang)]
    fixed = np.zeros(g.n, bool)
    fixed[ring] = True
    edges = np.array([(u, v) for u in range(g.n) for v in g.neighbors(u) if u < v])
    for _ in range(iters):
        d = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(d, axis=2) + 1e-3
        disp = (d / dist[..., None] * (0.0012 / dist ** 2)[..., None]).sum(axis=1)
        for u, v in edges:
            dd = pos[v] - pos[u]
            disp[u] += 0.05 * dd
            disp[v] -= 0.05 * dd
        disp[fixed] = 0
        step = np.linalg.norm(disp, axis=1, keepdims=True) + 1e-9
        pos += disp / step * np.minimum(step, 0.05)
        pos = np.clip(pos, -1, 1)
    return pos, edges


def main():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    ring = set(gen.ring_nodes(g))
    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
    sage, _ = train_sage(g, train_ids, val_ids, hidden_dim=16, k1=5, k2=5,
                         lr=0.1, batch_size=64, epochs=60, patience=12, seed=7)
    mlp, _ = train_mlp(g.features, g.labels, train_ids, val_ids, hidden_dim=16,
                       lr=0.3, batch_size=32, epochs=150, patience=20, seed=7)
    y = [g.labels[v] for v in val_ids]
    scores = {
        "GraphSAGE\n(mean agg)": [sage.predict(g, [v])[v] for v in val_ids],
        "MLP\n(features only)": mlp.predict([g.features[v] for v in val_ids]),
    }
    metrics = {}
    for name, s in scores.items():
        _, m = best_threshold_f1(y, s)
        metrics[name] = (m["f1"], evaluate_all(y, s)["auc"])

    pos, edges = layout(g, ring)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.8),
                               gridspec_kw={"width_ratios": [1.5, 1]})
    for u, v in edges:
        inring = u in ring and v in ring
        a.plot(*zip(pos[u], pos[v]), color=RED if inring else "#bbbbbb",
               lw=1.2 if inring else 0.6, alpha=0.9 if inring else 0.6, zorder=1)
    other = [i for i in range(g.n) if i not in ring]
    a.scatter(pos[other, 0], pos[other, 1], s=22, color=BLUE, zorder=2,
              label="ordinary account")
    a.scatter(pos[sorted(ring), 0], pos[sorted(ring), 1], s=40, color=RED,
              zorder=3, label="planted ring member")
    a.set_axis_off()
    a.legend(frameon=True, framealpha=0.95, edgecolor="#dddddd", loc="lower left", fontsize=9)
    a.set_title(f"Synthetic graph: {g.n} accounts, {g.edge_count()} edges, "
                f"ring density {g.density(sorted(ring)):.2f}", loc="left",
                fontsize=10)

    names = list(metrics)
    x = np.arange(len(names))
    w = 0.34
    for j, (lab, col) in enumerate((("F1 (tuned threshold)", BLUE), ("AUC", GREEN))):
        vals = [metrics[n][j] for n in names]
        bars = b.bar(x + (j - 0.5) * w, vals, w, color=col, label=lab)
        for bar, v in zip(bars, vals):
            b.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}",
                   ha="center", fontsize=9, color=GREY)
    b.set_xticks(x)
    b.set_xticklabels(names)
    b.set_ylim(0, 1.15)
    b.grid(axis="y", alpha=0.25)
    b.legend(frameon=False, fontsize=9, loc="upper right")
    b.set_title(f"Held-out nodes (n={len(val_ids)}, "
                f"{sum(y)} ring members)", loc="left", fontsize=10)

    fig.text(0.01, 0.01, "Synthetic accounts, seed 7. One seed and a small "
             "validation set: see the README seed-robustness table. "
             "Reproduce: python examples/plot_ring.py", fontsize=8, color=GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(ROOT, "docs", "ring.png"), dpi=150)
    for n, (f1, auc) in metrics.items():
        print(n.replace("\n", " "), f"F1 {f1:.3f} AUC {auc:.3f}")


if __name__ == "__main__":
    main()

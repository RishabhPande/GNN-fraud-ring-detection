"""Render the quickstart ring-detection result as a two-panel SVG figure.

Left panel: the planted ring (ground truth) inside 200 synthetic accounts.
Right panel: the same layout, nodes shaded by GraphSAGE fraud-score rank
from the quickstart protocol (``examples/quickstart.py``). Rank shading
is deliberate: the raw scores sit in a narrow band (the model ranks well
but is poorly calibrated, val AUC 0.978), so a raw 0-to-1 color scale
would render nearly every node the same pale gray. The point of the pair
is the repo's core claim: the ring's accounts look ordinary one at a
time, and the dense wiring is what the model scores.

Layout is a small Fruchterman-Reingold force-directed placement written
in pure Python (no new dependencies, matching the repo's zero-dependency
rule), seeded so the figure is reproducible. Output goes to
``docs/ring-detection.svg`` and is embedded in the README.
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (  # noqa: E402
    GraphGenerator,
    best_threshold_f1,
    evaluate_all,
    train_sage,
    train_val_split,
)

PANEL_W = 560
PANEL_H = 520
GAP = 40
MARGIN = 30

RING_FILL = "#dc2626"
BENIGN_FILL = "#94a3b8"
EDGE_STROKE = "#cbd5e1"
RING_EDGE_STROKE = "#fca5a5"
LOW_RGB = (226, 232, 240)
HIGH_RGB = (220, 38, 38)


def layout_positions(graph, seed=7, width=PANEL_W, height=PANEL_H,
                     iterations=200):
    """Force-directed layout, deterministic for a given seed.

    Returns a dict of node id -> (x, y) inside the panel box. Ring
    members cluster on their own because the dense pocket pulls them
    together, not because the layout knows the labels.
    """
    rng = random.Random(seed)
    n = graph.n
    pos = {u: (rng.uniform(0, width), rng.uniform(0, height))
           for u in range(n)}
    edges = [(u, v) for u in range(n)
             for v in graph.neighbors(u) if v > u]
    area = width * height
    k = math.sqrt(area / n)
    temp = width / 10.0
    for step in range(iterations):
        disp = {u: [0.0, 0.0] for u in range(n)}
        for u in range(n):
            xu, yu = pos[u]
            for v in range(u + 1, n):
                xv, yv = pos[v]
                dx, dy = xu - xv, yu - yv
                dist = math.hypot(dx, dy) or 0.01
                force = k * k / dist
                ux, uy = dx / dist * force, dy / dist * force
                disp[u][0] += ux
                disp[u][1] += uy
                disp[v][0] -= ux
                disp[v][1] -= uy
        for u, v in edges:
            xu, yu = pos[u]
            xv, yv = pos[v]
            dx, dy = xu - xv, yu - yv
            dist = math.hypot(dx, dy) or 0.01
            force = dist * dist / k
            ux, uy = dx / dist * force, dy / dist * force
            disp[u][0] -= ux
            disp[u][1] -= uy
            disp[v][0] += ux
            disp[v][1] += uy
        cooling = temp * (1.0 - step / iterations)
        for u in range(n):
            dx, dy = disp[u]
            dist = math.hypot(dx, dy) or 0.01
            step_len = min(dist, cooling)
            x = pos[u][0] + dx / dist * step_len
            y = pos[u][1] + dy / dist * step_len
            pos[u] = (min(max(x, MARGIN), width - MARGIN),
                      min(max(y, MARGIN), height - MARGIN))
    return pos


def score_color(score):
    """Interpolate a fraud score in [0, 1] from pale gray to red."""
    t = min(max(score, 0.0), 1.0)
    r = round(LOW_RGB[0] + (HIGH_RGB[0] - LOW_RGB[0]) * t)
    g = round(LOW_RGB[1] + (HIGH_RGB[1] - LOW_RGB[1]) * t)
    b = round(LOW_RGB[2] + (HIGH_RGB[2] - LOW_RGB[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _edges_svg(graph, pos, xoff, yoff, ring_set):
    parts = []
    for u in range(graph.n):
        for v in graph.neighbors(u):
            if v <= u:
                continue
            ring_edge = u in ring_set and v in ring_set
            stroke = RING_EDGE_STROKE if ring_edge else EDGE_STROKE
            w = 1.5 if ring_edge else 1.0
            parts.append(
                f'<line x1="{pos[u][0] + xoff:.1f}" y1="{pos[u][1] + yoff:.1f}"'
                f' x2="{pos[v][0] + xoff:.1f}" y2="{pos[v][1] + yoff:.1f}"'
                f' stroke="{stroke}" stroke-width="{w}" opacity="0.7" />')
    return parts


def _nodes_svg(graph, pos, xoff, yoff, fill_for):
    parts = []
    for u in range(graph.n):
        parts.append(
            f'<circle cx="{pos[u][0] + xoff:.1f}" cy="{pos[u][1] + yoff:.1f}"'
            f' r="5.5" fill="{fill_for(u)}" />')
    return parts


def render_svg(graph, ring_set, pos, scores):
    """Build the two-panel SVG: ground truth left, model scores right.

    Right-panel shading is by score rank (darkest = highest score),
    since the raw scores occupy a narrow band; see the module docstring.
    """
    order = sorted(range(graph.n), key=lambda u: (scores[u], u))
    rank = {u: i for i, u in enumerate(order)}
    denom = max(graph.n - 1, 1)
    ranked_scores = {u: rank[u] / denom for u in range(graph.n)}
    total_w = PANEL_W * 2 + GAP
    total_h = PANEL_H + 110
    left_x, right_x = 0, PANEL_W + GAP
    yoff = 60
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}"'
        f' height="{total_h}" font-family="sans-serif">',
        "<defs><linearGradient id=\"scoregrad\" x1=\"0\" y1=\"0\" x2=\"1\" y2=\"0\">"
        '<stop offset="0%" stop-color="#e2e8f0" />'
        '<stop offset="100%" stop-color="#dc2626" />'
        "</linearGradient></defs>",
        f'<text x="{left_x + 10}" y="28" font-size="17" font-weight="bold">'
        "Planted ring (ground truth)</text>",
        f'<text x="{right_x + 10}" y="28" font-size="17" font-weight="bold">'
        "GraphSAGE fraud score (same layout)</text>",
    ]
    parts += _edges_svg(graph, pos, left_x, yoff, ring_set)
    parts += _edges_svg(graph, pos, right_x, yoff, ring_set)
    parts += _nodes_svg(graph, pos, left_x, yoff,
                        lambda u: RING_FILL if u in ring_set else BENIGN_FILL)
    parts += _nodes_svg(graph, pos, right_x, yoff,
                        lambda u: score_color(ranked_scores[u]))
    legend_y = yoff + PANEL_H + 22
    parts += [
        f'<circle cx="{left_x + 16}" cy="{legend_y - 4}" r="5.5"'
        f' fill="{RING_FILL}" />',
        f'<text x="{left_x + 28}" y="{legend_y}" font-size="13">ring member'
        "</text>",
        f'<circle cx="{left_x + 140}" cy="{legend_y - 4}" r="5.5"'
        f' fill="{BENIGN_FILL}" />',
        f'<text x="{left_x + 152}" y="{legend_y}" font-size="13">benign account'
        "</text>",
        f'<rect x="{right_x + 10}" y="{legend_y - 12}" width="160" height="12"'
        ' fill="url(#scoregrad)" />',
        f'<text x="{right_x + 10}" y="{legend_y + 16}" font-size="13">low score'
        "</text>",
        f'<text x="{right_x + 118}" y="{legend_y + 16}" font-size="13">high score'
        "</text>",
    ]
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main():
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=200, ring_size=18)
    ring = gen.ring_nodes(g)
    ring_set = set(ring)
    print(f"generated {g.n} accounts, {g.edge_count()} edges, "
          f"{len(ring)} planted ring members")

    train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
    model, _ = train_sage(g, train_ids, val_ids, hidden_dim=16,
                          k1=5, k2=5, lr=0.1, batch_size=64,
                          epochs=60, patience=12, seed=7)
    scores = model.predict(g, list(range(g.n)))

    y_val = [g.labels[v] for v in val_ids]
    val_scores = [scores[v] for v in val_ids]
    thr, _ = best_threshold_f1(y_val, val_scores)
    auc = evaluate_all(y_val, val_scores)["auc"]
    flagged_ring = sum(1 for u in ring if scores[u] >= thr)
    flagged_benign = sum(1 for u in range(g.n)
                         if u not in ring_set and scores[u] >= thr)
    print(f"validation AUC {auc:.3f}, F1-tuned threshold {thr:.2f}")
    print(f"at that threshold: {flagged_ring}/{len(ring)} ring members flagged,"
          f" {flagged_benign} benign accounts flagged")

    pos = layout_positions(g)
    svg = render_svg(g, ring_set, pos, scores)
    out_dir = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "docs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ring-detection.svg")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"wrote {out_path} ({len(svg)} bytes)")


if __name__ == "__main__":
    main()

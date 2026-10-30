"""Tests for the demo visualization: deterministic layout, score-color
scale endpoints, and the two-panel SVG structure."""

import importlib.util
import os

from gnn_fraud import Graph, GraphGenerator

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "examples")


def _load_visualize():
    spec = importlib.util.spec_from_file_location(
        "visualize_ring",
        os.path.join(EXAMPLES_DIR, "visualize_ring.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _small_graph():
    g = Graph(6, 2)
    for u, v in [(0, 1), (1, 2), (2, 0), (3, 4)]:
        g.add_edge(u, v)
    return g


def test_layout_is_deterministic():
    viz = _load_visualize()
    g = _small_graph()
    first = viz.layout_positions(g, iterations=20)
    second = viz.layout_positions(g, iterations=20)
    assert first == second
    assert set(first) == set(range(6))
    for x, y in first.values():
        assert 0 <= x <= viz.PANEL_W
        assert 0 <= y <= viz.PANEL_H


def test_score_color_endpoints_and_midpoint():
    viz = _load_visualize()
    assert viz.score_color(0.0) == "#e2e8f0"
    assert viz.score_color(1.0) == "#dc2626"
    assert viz.score_color(0.5) == "#df878b"
    assert viz.score_color(-1.0) == "#e2e8f0"
    assert viz.score_color(2.0) == "#dc2626"


def test_render_svg_has_two_panels_and_all_nodes():
    viz = _load_visualize()
    gen = GraphGenerator(seed=7)
    g = gen.generate(n=30, ring_size=5)
    ring_set = set(gen.ring_nodes(g))
    pos = viz.layout_positions(g, iterations=10)
    scores = {u: (0.9 if u in ring_set else 0.1 + 0.01 * u)
              for u in range(g.n)}
    svg = viz.render_svg(g, ring_set, pos, scores)
    assert "Planted ring (ground truth)" in svg
    assert "GraphSAGE fraud score (same layout)" in svg
    assert svg.count("<circle") == 2 * g.n + 2  # nodes x2 panels + legend
    # Left panel colors every ring member solid red; the right panel's
    # top-ranked node also lands on solid red.
    assert svg.count('fill="#dc2626"') >= len(ring_set) + 1

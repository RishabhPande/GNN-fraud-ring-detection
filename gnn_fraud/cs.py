"""Correct and Smooth (C&S, Huang et al., 2020), from scratch.

Every other model in this repo propagates features, hidden
states, or predictions that a network produced. C&S propagates
something else: what the base model got wrong on the accounts
whose labels are already known, and then the corrected scores
themselves. It is a post-processing step, not a network - it
takes any base model's scores (an MLP, a GNN, a rule engine)
plus the confirmed labels and returns better scores, in two
stages::

    Correct:  e_v = y_v - p_v on labeled nodes, 0 elsewhere
              E <- alpha_c * S E + (1 - alpha_c) * E_0   (steps_c rounds)
              p'_v = clip(p_v + E_v, 0, 1)
    Smooth:   z_0 = p' with labeled nodes reset to y_v
              Z <- alpha_s * S Z + (1 - alpha_s) * Z_0   (steps_s rounds)

where ``S`` is the row-normalized adjacency with a self-loop,
``S[v][u] = 1 / (deg(v) + 1)`` for ``u`` in ``N(v) U {v}`` -
the same sparse rows :mod:`gnn_fraud.linkx` reads as input,
here used as a propagation operator. Each stage is a damped
diffusion: the ``(1 - alpha)`` term keeps pulling the field
back to its starting values, so a labeled node's own residual
(or label) never washes out completely. At the extremes the
stages degrade plainly: with ``alpha = 0`` or ``steps = 0``
nothing propagates, so Correct sets each labeled node exactly
to its label and leaves every other score alone, and Smooth
returns just the anchored field (labels reset, the rest
untouched).

Why it belongs here: fraud teams always hold a set of
confirmed cases (chargebacks, investigator verdicts) that a
scoring model was not necessarily trained on. The Correct
stage says a model that under-scored a confirmed fraudster
probably under-scored that account's counterparties too, by
an amount that decays with graph distance. The Smooth stage
then spreads the corrected scores the way APPNP spreads
predictions, except the anchors are hard labels, not the
model's own output, so the smoothing cannot drift away from
the confirmed cases. The two stages are deliberately separate
functions (``correct_scores``, ``smooth_scores``) so a caller
can run either one alone, and ``correct_and_smooth`` is just
their composition.

Two honesty notes. First, the labeled nodes must be training
or otherwise legitimately known labels: feeding validation
labels into C&S and then reporting validation AUC would be
label leakage, the exact failure :mod:`gnn_fraud.inductive`
warns about, so the example and the tests only ever pass
training labels. Second, C&S cannot create signal: an
unlabeled isolated node keeps its base score exactly (its
row of ``S`` is the one-hot on itself, so nothing reaches it
and its own residual is 0), so an account with no
counterparties is untouched by either stage.
All arithmetic is hand-rolled lists-of-floats - no numpy,
no torch.
"""

from __future__ import annotations


def _validate(scores, known_labels):
    if len(scores) == 0:
        raise ValueError("scores must be non-empty")
    for v, p in enumerate(scores):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"score for node {v} outside [0, 1]: {p}")
    for v, y in known_labels.items():
        if not (0 <= v < len(scores)):
            raise IndexError(f"labeled node id out of range: {v}")
        if y not in (0, 1):
            raise ValueError(f"label for node {v} must be 0 or 1, got {y}")


def _validate_stage(alpha, steps):
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
    if steps < 0:
        raise ValueError("steps must be non-negative")


def _propagate(graph, field, alpha, steps):
    """Damped diffusion of ``field`` under the row-normalized
    adjacency with a self-loop: ``steps`` rounds of
    ``F <- alpha * S F + (1 - alpha) * F_0``. Deterministic,
    and the identity when ``alpha == 0`` or ``steps == 0``."""
    out = list(field)
    for _ in range(steps):
        nxt = []
        for v in range(graph.n):
            w = 1.0 / (graph.degree(v) + 1)
            acc = w * out[v]
            for u in graph.neighbors(v):
                acc += w * out[u]
            nxt.append(alpha * acc + (1.0 - alpha) * field[v])
        out = nxt
    return out


def correct_scores(graph, scores, known_labels,
                   alpha: float = 0.8, steps: int = 20) -> list[float]:
    """The Correct stage: propagate the base model's residuals
    on the labeled nodes and add the diffused residual field
    back onto every score, clipped to [0, 1].

    A labeled node keeps at least ``(1 - alpha)`` of its own
    residual at every round, so a confirmed fraudster the base
    model scored at 0.2 is corrected upward by close to 0.8,
    its direct counterparties by roughly ``alpha / (deg + 1)``
    of that, and distant accounts by geometrically less.
    Unlabeled nodes start with residual 0 - the stage spreads
    known errors, it does not invent new ones.
    """
    _validate(scores, known_labels)
    _validate_stage(alpha, steps)
    if len(scores) != graph.n:
        raise ValueError(
            f"expected {graph.n} scores, got {len(scores)}")
    residual = [0.0] * graph.n
    for v, y in known_labels.items():
        residual[v] = float(y) - scores[v]
    spread = _propagate(graph, residual, alpha, steps)
    return [min(max(scores[v] + spread[v], 0.0), 1.0)
            for v in range(graph.n)]


def smooth_scores(graph, scores, known_labels,
                  alpha: float = 0.8, steps: int = 20) -> list[float]:
    """The Smooth stage: reset the labeled nodes to their true
    labels, then run the same damped diffusion over the scores.

    The reset is what separates this from APPNP-style prediction
    smoothing: the anchors are ground truth, and the damping
    term pulls every round back toward the reset field, so a
    labeled node's final score stays close to its label while
    its unlabeled counterparties are pulled toward it.
    """
    _validate(scores, known_labels)
    _validate_stage(alpha, steps)
    if len(scores) != graph.n:
        raise ValueError(
            f"expected {graph.n} scores, got {len(scores)}")
    anchored = list(scores)
    for v, y in known_labels.items():
        anchored[v] = float(y)
    return _propagate(graph, anchored, alpha, steps)


def correct_and_smooth(graph, scores, known_labels,
                       alpha_correct: float = 0.8, steps_correct: int = 20,
                       alpha_smooth: float = 0.8,
                       steps_smooth: int = 20) -> list[float]:
    """Both stages in order: correct the base scores with the
    diffused residuals, then smooth the corrected scores with
    the labeled nodes anchored at their true labels. Returns a
    new list of scores in [0, 1], one per node."""
    corrected = correct_scores(graph, scores, known_labels,
                               alpha=alpha_correct, steps=steps_correct)
    return smooth_scores(graph, corrected, known_labels,
                         alpha=alpha_smooth, steps=steps_smooth)

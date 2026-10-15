"""Precision / recall / F1 and ROC-AUC, computed from scratch."""

from __future__ import annotations


def precision_recall_f1(y_true: list[int], y_score: list[float],
                        threshold: float = 0.5) -> dict[str, float]:
    """Binary metrics at a fixed decision threshold."""
    tp = fp = fn = 0
    for y, s in zip(y_true, y_score):
        pred = 1 if s >= threshold else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 1:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def roc_auc(y_true: list[int], y_score: list[float]) -> float:
    """ROC-AUC via the Mann-Whitney U statistic, with tie correction.

    AUC = P(score(positive) > score(negative)) + 0.5 * P(tie).
    """
    pos = [s for y, s in zip(y_true, y_score) if y == 1]
    neg = [s for y, s in zip(y_true, y_score) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = ties = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                ties += 1.0
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def evaluate_all(y_true: list[int], y_score: list[float],
                 threshold: float = 0.5) -> dict[str, float]:
    """All metrics in one dict."""
    out = precision_recall_f1(y_true, y_score, threshold)
    out["auc"] = roc_auc(y_true, y_score)
    return out


def best_threshold_f1(y_true: list[int], y_score: list[float]
                      ) -> tuple[float, dict[str, float]]:
    """Threshold maximizing F1, scanned over candidate cutoffs.

    Fraud scores are rarely well-calibrated at 0.5 under class imbalance,
    so production teams tune the operating point. Returns
    (threshold, metrics-at-threshold).
    """
    candidates = sorted(set([0.5] + list(y_score)))
    best_thr, best_m = 0.5, None
    best_f1 = -1.0
    for thr in candidates:
        m = precision_recall_f1(y_true, y_score, thr)
        if m["f1"] > best_f1:
            best_f1, best_thr, best_m = m["f1"], thr, m
    return best_thr, best_m

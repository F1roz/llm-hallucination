"""Evaluation metrics and statistical tests.

- AUROC with a stratified bootstrap 95% interval.
- Paired bootstrap interval for an AUROC difference.
- DeLong test for two correlated AUROCs (DeLong et al., 1988), using the
  fast midrank algorithm of Sun and Xu (2014).
- Holm step-down correction (Holm, 1979).
- AURAC, the area under the rejection accuracy curve (Farquhar et al., 2024).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


@dataclass
class Interval:
    value: float
    low: float
    high: float


def impute(scores: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace missing scores with the median, so they rank in the middle."""
    scores = np.asarray(scores, dtype=float)
    missing = np.isnan(scores)
    if missing.all():
        return np.full_like(scores, 0.5), int(missing.sum())
    filled = scores.copy()
    filled[missing] = np.nanmedian(scores)
    return filled, int(missing.sum())


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def bootstrap_auroc(labels: np.ndarray, scores: np.ndarray, n_boot: int, seed: int) -> Interval:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    point = auroc(labels, scores)
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    if len(pos) == 0 or len(neg) == 0:
        return Interval(point, float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        values[b] = roc_auc_score(labels[idx], scores[idx])
    low, high = np.percentile(values, [2.5, 97.5])
    return Interval(point, float(low), float(high))


def bootstrap_auroc_difference(labels: np.ndarray, first: np.ndarray, second: np.ndarray, n_boot: int,
                               seed: int) -> Interval:
    labels = np.asarray(labels, dtype=int)
    point = auroc(labels, first) - auroc(labels, second)
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    if len(pos) == 0 or len(neg) == 0:
        return Interval(point, float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        values[b] = roc_auc_score(labels[idx], first[idx]) - roc_auc_score(labels[idx], second[idx])
    low, high = np.percentile(values, [2.5, 97.5])
    return Interval(point, float(low), float(high))


def delong_test(labels: np.ndarray, first: np.ndarray, second: np.ndarray) -> tuple[float, float, float, float]:
    """Two sided DeLong test. Returns (auc_first, auc_second, z, p)."""
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(-labels, kind="mergesort")
    n_pos = int(labels.sum())
    if n_pos < 2 or len(labels) - n_pos < 2:  # the variance needs two items per class
        nan = float("nan")
        return auroc(labels, first), auroc(labels, second), nan, nan
    preds = np.vstack([np.asarray(first, float), np.asarray(second, float)])[:, order]
    aucs, cov = _fast_delong(preds, n_pos)
    diff = aucs[0] - aucs[1]
    variance = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if variance <= 0:
        return float(aucs[0]), float(aucs[1]), 0.0, 1.0
    z = diff / np.sqrt(variance)
    p = 2 * norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


def holm(p_values: list[float]) -> list[float]:
    """Holm adjusted p-values in the original order. Missing p-values stay missing and are not counted."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(values), np.nan)
    finite = np.flatnonzero(~np.isnan(values))
    m = len(finite)
    running = 0.0
    for rank, index in enumerate(finite[np.argsort(values[finite])]):
        running = max(running, min(1.0, (m - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def aurac(correct: np.ndarray, scores: np.ndarray) -> float:
    """Mean accuracy over every retention size, keeping the lowest scored answers first."""
    correct = np.asarray(correct, dtype=float)
    order = np.argsort(scores, kind="mergesort")
    cumulative = np.cumsum(correct[order])
    retained = np.arange(1, len(correct) + 1)
    return float(np.mean(cumulative / retained))


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def _fast_delong(preds: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    k = preds.shape[0]
    n_neg = preds.shape[1] - n_pos
    pos, neg = preds[:, :n_pos], preds[:, n_pos:]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds[r]) for r in range(k)])
    aucs = tz[:, :n_pos].sum(axis=1) / n_pos / n_neg - (n_pos + 1.0) / 2.0 / n_neg
    v01 = (tz[:, :n_pos] - tx) / n_neg
    v10 = 1.0 - (tz[:, n_pos:] - ty) / n_pos
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / n_pos + sy / n_neg

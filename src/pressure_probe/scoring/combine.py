"""Combining pressure and cross-family signals.

The main combination needs no training: the mean of percentile ranks. A logistic
regression with out-of-fold predictions is reported only as a reference ceiling.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def rank_mean(*columns: np.ndarray) -> np.ndarray:
    ranks = [rankdata(np.asarray(c, dtype=float)) / len(c) for c in columns]
    return np.mean(ranks, axis=0)


def logistic_out_of_fold(features: np.ndarray, labels: np.ndarray, folds: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    minority = min(int(labels.sum()), int(len(labels) - labels.sum()))
    folds = min(folds, minority)
    if folds < 2:
        return np.full(len(labels), np.nan)
    out = np.empty(len(labels))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, test in splitter.split(features, labels):
        model = LogisticRegression(max_iter=1000)
        model.fit(features[train], labels[train])
        out[test] = model.predict_proba(features[test])[:, 1]
    return out

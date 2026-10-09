"""Evaluation metrics for all three tasks.

Sign convention: `risk_score` follows src/losses.py -- higher = higher risk =
shorter expected survival. `concordance_index()` below negates it before
calling lifelines, because lifelines' `concordance_index(times, scores,
event)` expects the OPPOSITE convention (higher score = longer survival).
This was verified empirically (not assumed) with a synthetic script during
development: passing true survival time directly gives C=1.0, while passing
`-true_survival_time` (i.e. a "risk"-like score) gives C=0.0 on the same
data.
"""
from __future__ import annotations

import numpy as np
from lifelines.utils import concordance_index as _lifelines_concordance_index
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def concordance_index(follow_up_years: np.ndarray, risk_score: np.ndarray, event: np.ndarray) -> float:
    """Harrell's C-index under the risk_score convention used throughout
    this project (higher risk_score = higher risk = shorter survival).
    """
    return float(_lifelines_concordance_index(follow_up_years, -np.asarray(risk_score), event))


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity_recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
    }
    # specificity = recall of the negative class
    metrics["specificity"] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

    if len(np.unique(y_true)) < 2:
        metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        spearman = float(spearmanr(y_true, y_pred).correlation)
        pearson = float(pearsonr(y_true, y_pred)[0])
    else:
        spearman = pearson = float("nan")

    return {
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "r2": r2,
        "spearman": spearman,
        "pearson": pearson,
    }


def survival_metrics(follow_up_years: np.ndarray, risk_score: np.ndarray, event: np.ndarray) -> dict:
    """C-index is undefined when a split has zero admissible pairs (e.g. no
    observed events at all, which can happen on a small validation split).
    lifelines raises ZeroDivisionError in that case; we report NaN instead
    of crashing the run, since this is an expected small-sample edge case,
    not a bug -- callers must not silently treat NaN as a real 0/1 score.
    """
    try:
        c_index = concordance_index(follow_up_years, risk_score, event)
    except ZeroDivisionError:
        c_index = float("nan")
    return {"c_index": c_index}

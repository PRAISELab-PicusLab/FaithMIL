#!/usr/bin/env python3
"""Aggregate per-fold results into mean +/- std per task/backbone, plus a
pooled out-of-fold metric (ROC-AUC for classification, C-index for
survival) computed from all subjects' concatenated held-out predictions --
important on a dataset this small, where per-fold metrics are noisy.

Usage:
    python scripts/summarize_results.py [--results-dir results] [--n-folds 10]

Writes results/comparison.csv with columns:
    task, backbone, metric, mean, std, n_folds_available, pooled
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import concordance_index  # noqa: E402
from src.models import BACKBONE_NAMES  # noqa: E402

TASKS = ["classification", "regression", "survival"]


def load_fold_metrics(results_dir: Path, task: str, backbone: str, fold: int) -> dict | None:
    path = results_dir / task / backbone / f"fold_{fold}" / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_fold_predictions(results_dir: Path, task: str, backbone: str, fold: int) -> pd.DataFrame | None:
    path = results_dir / task / backbone / f"fold_{fold}" / "test_predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def pooled_metric(task: str, all_predictions: pd.DataFrame) -> dict:
    if task == "classification":
        if all_predictions["event"].nunique() < 2:
            return {}
        return {"pooled_roc_auc": float(roc_auc_score(all_predictions["event"], all_predictions["probability"]))}
    if task == "survival":
        return {
            "pooled_c_index": concordance_index(
                all_predictions["follow_up_years"], all_predictions["risk_score"], all_predictions["event"]
            )
        }
    return {}


def summarize(results_dir: Path, n_folds: int) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        for backbone in BACKBONE_NAMES:
            fold_metrics = []
            fold_preds = []
            for fold in range(n_folds):
                m = load_fold_metrics(results_dir, task, backbone, fold)
                p = load_fold_predictions(results_dir, task, backbone, fold)
                if m is not None:
                    fold_metrics.append(m)
                if p is not None:
                    fold_preds.append(p)

            if not fold_metrics:
                continue

            metric_names = sorted({k for m in fold_metrics for k in m.keys()})
            for name in metric_names:
                vals = [m[name] for m in fold_metrics if name in m and not (isinstance(m[name], float) and np.isnan(m[name]))]
                if not vals:
                    continue
                rows.append(
                    {
                        "task": task,
                        "backbone": backbone,
                        "metric": name,
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "n_folds_available": len(vals),
                        "pooled": np.nan,
                    }
                )

            if fold_preds:
                all_preds = pd.concat(fold_preds, ignore_index=True)
                for name, val in pooled_metric(task, all_preds).items():
                    rows.append(
                        {
                            "task": task,
                            "backbone": backbone,
                            "metric": name,
                            "mean": np.nan,
                            "std": np.nan,
                            "n_folds_available": len(fold_preds),
                            "pooled": val,
                        }
                    )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output = Path(args.output) if args.output else results_dir / "comparison.csv"

    df = summarize(results_dir, args.n_folds)
    if df.empty:
        print(f"No completed runs found under {results_dir}. Nothing to summarize yet.")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote {len(df)} rows to {output}\n")

    for task in TASKS:
        sub = df[df["task"] == task]
        if sub.empty:
            continue
        print(f"=== {task} ===")
        print(sub.pivot_table(index=["backbone", "metric"], values=["mean", "std", "pooled"]).to_string())
        print()


if __name__ == "__main__":
    main()

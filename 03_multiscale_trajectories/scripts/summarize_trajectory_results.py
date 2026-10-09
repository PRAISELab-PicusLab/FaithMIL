#!/usr/bin/env python3
"""Compare trajectory-MIL runs: per-fold mean/std and pooled out-of-fold C-index.

Usage:
    python scripts/summarize_trajectory_results.py [--results-dir results/trajectory]

Writes <results-dir>/comparison.csv (one row per run) and prints it sorted by
pooled C-index, next to the thumbnail baseline for reference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import survival_metrics  # noqa: E402


def summarize_run(run_dir: Path) -> dict | None:
    split_dirs = sorted(run_dir.glob("fold_*")) or ([run_dir / "holdout"] if (run_dir / "holdout").exists() else [])
    metrics, preds, cfg = [], [], None
    for d in split_dirs:
        if not (d / "metrics.json").exists():
            continue
        with open(d / "metrics.json") as f:
            metrics.append(json.load(f))
        preds.append(pd.read_csv(d / "test_predictions.csv"))
        if cfg is None:
            with open(d / "config.yaml") as f:
                cfg = yaml.safe_load(f)
    if not metrics:
        return None
    m = pd.DataFrame(metrics)
    p = pd.concat(preds, ignore_index=True)
    pooled = survival_metrics(p["follow_up_years"].to_numpy(), p["risk_score"].to_numpy(), p["event"].to_numpy())
    return {
        "run": run_dir.name,
        "mode": cfg["split"]["mode"],
        "levels": "".join(map(str, cfg["data"]["levels"])),
        "backbone": cfg["data"]["backbone"],
        "level_backbones": cfg["data"].get("level_backbones") or "",
        "aggregator": cfg["model"]["aggregator"],
        "pooling": cfg["model"]["pooling"],
        "n_splits": len(m),
        "n_test_cases": len(p),
        "pooled_c_index": pooled["c_index"],
        "test_c_index_mean": m["test_c_index"].mean(),
        "test_c_index_std": m["test_c_index"].std(ddof=1) if len(m) > 1 else np.nan,
        "val_c_index_mean": m["val_c_index"].mean(),
        "train_c_index_mean": m["train_c_index"].mean(),
        "best_epoch_mean": m["best_epoch"].mean(),
        "attention_eff_n_median": p["attention_eff_n"].median() if "attention_eff_n" in p else np.nan,
        "only_available_n_cases": cfg.get("only_available_n_cases", ""),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results/trajectory")
    args = ap.parse_args()
    root = Path(args.results_dir)
    rows = [r for d in sorted(root.iterdir()) if d.is_dir() and (r := summarize_run(d))]
    if not rows:
        sys.exit(f"no finished runs under {root}")
    df = pd.DataFrame(rows).sort_values("pooled_c_index", ascending=False)
    df.to_csv(root / "comparison.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(df.drop(columns=["level_backbones", "only_available_n_cases"]).to_string(index=False))
    base = Path("results/headline_comparison.csv")
    if base.exists():
        b = pd.read_csv(base)
        print("\nthumbnail baseline (pooled C-index):", b[b["task"] == "survival"].iloc[0, 2:].to_dict())
    print(f"\nwritten: {root / 'comparison.csv'}")


if __name__ == "__main__":
    main()

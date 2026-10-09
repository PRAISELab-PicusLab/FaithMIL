#!/usr/bin/env python3
"""Backbone comparison table for the survival task, with uncertainty.

For every backbone under each results directory:
  * per-fold test C-index -> mean, std, and a t-based 95% CI over the 10
    folds (each fold's test set is disjoint, so the 10 values are the
    primary evidence);
  * one-sample t-test of the fold C-indices against 0.5 (chance);
  * pooled out-of-fold C-index over all 508 patients with a 95% bootstrap CI
    (patients resampled). Caveat: pooled risk scores come from 10 different
    models; for the linear probe each fold has its own lambda/standardization,
    so the pooled value is indicative only.

Also reports Holm-adjusted p-values across all backbones compared, since
with ~17 comparisons one "significant" result at alpha=0.05 is expected by
chance alone.

Usage:
    python scripts/compare_backbones.py \
        --results-dirs results_probe:frozen_linear_probe results:finetuned \
        --output results_probe/backbone_comparison.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import concordance_index  # noqa: E402


def holm(pvals: np.ndarray) -> np.ndarray:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def summarize_backbone(bdir: Path, n_boot: int, rng: np.random.Generator) -> dict | None:
    fold_files = sorted(bdir.glob("fold_*/test_predictions.csv"))
    if len(fold_files) == 0:
        return None
    per_fold, parts = [], []
    for f in fold_files:
        d = pd.read_csv(f)
        parts.append(d)
        try:
            per_fold.append(concordance_index(d.follow_up_years, d.risk_score, d.event))
        except ZeroDivisionError:
            per_fold.append(np.nan)
    c = np.array(per_fold, dtype=float)
    c = c[np.isfinite(c)]
    n = len(c)
    mean, sd = c.mean(), c.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    t_p = stats.ttest_1samp(c, 0.5).pvalue

    pooled = pd.concat(parts, ignore_index=True)
    t, r, e = pooled.follow_up_years.to_numpy(), pooled.risk_score.to_numpy(), pooled.event.to_numpy()
    c_pool = concordance_index(t, r, e)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(pooled), len(pooled))
        if e[idx].sum() == 0:
            continue
        boots.append(concordance_index(t[idx], r[idx], e[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])

    return {
        "n_folds": n,
        "c_index_mean": mean,
        "c_index_std": sd,
        "c_index_ci95_low": mean - half,
        "c_index_ci95_high": mean + half,
        "p_vs_chance": t_p,
        "pooled_c_index": c_pool,
        "pooled_ci95_low": lo,
        "pooled_ci95_high": hi,
        "n_patients": len(pooled),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dirs", nargs="+", default=["results_probe:frozen_linear_probe", "results:finetuned"],
                   help="dir:label pairs")
    p.add_argument("--task", default="survival")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--output", default="results_probe/backbone_comparison.csv")
    p.add_argument("--include", default=None,
                   help="regex: only backbones whose name matches are compared (and Holm-corrected)")
    args = p.parse_args()

    rng = np.random.default_rng(42)
    rows = []
    for spec in args.results_dirs:
        rdir, label = spec.split(":", 1)
        task_dir = Path(rdir) / args.task
        pat = re.compile(args.include) if args.include else None
        for bdir in sorted(d for d in task_dir.iterdir() if d.is_dir()):
            if pat is not None and not pat.search(bdir.name):
                continue
            s = summarize_backbone(bdir, args.n_boot, rng)
            if s is not None:
                rows.append({"protocol": label, "backbone": bdir.name, **s})

    df = pd.DataFrame(rows)
    df["p_vs_chance_holm"] = holm(df["p_vs_chance"].to_numpy())
    df = df.sort_values("c_index_mean", ascending=False).reset_index(drop=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    show = df[["protocol", "backbone", "c_index_mean", "c_index_std", "c_index_ci95_low", "c_index_ci95_high",
               "p_vs_chance", "p_vs_chance_holm", "pooled_c_index", "pooled_ci95_low", "pooled_ci95_high"]]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(show.round(3).to_string(index=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()

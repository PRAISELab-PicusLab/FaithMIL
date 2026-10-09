#!/usr/bin/env python3
"""Quick EDA on splits/folds.csv: checks that the 10-fold stratification
(event x thumbnail_level) actually produced balanced folds, and that
follow_up_years isn't accidentally skewed across folds (not stratified on,
so worth checking it didn't end up lopsided by chance).

Usage: python scripts/eda_folds.py [--folds-csv splits/folds.csv]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, kruskal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds-csv", default="splits/folds.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.folds_csv)
    n_folds = df["fold"].nunique()

    print(f"Loaded {len(df)} subjects, {n_folds} folds\n")

    print("=== Fold sizes ===")
    print(df["fold"].value_counts().sort_index().to_string())

    print("\n=== Event count / rate per fold ===")
    ev = df.groupby("fold")["event"].agg(["sum", "count", "mean"])
    ev.columns = ["n_events", "n_total", "event_rate"]
    print(ev.to_string())
    print(f"\noverall event rate: {df['event'].mean():.4f}")
    print(f"event_rate range across folds: {ev['event_rate'].min():.4f} - {ev['event_rate'].max():.4f}  (std={ev['event_rate'].std():.4f})")

    print("\n=== Thumbnail level composition per fold ===")
    level_ct = pd.crosstab(df["fold"], df["thumbnail_level"])
    print(level_ct.to_string())
    level_frac = level_ct.div(level_ct.sum(axis=1), axis=0)
    print("\nfraction L5 per fold:")
    print(level_frac["L5"].to_string())

    print("\n=== Joint event x level per fold (checking the actual stratification target) ===")
    joint = df.groupby(["fold", "event", "thumbnail_level"]).size().unstack(fill_value=0)
    print(joint.to_string())

    print("\n=== follow_up_years per fold (NOT stratified on -- sanity check only) ===")
    fu = df.groupby("fold")["follow_up_years"].agg(["mean", "median", "std", "min", "max"])
    print(fu.to_string())

    print("\n=== follow_up_years per fold, split by event (more relevant clinically) ===")
    fu_by_event = df.groupby(["fold", "event"])["follow_up_years"].agg(["mean", "median", "count"])
    print(fu_by_event.to_string())

    # --- Formal homogeneity checks ---
    print("\n=== Statistical homogeneity checks across folds ===")

    # event distribution homogeneous across folds?
    event_ct = pd.crosstab(df["fold"], df["event"])
    chi2, p, dof, _ = chi2_contingency(event_ct)
    print(f"Chi-square test, event x fold independence: chi2={chi2:.3f}, dof={dof}, p={p:.4f}"
          f" ({'OK: no evidence of imbalance' if p > 0.05 else 'WARNING: folds differ in event rate more than chance'})")

    # level distribution homogeneous across folds?
    chi2, p, dof, _ = chi2_contingency(level_ct)
    print(f"Chi-square test, level x fold independence:  chi2={chi2:.3f}, dof={dof}, p={p:.4f}"
          f" ({'OK: no evidence of imbalance' if p > 0.05 else 'WARNING: folds differ in level composition more than chance'})")

    # follow_up_years homogeneous across folds? (not stratified on, so check it's not a fluke)
    groups = [g["follow_up_years"].values for _, g in df.groupby("fold")]
    f_stat, p_anova = f_oneway(*groups)
    h_stat, p_kw = kruskal(*groups)
    print(f"ANOVA, follow_up_years across folds:    F={f_stat:.3f}, p={p_anova:.4f}"
          f" ({'OK' if p_anova > 0.05 else 'WARNING: means differ across folds more than chance'})")
    print(f"Kruskal-Wallis, follow_up_years across folds: H={h_stat:.3f}, p={p_kw:.4f}"
          f" ({'OK' if p_kw > 0.05 else 'WARNING: distributions differ across folds more than chance'})")

    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Contingency table + chi-square test between n_levels_wsi and event.

n_levels_wsi (5 vs 6) is an acquisition artifact, not a biological signal: any model
trained on the nativeLast thumbnails (which mix two different native scales, 62.09 and
248.68 mpp, depending on this count) could learn to exploit it instead of real tissue
signal if it correlates with the recurrence label. This script quantifies that risk.

Run after the full batch (needs thumbnails/nativeLast/nativeLast_metadata.csv).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency

DEFAULT_METADATA_CSV = "/leonardo_scratch/large/userexternal/vmoscato/wacv_challenge/thumbnails/nativeLast/nativeLast_metadata.csv"
DEFAULT_LABELS_CSV = "/leonardo_scratch/large/userexternal/vmoscato/wacv_challenge/training_labels.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV, type=Path)
    parser.add_argument("--labels-csv", default=DEFAULT_LABELS_CSV, type=Path)
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata_csv)
    labels = pd.read_csv(args.labels_csv)
    labels["case_id"] = labels["case_id"].str.replace(".tif", "", regex=False)

    df = meta.merge(labels[["case_id", "event"]], on="case_id", how="inner")
    if len(df) != len(meta):
        print(f"WARNING: {len(meta) - len(df)} cases in metadata had no matching label row")

    table = pd.crosstab(df["n_levels_wsi"], df["event"])
    chi2, p, dof, expected = chi2_contingency(table)

    print("Contingency table (n_levels_wsi x event):")
    print(table)
    print()
    print("Row proportions (event rate by n_levels_wsi):")
    print(table.div(table.sum(axis=1), axis=0))
    print()
    print(f"chi2 = {chi2:.4f}, dof = {dof}, p-value = {p:.4f}")
    if p < 0.05:
        print("=> SIGNIFICANT association at alpha=0.05: n_levels_wsi correlates with event.")
        print("   nativeLast thumbnails carry a scale confound with the label; models trained")
        print("   on that folder should not be trusted without controlling for n_levels_wsi.")
    else:
        print("=> No significant association detected at alpha=0.05.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

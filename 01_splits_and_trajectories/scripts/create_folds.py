#!/usr/bin/env python3
"""Build ONE fixed 10-fold CV split, reused for every task/backbone/fold.

Joins training_labels.csv (case_id, event, follow_up_years) with the
thumbnail metadata CSV (which records level_used per case) purely by reading
these files -- no assumed column names or paths beyond the CLI defaults,
which themselves point at files already inspected on disk.

Stratifies jointly by event x thumbnail_level (discovering whatever levels
are actually present, not assuming only L4/L5). Falls back to stratifying by
event alone if any joint stratum has fewer than `n_splits` members, and
reports which path was taken.

Output: splits/folds.csv with columns
    case_id, fold, event, follow_up_years, thumbnail_level, thumbnail_path
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def build_folds(
    labels_csv: Path,
    thumbnail_metadata_csv: Path,
    thumbnail_dir: Path,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    meta = pd.read_csv(thumbnail_metadata_csv)

    # training_labels.csv's case_id carries the original ".tif" filename;
    # the thumbnail metadata's case_id is the bare case id. Strip rather
    # than assume, so this keeps working if that ever changes.
    labels = labels.copy()
    labels["case_id"] = labels["case_id"].astype(str).str.replace(r"\.tif$", "", regex=True)

    dup = labels["case_id"].duplicated()
    if dup.any():
        raise ValueError(f"Duplicate case_id in {labels_csv}: {labels.loc[dup, 'case_id'].tolist()}")
    dup_meta = meta["case_id"].duplicated()
    if dup_meta.any():
        raise ValueError(f"Duplicate case_id in {thumbnail_metadata_csv}: {meta.loc[dup_meta, 'case_id'].tolist()}")

    df = labels.merge(meta[["case_id", "level_used"]], on="case_id", how="outer", indicator=True)
    only_labels = df[df["_merge"] == "left_only"]
    only_meta = df[df["_merge"] == "right_only"]
    if len(only_labels) or len(only_meta):
        raise ValueError(
            f"case_id mismatch between labels and thumbnail metadata: "
            f"{len(only_labels)} only in labels, {len(only_meta)} only in metadata.\n"
            f"Examples missing thumbnails: {only_labels['case_id'].tolist()[:10]}\n"
            f"Examples missing labels: {only_meta['case_id'].tolist()[:10]}"
        )
    df = df.drop(columns="_merge")

    df["thumbnail_level"] = "L" + df["level_used"].astype(int).astype(str)
    df["thumbnail_path"] = df["case_id"].apply(lambda c: str(thumbnail_dir / f"{c}.png"))

    missing_files = [p for p in df["thumbnail_path"] if not Path(p).exists()]
    if missing_files:
        raise FileNotFoundError(f"{len(missing_files)} thumbnail files listed in metadata do not exist on disk, e.g.: {missing_files[:5]}")

    print(f"Number of subjects: {len(df)}")
    print("\nDistribution of event:")
    print(df["event"].value_counts().sort_index().to_string())
    print("\nDistribution of thumbnail_level:")
    print(df["thumbnail_level"].value_counts().sort_index().to_string())
    print("\nJoint distribution event x thumbnail_level:")
    joint = pd.crosstab(df["event"], df["thumbnail_level"])
    print(joint.to_string())

    df["stratum"] = df["event"].astype(str) + "_" + df["thumbnail_level"]
    stratum_counts = df["stratum"].value_counts()
    min_stratum = stratum_counts.min()
    print(f"\nSmallest joint stratum size: {min_stratum} (need >= {n_splits} for {n_splits}-fold stratification)")

    if min_stratum >= n_splits:
        strat_col = "stratum"
        fallback_used = False
        print(f"=> Using JOINT stratification on event x thumbnail_level ({n_splits}-fold).")
    else:
        strat_col = "event"
        fallback_used = True
        print(
            f"=> WARNING: joint event x thumbnail_level stratification is NOT possible "
            f"(smallest stratum has {min_stratum} < {n_splits} samples). "
            f"Falling back to stratifying by event only."
        )
        event_counts = df["event"].value_counts()
        if event_counts.min() < n_splits:
            raise ValueError(
                f"Even event-only stratification is impossible: smallest event class has "
                f"{event_counts.min()} samples, need >= {n_splits}."
            )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold_idx, (_, test_idx) in enumerate(skf.split(df, df[strat_col])):
        df.loc[df.index[test_idx], "fold"] = fold_idx

    assert (df["fold"] >= 0).all()

    print("\nFold size distribution:")
    print(df["fold"].value_counts().sort_index().to_string())
    print("\nEvent rate per fold:")
    print(df.groupby("fold")["event"].mean().to_string())
    print("\nThumbnail-level composition per fold:")
    print(pd.crosstab(df["fold"], df["thumbnail_level"]).to_string())

    df["stratification_fallback_used"] = fallback_used
    return df[
        [
            "case_id",
            "fold",
            "event",
            "follow_up_years",
            "thumbnail_level",
            "thumbnail_path",
            "stratification_fallback_used",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", default="training_labels.csv")
    parser.add_argument("--thumbnail-metadata-csv", default="thumbnails/nativeLast/nativeLast_metadata.csv")
    parser.add_argument("--thumbnail-dir", default="thumbnails/nativeLast")
    parser.add_argument("--output", default="splits/folds.csv")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing folds.csv")
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"{output_path} already exists. Folds must not be silently regenerated "
            f"(every run/model must reuse the exact same splits). Pass --force to override."
        )

    np.random.seed(args.seed)
    df = build_folds(
        labels_csv=Path(args.labels_csv),
        thumbnail_metadata_csv=Path(args.thumbnail_metadata_csv),
        thumbnail_dir=Path(args.thumbnail_dir),
        n_splits=args.n_splits,
        seed=args.seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()

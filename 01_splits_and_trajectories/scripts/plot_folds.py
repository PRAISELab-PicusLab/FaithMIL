#!/usr/bin/env python3
"""Static PNG versions of the fold-balance diagnostics (same content as the
interactive dashboard) for archiving / slides, since PNGs don't need a
browser to view on the cluster.

Usage: python scripts/plot_folds.py [--folds-csv splits/folds.csv] [--output-dir splits/fold_eda_plots]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR_L4 = "#2a78d6"
COLOR_L5 = "#eb6834"


def plot_event_rate(df: pd.DataFrame, ax) -> None:
    ev = df.groupby("fold")["event"].mean()
    overall = df["event"].mean()
    ax.bar(ev.index, ev.values, color=COLOR_L4, width=0.6)
    ax.axhline(overall, color="0.4", linestyle="--", linewidth=1.2, label=f"overall = {overall:.1%}")
    for f, v in ev.items():
        ax.text(f, v + 0.004, f"{v:.1%}", ha="center", va="bottom", fontsize=8, color="0.3")
    ax.set_xlabel("fold")
    ax.set_ylabel("event rate")
    ax.set_title("Event rate per fold")
    ax.set_xticks(ev.index)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def plot_level_composition(df: pd.DataFrame, ax) -> None:
    ct = pd.crosstab(df["fold"], df["thumbnail_level"]).reindex(columns=["L4", "L5"], fill_value=0)
    ax.bar(ct.index, ct["L4"], color=COLOR_L4, width=0.6, label="L4")
    ax.bar(ct.index, ct["L5"], bottom=ct["L4"], color=COLOR_L5, width=0.6, label="L5")
    ax.set_xlabel("fold")
    ax.set_ylabel("count")
    ax.set_title("Thumbnail level composition per fold")
    ax.set_xticks(ct.index)
    ax.legend(frameon=False, loc="upper right", fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)


def plot_followup_boxplot(df: pd.DataFrame, ax) -> None:
    folds = sorted(df["fold"].unique())
    data = [df.loc[df["fold"] == f, "follow_up_years"].values for f in folds]
    bp = ax.boxplot(data, positions=folds, widths=0.5, patch_artist=True, showmeans=False)
    for box in bp["boxes"]:
        box.set(facecolor=COLOR_L4, alpha=0.15, edgecolor=COLOR_L4, linewidth=1.5)
    for med in bp["medians"]:
        med.set(color=COLOR_L4, linewidth=2)
    for whisk in bp["whiskers"] + bp["caps"]:
        whisk.set(color=COLOR_L4, linewidth=1.3)
    for flier in bp["fliers"]:
        flier.set(marker="o", markersize=3, markerfacecolor=COLOR_L4, markeredgecolor="none", alpha=0.6)
    ax.set_xlabel("fold")
    ax.set_ylabel("follow_up_years")
    ax.set_title("Follow-up years distribution per fold")
    ax.set_xticks(folds)
    ax.spines[["top", "right"]].set_visible(False)


def plot_joint_heatmap(df: pd.DataFrame, ax) -> None:
    folds = sorted(df["fold"].unique())
    cols = [(0, "L4"), (0, "L5"), (1, "L4"), (1, "L5")]
    col_labels = ["event=0\nL4", "event=0\nL5", "event=1\nL4", "event=1\nL5"]
    joint = df.groupby(["fold", "event", "thumbnail_level"]).size()

    mat = np.zeros((len(folds), len(cols)))
    for i, f in enumerate(folds):
        for j, (ev, lvl) in enumerate(cols):
            mat[i, j] = joint.get((f, ev, lvl), 0)

    # column-wise normalization, matching the interactive dashboard
    norm = (mat - mat.min(axis=0, keepdims=True)) / np.clip(
        mat.max(axis=0, keepdims=True) - mat.min(axis=0, keepdims=True), 1, None
    )

    im = ax.imshow(norm, cmap="Blues", aspect="auto", vmin=0, vmax=1.3)
    for i in range(len(folds)):
        for j in range(len(cols)):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=9, color="0.15")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(folds)))
    ax.set_yticklabels(folds)
    ax.set_ylabel("fold")
    ax.set_title("Joint event x level counts per fold")
    im.set_visible(False)  # only used for the color mapping; cells drawn below
    for i in range(len(folds)):
        for j in range(len(cols)):
            ax.add_patch(
                plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=plt.cm.Blues(0.15 + 0.55 * norm[i, j]), edgecolor="white", linewidth=1)
            )
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(folds) - 0.5, -0.5)
    for i in range(len(folds)):
        for j in range(len(cols)):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=9, color="0.15")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds-csv", default="splits/folds.csv")
    parser.add_argument("--output-dir", default="splits/fold_eda_plots")
    args = parser.parse_args()

    df = pd.read_csv(args.folds_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    specs = [
        ("event_rate_per_fold.png", plot_event_rate, (7, 4)),
        ("level_composition_per_fold.png", plot_level_composition, (7, 4)),
        ("followup_boxplot_per_fold.png", plot_followup_boxplot, (7, 4)),
        ("joint_event_level_heatmap.png", plot_joint_heatmap, (6, 6)),
    ]
    for filename, fn, figsize in specs:
        fig, ax = plt.subplots(figsize=figsize)
        fn(df, ax)
        fig.tight_layout()
        path = out_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {path}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    plot_event_rate(df, axes[0, 0])
    plot_level_composition(df, axes[0, 1])
    plot_followup_boxplot(df, axes[1, 0])
    plot_joint_heatmap(df, axes[1, 1])
    fig.suptitle("Fold stratification check (event x thumbnail_level, seed=42)", fontsize=13)
    fig.tight_layout()
    combined_path = out_dir / "fold_eda_summary.png"
    fig.savefig(combined_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {combined_path}")


if __name__ == "__main__":
    main()

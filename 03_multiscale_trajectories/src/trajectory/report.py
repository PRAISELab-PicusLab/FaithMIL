"""Per-case interpretability report (PDF, <= 5 pages, matplotlib only).

  1. Prediction   time to recurrence, risk group, risk percentile, survival curve
  2. Where        signed Shapley map + reliability (attention) map on the slide
  3. Risk-increasing evidence   top trajectories as zoom strips (coarse -> fine)
  4. Protective evidence        same, for the most negative contributions
  5. Which scale  level occlusion, concentration of the attribution, caveats

Everything written in the report is computed from the model's outputs; no
morphological description is generated automatically, to avoid asserting
biology the model did not demonstrably use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402

from src.trajectory.survival import BreslowModel  # noqa: E402
from src.trajectory.wsi import TiledPyramid, level_fov_um  # noqa: E402

A4 = (8.27, 11.69)
SCALE_NOTES = {
    5: "whole-slide context",
    4: "large region / slide sector",
    3: "regional tissue architecture",
    2: "glandular architecture (Gleason-pattern scale)",
    1: "glands and cell groups",
    0: "cellular / nuclear detail",
}


@dataclass
class CaseExplanation:
    case_id: str
    risk: float
    baseline: float
    predicted_time: float
    risk_group: str
    anchors: pd.DataFrame          # anchor_id, x, y, gx, gy
    scores: np.ndarray
    weights: np.ndarray
    shapley: np.ndarray
    occlusion: dict                # level -> delta risk
    levels: list                   # coarse -> fine, levels used by the model
    breslow: BreslowModel
    train_risks: np.ndarray
    risk_group_thresholds: list
    patch_size: int
    stride_l0: int
    n_train_cases: int | None = None
    n_train_events: int | None = None
    run_info: dict = field(default_factory=dict)


def _fmt_fov(um: float) -> str:
    return f"{um / 1000:.1f} mm" if um >= 1000 else f"{um:.0f} µm"


def _text_page(fig, y, lines, size=10, x=0.07, line_h=0.022):
    for ln in lines:
        weight = "bold" if ln.startswith("## ") else "normal"
        fig.text(x, y, ln.removeprefix("## "), fontsize=size + (2 if weight == "bold" else 0), weight=weight,
                 va="top", wrap=True)
        y -= line_h * (1.4 if weight == "bold" else 1.0)
    return y


def _overlay(ax, thumb, ex: CaseExplanation, values, cmap, vmin, vmax, title, ds_thumb=256):
    ax.imshow(thumb)
    half = ex.stride_l0 / 2 / ds_thumb
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    rects = [Rectangle((x / ds_thumb - half, y / ds_thumb - half), 2 * half, 2 * half)
             for x, y in zip(ex.anchors["x"], ex.anchors["y"])]
    pc = PatchCollection(rects, cmap=cmap, alpha=0.55, linewidths=0)
    pc.set_array(np.asarray(values))
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    return pc


def write_report(ex: CaseExplanation, wsi_path, out_path: Path | str, top_k: int = 4) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    phi = np.asarray(ex.shapley)
    order_pos = np.argsort(-phi)[:top_k]
    order_neg = np.argsort(phi)[:top_k]
    pct = 100.0 * (np.asarray(ex.train_risks) < ex.risk).mean()
    abs_sorted = np.sort(np.abs(phi))[::-1]
    mass = np.cumsum(abs_sorted) / max(abs_sorted.sum(), 1e-12)
    top10_mass = float(mass[min(9, len(mass) - 1)]) if len(mass) else 0.0
    eff_n = 1.0 / float(np.sum(np.asarray(ex.weights) ** 2))

    with TiledPyramid(wsi_path) as pyr:
        H4, W4 = pyr.shapes[4]
        thumb = pyr.read_region(4, 0, 0, W4, H4)
        mpp = pyr.mpp_l0
        strips = {}
        for idx in np.concatenate([order_pos, order_neg]):
            row = ex.anchors.iloc[int(idx)]
            strips[int(idx)] = pyr.read_trajectory(float(row["x"]), float(row["y"]), ex.levels, ex.patch_size)

    with PdfPages(out_path) as pdf:
        # ---- page 1: prediction ------------------------------------------------
        fig = plt.figure(figsize=A4)
        lines = [
            f"## OWL — BCR prediction report: {ex.case_id}",
            "",
            f"Predicted time to biochemical recurrence: {ex.predicted_time:.2f} years",
            f"   (restricted mean BCR-free time over {ex.breslow.horizon:.1f} years)",
            f"Risk group: {ex.risk_group.upper()}   (groups = tertiles of predicted time in the training cohort)",
            f"Risk score: {ex.risk:+.3f}   ->  higher than {pct:.0f}% of training patients",
            "",
            "## How the prediction is made",
            "The slide is covered by a regular grid of anchor points on tissue. For each anchor the model",
            "looks at a 'trajectory' of patches centred on the same point, zooming in from coarse to fine:",
        ]
        for l in ex.levels:
            lines.append(f"   L{l}: {_fmt_fov(level_fov_um(l, ex.patch_size, mpp))} field of view — {SCALE_NOTES[l]}")
        lines += [
            f"Each of the {len(ex.anchors)} trajectories receives a local risk score and a reliability weight;",
            "the patient risk is the reliability-weighted average of the local scores. Contributions below",
            "are Shapley values: they sum exactly to (patient risk - average training risk)",
            f"= {ex.risk - ex.baseline:+.3f}; red = pushes risk up, blue = protective.",
        ]
        if ex.run_info:
            lines += ["", "Model: " + ", ".join(f"{k}={v}" for k, v in ex.run_info.items())]
        _text_page(fig, 0.96, lines)

        ax = fig.add_axes([0.1, 0.08, 0.38, 0.26])
        t = np.linspace(0, ex.breslow.horizon, 300)
        qs = np.quantile(ex.train_risks, [0.1, 0.5, 0.9])
        for q, lab, ls in zip(qs, ["10th pct", "median", "90th pct"], [":", "--", ":"]):
            ax.plot(t, ex.breslow.survival_at(q, t)[0], color="grey", ls=ls, lw=1, label=f"training {lab}")
        ax.plot(t, ex.breslow.survival_at(ex.risk, t)[0], color="C3", lw=2, label="this patient")
        ax.set_xlabel("years after prostatectomy")
        ax.set_ylabel("P(BCR-free)")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7)
        ax.set_title("Predicted BCR-free survival", fontsize=10)

        ax = fig.add_axes([0.58, 0.08, 0.36, 0.26])
        train_times = ex.breslow.rmst(ex.train_risks)
        ax.hist(train_times, bins=30, color="lightgrey")
        for thr in ex.risk_group_thresholds:
            ax.axvline(thr, color="k", ls="--", lw=0.8)
        ax.axvline(ex.predicted_time, color="C3", lw=2)
        ax.set_xlabel("predicted time to BCR (years)")
        ax.set_title("Training cohort (dashed = group thresholds)", fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 2: where ------------------------------------------------------
        fig = plt.figure(figsize=A4)
        fig.text(0.07, 0.97, "Where: spatial attribution", fontsize=13, weight="bold", va="top")
        vmax = float(np.percentile(np.abs(phi), 99)) or 1.0
        ax = fig.add_axes([0.04, 0.50, 0.78, 0.42])  # leaves room for the colorbar
        pc = _overlay(ax, thumb, ex, phi, "RdBu_r", -vmax, vmax,
                      "Shapley contribution (red = risk-increasing, blue = protective)")
        fig.colorbar(pc, ax=ax, fraction=0.03, format=FormatStrFormatter("%.1e"))
        for rank, idx in enumerate(order_pos, 1):
            r = ex.anchors.iloc[int(idx)]
            ax.annotate(f"+{rank}", (r["x"] / 256, r["y"] / 256), color="darkred", fontsize=8, weight="bold")
        for rank, idx in enumerate(order_neg, 1):
            r = ex.anchors.iloc[int(idx)]
            ax.annotate(f"-{rank}", (r["x"] / 256, r["y"] / 256), color="navy", fontsize=8, weight="bold")
        ax = fig.add_axes([0.04, 0.08, 0.78, 0.36])
        w = np.asarray(ex.weights)
        pc = _overlay(ax, thumb, ex, w / w.max(), "viridis", 0, 1,
                      f"Reliability weight (relative); effective n. of trajectories = {eff_n:.0f}")
        fig.colorbar(pc, ax=ax, fraction=0.03)
        fig.text(0.07, 0.045, f"The 10 most influential trajectories carry {100 * top10_mass:.0f}% of the total absolute "
                 f"attribution (of {len(phi)} analysed).", fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)

        # ---- pages 3-4: evidence strips ----------------------------------------
        for title, order, sign in [("Risk-increasing evidence", order_pos, "+"), ("Protective evidence", order_neg, "-")]:
            fig = plt.figure(figsize=A4)
            fig.text(0.07, 0.97, title, fontsize=13, weight="bold", va="top")
            fig.text(0.07, 0.945, "Each row is one trajectory, zooming in left to right; the red box marks the next, "
                     "finer patch.", fontsize=9, va="top")
            n_l = len(ex.levels)
            for r, idx in enumerate(order):
                row = ex.anchors.iloc[int(idx)]
                y0 = 0.72 - r * 0.22
                fig.text(0.07, y0 + 0.19, f"{sign}{r + 1}: Shapley {phi[idx]:+.4f} | local score {ex.scores[idx]:+.3f} | "
                         f"reliability {ex.weights[idx] * len(phi):.2f}x uniform | anchor ({int(row['x'])}, {int(row['y'])}) L0 px",
                         fontsize=8.5)
                for i, l in enumerate(ex.levels):
                    ax = fig.add_axes([0.07 + i * (0.86 / n_l), y0, 0.86 / n_l - 0.01, 0.17])
                    ax.imshow(strips[int(idx)][i])
                    if i + 1 < n_l:
                        f = 4 ** (l - ex.levels[i + 1])
                        side = ex.patch_size / f
                        c = ex.patch_size / 2
                        ax.add_patch(plt.Rectangle((c - side / 2, c - side / 2), side, side, fill=False, ec="red", lw=1))
                    ax.set_title(f"L{l} · {_fmt_fov(level_fov_um(l, ex.patch_size, mpp))}", fontsize=8)
                    ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)

        # ---- page 5: which scale -----------------------------------------------
        fig = plt.figure(figsize=A4)
        fig.text(0.07, 0.97, "Which scale drives the prediction", fontsize=13, weight="bold", va="top")
        ax = fig.add_axes([0.1, 0.62, 0.38, 0.28])
        lv = list(ex.occlusion)
        dv = [ex.occlusion[l] for l in lv]
        ax.bar([f"L{l}" for l in lv], dv, color=["C3" if v > 0 else "C0" for v in dv])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("risk change when the level is neutralised")
        ax.set_title("Level occlusion", fontsize=10)

        ax = fig.add_axes([0.58, 0.62, 0.36, 0.28])
        ax.plot(np.arange(1, len(mass) + 1), mass, color="k")
        ax.set_xscale("log")
        ax.set_xlabel("top-k trajectories (by |Shapley|)")
        ax.set_ylabel("fraction of total |attribution|")
        ax.set_title("Concentration of the evidence", fontsize=10)

        ax = fig.add_axes([0.1, 0.30, 0.38, 0.24])
        sc = ax.scatter(ex.scores, np.asarray(ex.weights) * len(phi), c=phi, cmap="RdBu_r", vmin=-vmax, vmax=vmax, s=5)
        ax.set_yscale("log")
        ax.set_xlabel("local risk score")
        ax.set_ylabel("reliability (x uniform)")
        ax.set_title("Local score vs reliability", fontsize=10)
        fig.colorbar(sc, ax=ax, fraction=0.04, format=FormatStrFormatter("%.1e"))

        top_level = max(ex.occlusion, key=lambda l: abs(ex.occlusion[l])) if ex.occlusion else None
        lines = ["## Reading this page"]
        if top_level is not None:
            lines.append(f"Neutralising L{top_level} ({SCALE_NOTES[top_level]}) changes the risk the most "
                         f"({ex.occlusion[top_level]:+.3f}).")
        lines += [
            f"{100 * top10_mass:.0f}% of the attribution comes from the top 10 of {len(phi)} trajectories "
            f"({'focal' if top10_mass > 0.5 else 'diffuse'} evidence).",
            "",
            "## Caveats",
            "Attributions describe what THIS MODEL relies on, not established biology; they are",
            "computed by removing trajectories (Shapley) or neutralising a scale (occlusion).",
            "Morphological interpretation of the highlighted regions is left to the pathologist.",
            (f"The model was trained on {ex.n_train_cases} prostatectomy slides ({ex.n_train_events} recurrences) "
             "with patient-level outcomes only.") if ex.n_train_cases else
            "The model was trained with patient-level outcomes only.",
        ]
        _text_page(fig, 0.22, lines, size=9, line_h=0.019)
        pdf.savefig(fig)
        plt.close(fig)

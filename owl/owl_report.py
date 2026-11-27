"""Per-patient pathology report of the OWL container (PDF, 3 pages, A4).

  page 1  the prediction: time to biochemical recurrence, risk class, where the patient sits in the OWL training
          cohort, and the two maps (signed tile contributions, and the relevance map submitted as heatmap)
  page 2  the regions that raise and lower the predicted risk most, at the resolution the model reads (0.22 mm tiles)
  page 3  how to read the report: what the model measured, what is interpretation, and its limits

Everything on pages 1-2 is measured from the model (the contributions add up exactly to the risk); page 3 is the only
interpretive text and never refers to grading, which the model does not use.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from owl_model import rank01
from trace_mil.layout import PATCH
from trace_mil.wsi import TiledPyramid, patch_top_left

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import TwoSlopeNorm, rgb_to_hsv  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

A4 = (8.27, 11.69)
INK, MUTED, RISK_UP, RISK_DOWN = "#1b1b1b", "#5a5a5a", "#c0392b", "#2463a6"
CLASS_COLOUR = {"low": "#2e8b57", "medium": "#d4a017", "high": "#c0392b"}


def thumbnail(rd: TiledPyramid, max_side: int = 2000):
    """RGB thumbnail of the coarsest native level whose long side is still >= 600 px, and its L0 downsample."""
    levels = [k for k in range(rd.n_levels) if k <= rd.max_native_level]
    k = next((k for k in levels if max(rd.shapes[k]) <= max_side), levels[-1])
    while k > 0 and max(rd.shapes[k]) < 600:
        k -= 1
    H, W = rd.shapes[k]
    return rd.read_region(k, 0, 0, W, H), 4 ** k


def shows_tissue(rgb: np.ndarray) -> bool:
    """True if at least 25% of the tile is H&E-stained (HSV saturation >= 0.07, not near-black, hue of eosin or
    haematoxylin: >= 210 or < 20 deg) and at most 12% of the non-background pixels are dark (V < 0.35). Marker ink
    fails: green-blue ink by its hue, black ink by its dark pixels (H&E tissue, dense lymphocytes included, stays below
    ~7%)."""
    hsv = rgb_to_hsv(rgb.astype(np.float32) / 255)
    hue, sat, val = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    stained = (sat >= 0.07) & (val > 0.25) & ((hue >= 210) | (hue < 20))
    dark = val < 0.35
    return stained.mean() >= 0.25 and dark.sum() <= 0.12 * max((dark | (sat >= 0.07)).sum(), 1)


def _paint(shape, x, y, values, side_l0, ds, fill=np.nan):
    out = np.full(shape, fill, np.float32)
    half = side_l0 / 2
    for xi, yi, v in zip(x, y, values):
        x0, x1 = int((xi - half) // ds), int(np.ceil((xi + half) / ds))
        y0, y1 = int((yi - half) // ds), int(np.ceil((yi + half) / ds))
        out[max(y0, 0):y1, max(x0, 0):x1] = v
    return out


def _km(time, event, grid):
    """Kaplan-Meier survival on `grid`."""
    t, e = np.asarray(time), np.asarray(event).astype(bool)
    times = np.unique(t[e])
    surv = np.cumprod([1 - ((t == u) & e).sum() / (t >= u).sum() for u in times])
    i = np.searchsorted(times, grid, side="right") - 1
    return np.where(i < 0, 1.0, surv[np.clip(i, 0, None)] if len(surv) else 1.0)


def _text(ax, x, y, s, **kw):
    ax.text(x, y, s, transform=ax.transAxes, va="top", color=kw.pop("color", INK), **kw)


def _box_off(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def make_report(path: Path, wsi_path: Path, children: pd.DataFrame, phi: np.ndarray, relevance_score: np.ndarray,
                z: float, years: float, risk_class: str, cal: dict, cohort: pd.DataFrame, tile_side_l0: int,
                mpp_l0: float, relevance_rule: str, n_top: int = 6) -> None:
    rd = TiledPyramid(wsi_path, max_native_level=4, tile_cache_tiles=64, strict=False)
    thumb, ds = thumbnail(rd)
    if len(children) == 0:  # no tissue tile: the prediction is the cohort median
        with PdfPages(path) as pdf:
            fig = plt.figure(figsize=A4)
            fig.text(0.07, 0.95, "FaithMIL  -  Biochemical recurrence after radical prostatectomy", fontsize=14, weight="bold")
            fig.text(0.07, 0.91, "The tissue mask contains no analysable tissue tile. The reported values "
                                 f"({years:.1f} years, {risk_class}) are the training-cohort median, not a prediction.",
                     fontsize=9, wrap=True)
            ax = fig.add_axes([0.1, 0.3, 0.8, 0.55])
            ax.imshow(thumb)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
        rd.close()
        return
    x, y = children["x"].to_numpy(), children["y"].to_numpy()
    phi_map = _paint(thumb.shape[:2], x, y, phi, tile_side_l0, ds)
    rel = rank01(relevance_score)  # as the submitted heatmap: only the order matters
    rel_map = _paint(thumb.shape[:2], x, y, rel, tile_side_l0, ds)

    pct = float((cohort.z < z).mean() * 100)
    max_fu = cal["max_follow_up_years"]
    n_tiles = len(children)
    area_mm2 = n_tiles * (tile_side_l0 * mpp_l0 / 1000) ** 2
    pos, neg = phi[phi > 0].sum(), -phi[phi < 0].sum()

    def tile_rgb(i):
        tx, ty = int(patch_top_left(x[i], 1, PATCH)), int(patch_top_left(y[i], 1, PATCH))
        return rd.read_region(1, tx, ty, PATCH, PATCH)

    def top_tissue_tiles(order, sign):
        """The first n_top tiles of `order` on the `sign` side that show tissue (the mask is coarse: some kept tiles
        are mostly background or ink); returns (tiles, number skipped)."""
        shown, skipped = [], 0
        for i in order[:100 * n_top]:
            if len(shown) == n_top or np.sign(phi[i]) != sign:
                break
            rgb = tile_rgb(i)
            if not shows_tissue(rgb):
                skipped += 1
                continue
            shown.append((i, rgb))
        return shown, skipped

    top_up, skip_up = top_tissue_tiles(np.argsort(-phi, kind="stable"), 1)
    top_down, skip_down = top_tissue_tiles(np.argsort(phi, kind="stable"), -1)
    k5 = max(1, int(np.ceil(0.05 * n_tiles)))
    conc = float(np.sort(np.abs(phi))[::-1][:k5].sum() / max(np.abs(phi).sum(), 1e-12))
    weak = abs(pct - 50) < 10

    with PdfPages(path) as pdf:
        # ------------------------------------------------------------------ page 1
        fig = plt.figure(figsize=A4)
        gs = fig.add_gridspec(4, 2, height_ratios=[1.05, 1.5, 1.5, 1.2], hspace=0.35, wspace=0.15,
                              left=0.07, right=0.93, top=0.95, bottom=0.05)
        ax = fig.add_subplot(gs[0, :])
        _box_off(ax)
        _text(ax, 0, 1.0, "FaithMIL  -  Biochemical recurrence after radical prostatectomy", fontsize=14, weight="bold")
        _text(ax, 0, 0.84, f"Slide {wsi_path.stem}  |  H&E, provided tissue mask  |  research use only, not a clinical "
                           "device", fontsize=7.5, color=MUTED)
        _text(ax, 0, 0.66, "Predicted median time to recurrence", fontsize=9, color=MUTED)
        _text(ax, 0, 0.56, f"{years:.1f} years", fontsize=12, weight="bold")
        if years > max_fu:
            _text(ax, 0.32, 0.545, f"beyond the longest follow-up of the training cohort ({max_fu:.0f} y): read as 'long'",
                  fontsize=8.5, color=MUTED)
        _text(ax, 0, 0.36, "Risk class", fontsize=9, color=MUTED)
        _text(ax, 0, 0.26, risk_class.upper(), fontsize=13, weight="bold", color=CLASS_COLOUR[risk_class])
        _text(ax, 0.32, 0.36, "Risk relative to the OWL training cohort", fontsize=9, color=MUTED)
        _text(ax, 0.32, 0.26, f"higher than {pct:.0f}% of the {len(cohort)} training patients (risk score z = {z:+.2f})",
              fontsize=10)
        evidence = (f"Evidence: {n_tiles:,} tissue tiles of 0.22 mm ({area_mm2:.0f} mm² of tissue). "
                    f"The top 5% of the tiles carry {conc:.0%} of the total contribution."
                    + (" The prediction is close to the cohort median: weak evidence for either direction." if weak else ""))
        _text(ax, 0, 0.12, "\n".join(textwrap.wrap(evidence, 125)), fontsize=8.5, linespacing=1.3)

        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(thumb)
        ax.set_title("H&E", fontsize=9)
        ax.axis("off")
        # the map and its colour bar share the cell; the bar is then fitted to the height of the drawn image
        sub = gs[1, 1].subgridspec(1, 2, width_ratios=[1, 0.035], wspace=0.04)
        ax, cax = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        ax.imshow(thumb, alpha=0.35)
        v = max(float(np.quantile(np.abs(phi), 0.99)), 1e-9)  # colour scale saturates at the 99th percentile
        im = ax.imshow(np.clip(phi_map, -v, v), cmap="RdBu_r", norm=TwoSlopeNorm(0, -v, v), interpolation="nearest")
        ax.set_title("Tile contributions to the risk: red raises, blue lowers\n(scale saturated at the 99th "
                     "percentile; boxes: top tiles)", fontsize=8.5)
        ax.axis("off")
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=6)
        cb.ax.yaxis.get_offset_text().set_fontsize(6)
        ax.apply_aspect()
        p, q = ax.get_position(), cax.get_position()
        cax.set_position([p.x1 + 0.01, p.y0, q.width, p.height])
        for i, _ in top_up:
            ax.add_patch(Rectangle(((x[i] - tile_side_l0 / 2) / ds, (y[i] - tile_side_l0 / 2) / ds), tile_side_l0 / ds,
                                   tile_side_l0 / ds, fill=False, ec="black", lw=0.8))
        ax = fig.add_subplot(gs[2, 0])
        ax.imshow(thumb, alpha=0.35)
        ax.imshow(rel_map, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title("Relevance map (submitted heatmap, by rank): removing\nbright regions changes the prediction most",
                     fontsize=8.5)
        ax.axis("off")

        ax = fig.add_subplot(gs[2, 1])
        grid = np.linspace(0, 15, 61)
        for g in ("low", "medium", "high"):
            c = cohort[cohort.risk_class == g]
            ax.step(grid, 1 - _km(c.follow_up_years, c.event, grid), where="post", color=CLASS_COLOUR[g],
                    lw=2.2 if g == risk_class else 1.0, label=f"{g} (n={len(c)})")
        ax.set_xlabel("years after surgery", fontsize=8)
        ax.set_ylabel("cumulative recurrence", fontsize=8)
        ax.set_title("Recurrence in the training cohort by risk class\n(out-of-fold predictions)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

        ax = fig.add_subplot(gs[3, :])
        ax.hist(cohort.z, bins=40, color="#c9c9c9")
        ax.axvline(z, color=CLASS_COLOUR[risk_class], lw=2)
        ax.set_title("This patient within the OWL training cohort", fontsize=9)
        ax.set_ylabel("patients", fontsize=8)
        ax.set_xlabel("risk score z of the training patients (this patient: coloured line)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        pdf.savefig(fig)
        plt.close(fig)

        # ------------------------------------------------------------------ page 2
        fig = plt.figure(figsize=A4)
        fig.text(0.07, 0.965, "Regions that drive the prediction", fontsize=13, weight="bold")
        sub = (f"Tiles of 0.22 mm as the model reads them. Contribution = the exact share of the tile in the risk score "
               f"z = {z:+.2f} (all tiles together: +{pos:.2f} raising, -{neg:.2f} lowering).")
        fig.text(0.07, 0.95, "\n".join(textwrap.wrap(sub, 120)), fontsize=8, color=MUTED, va="top")
        rows = [("Raise the predicted risk", top_up, skip_up, RISK_UP),
                ("Lower the predicted risk", top_down, skip_down, RISK_DOWN)]
        for r, (title, idx, skipped, colour) in enumerate(rows):
            top = 0.91 - r * 0.455
            fig.text(0.07, top, title, fontsize=11, weight="bold", color=colour)
            if skipped:
                fig.text(0.40, top, f"({skipped} higher-ranked tile{'s' if skipped > 1 else ''} with little tissue, "
                                    "background or ink, not shown)", fontsize=7.5, color=MUTED)
            for j, (i, rgb) in enumerate(idx):
                ax = fig.add_axes([0.07 + (j % 3) * 0.30, top - 0.035 - (j // 3 + 1) * 0.2, 0.26, 0.18])
                ax.imshow(rgb)
                _box_off(ax)
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_color(colour)
                    sp.set_linewidth(2)
                ax.set_title(f"{phi[i]:+.2g}  ({100 * abs(phi[i]) / (pos if phi[i] > 0 else neg):.2g}% of that side)",
                             fontsize=7.5)
            if not len(idx):
                fig.text(0.07, top - 0.04, "no tile on this side", fontsize=9, color=MUTED)
        pdf.savefig(fig)
        plt.close(fig)

        # ------------------------------------------------------------------ page 3
        fig = plt.figure(figsize=A4)
        sections = [
            ("What the model measured", [
                f"The slide was divided into {n_tiles:,} tissue tiles of 0.22 mm (224 px at about 1 µm/px), taken only "
                "where the provided tissue mask has tissue. Each tile was encoded by a DINO vision transformer trained "
                "from scratch on prostatectomy H&E tiles, without labels.",
                "An ensemble of 30 attention-based MIL models (10 cross-validation folds x 3 seeds), trained on the OWL "
                "training cohort and TCGA-PRAD with a Cox loss on time to biochemical recurrence, gives each tile an "
                "attention weight and a score. The risk score z is the sum over the tiles of weight x score, standardised "
                "per model, so the tile contributions on page 2 add up exactly to z.",
                "The relevance map (the submitted heatmap) ranks the tiles by how much removing them changes the "
                f"prediction, computed exactly for this model (rule: {relevance_rule.replace('_', ' ')}).",
            ]),
            ("From the risk score to years and class", [
                "Years are the median time to recurrence of a Weibull model fitted on the out-of-fold risk scores of the "
                "508 OWL training patients; they decrease monotonically with the risk. Values beyond the follow-up of the "
                "cohort are an extrapolation and only mean 'long'.",
                "The risk class splits the training patients into thirds of predicted years (low / medium / high). In the "
                "training cohort the share of patients with recurrence was 10%, 15% and 26% in the three classes.",
            ]),
            ("Interpretation (not measured)", [
                "The tiles on page 2 are the regions the model relies on; their morphology is for the pathologist to judge. "
                "The model was not given, and does not predict, any histological grade or pattern: its output depends only "
                "on the tissue appearance learnt from outcome data.",
                "Contributions are associations learnt from the training cohorts, not causal statements about the tissue.",
            ]),
            ("Limits", [
                "Discrimination in cross-validation on the OWL training cohort: C-index 0.67. A single H&E slide is used; "
                "clinical variables (PSA, stage, margins) are not.",
                "Scanner, staining or tissue very different from the training cohorts can make the prediction unreliable. "
                "Areas excluded by the tissue mask are not seen by the model.",
            ]),
        ]
        yy = 0.95
        fig.text(0.07, yy, "How to read this report", fontsize=13, weight="bold")
        yy -= 0.045
        for head, paras in sections:
            fig.text(0.07, yy, head, fontsize=10.5, weight="bold")
            yy -= 0.025
            for p in paras:
                lines = textwrap.wrap(p, 105)
                fig.text(0.07, yy, "\n".join(lines), fontsize=8.5, va="top", linespacing=1.4)
                yy -= 0.0165 * len(lines) + 0.012
            yy -= 0.015
        pdf.savefig(fig)
        plt.close(fig)
    rd.close()

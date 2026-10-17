"""Patient-level explanations and report.

The risk is a sum of patch terms, so the Shapley values of the patches add up exactly to R - b and the contribution of
a region (an L2 tile with its L1 patches) is the sum of those of its patches. All attributions are out of fold: only the
models of the patient's test fold are used, and the reference patients are the patients of the same cohort in the
other folds (their risks are the out-of-fold risks of their own fold).

    explain()           Shapley values of the patches of every patient, deletion check          report/explanations/<run>/
    fit_atlas()         cohort patterns: k-means on the encoder features of the most risk-raising and most protective
                        patches of all patients                                                  report/atlas.npz
    describe_patterns() tissue class and ontology terms of every pattern (editable)             report/patterns.json
    make_report()       the report of one patient (PDF) and, optionally, the figures of the paper

The report has three levels, so that the reader can tell what the model measured from what is inferred:
  1. model contribution   signed contribution of every region and of its patches (the only level that describes the model)
  2. cohort pattern       the pattern the patches of the region belong to
  3. concept match        automatic QuiltNet text match, shown only for glandular regions and for concepts with
                          cohort-level evidence; otherwise the report abstains
"""
from __future__ import annotations

import json
import textwrap
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from . import concepts as cc  # noqa: E402
from .bank import load_bank  # noqa: E402
from .explain import shapley  # noqa: E402
from .faithfulness import load_members, member_parts, subset_score  # noqa: E402
from .layout import PATCH, Layout  # noqa: E402
from .survival import assign_risk_group, fit_breslow, risk_group_thresholds  # noqa: E402
from .utils import get_logger, load_folds  # noqa: E402
from .wsi import TiledPyramid  # noqa: E402

RED, BLUE, GREY, DARK = "#c0392b", "#2c6fbb", "#7f8c8d", "#333333"
GROUP_COLOR = {"low": "#27ae60", "medium": "#e67e22", "high": "#c0392b"}
CMAP = LinearSegmentedColormap.from_list("risk", [BLUE, "#f7f7f7", RED])
A4 = (8.27, 11.69)
S1 = PATCH * 4  # side of an L1 patch in level-0 px
NOT_ASSESSED = "ISUP grade, extraprostatic extension, surgical margins, perineural or lymphovascular invasion, PSA"
MIN_CONTRAST = 0.6   # a concept is suggested for a region if it is this many sd more similar than the average glandular patch
ARTIFACT_Z = 1.5     # patches whose artifact score is above this are flagged
SUGGESTABLE = {"benign", "atrophy", "gleason3", "gleason4", "cribriform", "gleason5", "necrosis"}  # glandular concepts


# ----------------------------------------------------------------------------- explanations
def explain(layout: Layout, run_name: str, n_permutations: int = 256, n_random: int = 20, device=None,
            max_cases: int | None = None, horizon_fraction: float = 0.10) -> Path:
    """Shapley values of the patches of every patient and the patient-specific deletion check."""
    import torch
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rdir = layout.run(run_name)
    cfg = json.loads((rdir / "config.json").read_text())
    out = layout.report / "explanations" / run_name
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger(out / "log.txt", f"explain_{run_name}")
    sp = load_folds(layout).set_index("case_id")
    folds = sorted(sp.fold.unique())
    oof = pd.concat([pd.read_csv(rdir / f"fold_{k}" / "test_predictions.csv").assign(fold=k) for k in folds]).set_index("case_id")
    for fold in folds:
        test = [c for c in oof[oof.fold == fold].index if not (out / f"{c}.npz").exists()][:max_cases]
        if not test:
            continue
        members = load_members(rdir / f"fold_{fold}", cfg, device)
        bank, _ = load_bank(layout, test, cfg["encoder"], cfg["views"])
        for c in test:
            coh = sp.loc[c, "cohort"]
            base_r = float(oof[(oof.fold != fold) & (sp.loc[oof.index, "cohort"].to_numpy() == coh)].risk_score.mean())
            parts, phis = [], []
            for m, mu, sd in members:
                s, g, _ = member_parts(m, bank[c].X.float(), device)
                parts.append((s, g, mu, sd))
                phis.append(shapley(s, g, baseline=mu + base_r * sd, n_permutations=n_permutations) / sd)
            phi = np.mean(phis, 0)
            n = len(phi)
            ones = np.ones(n, bool)
            full = subset_score(parts, ones)
            sign = 1.0 if full >= base_r else -1.0
            k = max(1, min(int(np.ceil(horizon_fraction * n)), n - 1))

            def shift(order):  # share of the deviation from the reference risk removed by hiding these patches
                keep = ones.copy()
                keep[order[:k]] = False
                return (full - subset_score(parts, keep)) / (full - base_r) if full != base_r else 0.0

            rng = np.random.default_rng(zlib.crc32(c.encode()))
            phi_anchor = np.empty(n)
            phi_anchor[bank[c].order] = phi  # back to the row order of the anchor file
            np.savez_compressed(
                out / f"{c}.npz", phi=phi_anchor.astype(np.float32), risk_z=full, baseline=base_r, fold=fold,
                member_z=np.array([subset_score([p], ones) for p in parts]),
                deletion=shift(np.argsort(-sign * phi)),
                deletion_random=float(np.mean([shift(rng.permutation(n)) for _ in range(n_random)])))
        log.info(f"fold {fold}: {len(test)} patients")
    return out


# ----------------------------------------------------------------------------- cohort patterns
def _l2(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def fit_atlas(layout: Layout, run_name: str, encoder: str = "dino", k: int = 8, top_frac: float = 0.02, seed: int = 0):
    """k-means (on L2-normalised features) of the `top_frac` most risk-raising and most protective patches of every patient."""
    from sklearn.cluster import KMeans
    exp = layout.report / "explanations" / run_name
    cases = [c for c in load_folds(layout).case_id if (exp / f"{c}.npz").exists()]
    feats, meta = [], []
    for ci, c in enumerate(cases):
        phi = np.load(exp / f"{c}.npz")["phi"]
        n = max(1, int(round(top_frac * len(phi))))
        rows = np.concatenate([np.argsort(-phi)[:n], np.argsort(phi)[:n]])
        order = np.argsort(rows)
        F = np.load(layout.features(encoder, "children") / f"{c}.npy", mmap_mode="r")[rows[order]]
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        tissue = pd.read_csv(layout.tree_anchors("children") / f"{c}.csv", usecols=["tissue_L1"]).tissue_L1.to_numpy()
        feats.append(np.asarray(F)[inv])
        meta.append(np.stack([np.full(len(rows), ci), rows, np.r_[np.ones(n), np.zeros(n)], tissue[rows]], 1))
    X, meta = _l2(np.concatenate(feats)), np.concatenate(meta)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    out = layout.report
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "atlas.npz", centroids=_l2(km.cluster_centers_), labels=km.labels_.astype(np.int16),
                        case_idx=meta[:, 0].astype(np.int32), rows=meta[:, 1].astype(np.int32),
                        raising=meta[:, 2].astype(bool), tissue=meta[:, 3].astype(np.float32), cases=np.array(cases),
                        encoder=encoder, top_frac=top_frac, run=run_name)
    return out / "atlas.npz"


def describe_patterns(layout: Layout, tissue_threshold: float = 0.7) -> list:
    """Tissue class and ontology terms of every pattern, written to report/patterns.json (edit the file to curate it).

    class  'edge'        mean tissue fraction of its patches < `tissue_threshold` (partly empty tiles at the section border)
           'background'  its patches resemble most the artifact or fat concepts
           'stroma'      otherwise, if they resemble most a stromal concept
           'glandular'   otherwise
    terms  the two most enriched concepts of the pattern's class (enrichment = share of its patches above the 90th
           percentile of the concept score, relative to 10%)
    """
    pack, _ = cc.load(layout)
    a = np.load(layout.report / "atlas.npz", allow_pickle=True)
    labels, cases = a["labels"], list(a["cases"])
    Q = np.zeros((len(labels), len(pack["names"])), np.float32)
    for ci, c in enumerate(cases):
        sel = np.flatnonzero(a["case_idx"] == ci)
        rows = a["rows"][sel]
        order = np.argsort(rows)
        q = np.load(layout.features("quiltnet", "children") / f"{c}.npy", mmap_mode="r")[rows[order]]
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        Q[sel] = cc.patch_z(pack, np.asarray(q)[inv])
    thr = np.quantile(Q, 0.9, axis=0)
    names, terms = pack["names"], pack["terms"]
    res = []
    for j in range(len(a["centroids"])):
        m = labels == j
        enr = (Q[m] >= thr).mean(0) / 0.1 if m.any() else np.zeros(len(names))
        top = names[int(np.argmax(enr))]
        mean_tissue = float(a["tissue"][m].mean()) if m.any() else float("nan")
        if mean_tissue < tissue_threshold:
            tissue = "edge"
        elif top in ("artifact", "fat"):
            tissue = "background"
        elif top in cc.STROMAL:
            tissue = "stroma"
        else:
            tissue = "glandular"
        allowed = cc.GLANDULAR if tissue == "glandular" else cc.STROMAL if tissue == "stroma" else set()
        best = sorted([i for i, n in enumerate(names) if n in allowed], key=lambda i: -enr[i])[:2]
        res.append({"pattern": f"P{j + 1}", "index": j, "tissue": tissue, "n_patches": int(m.sum()),
                    "share_raising": float(a["raising"][m].mean()) if m.any() else float("nan"),
                    "mean_tissue_fraction": mean_tissue,
                    "name": f"{tissue} pattern {j + 1}",
                    "ontology": [f"{terms[names[i]]['terms'][0]['label']} [{terms[names[i]]['terms'][0]['id']}]" for i in best]})
    (layout.report / "patterns.json").write_text(json.dumps(res, indent=1))
    return res


def save_prototypes(layout: Layout, n: int = 12) -> None:
    """report/patterns/pattern_<k>.png: the n pooled patches closest to every centroid, to inspect and curate the patterns."""
    a = np.load(layout.report / "atlas.npz", allow_pickle=True)
    cases = list(a["cases"])
    out = layout.report / "patterns"
    out.mkdir(exist_ok=True)
    cen = a["centroids"]
    for j in range(len(cen)):
        idx = np.flatnonzero(a["labels"] == j)
        sims = []
        for ci in np.unique(a["case_idx"][idx]):
            sel = idx[a["case_idx"][idx] == ci]
            F = _l2(np.load(layout.features(str(a["encoder"]), "children") / f"{cases[ci]}.npy", mmap_mode="r")[np.sort(a["rows"][sel])])
            sims += list(zip((F @ cen[j]).tolist(), [ci] * len(sel), np.sort(a["rows"][sel]).tolist()))
        sims = sorted(sims, reverse=True)[:n]
        fig, axes = plt.subplots(2, n // 2, figsize=(2 * (n // 2), 4))
        for ax, (_, ci, row) in zip(axes.ravel(), sims):
            c = cases[ci]
            xy = pd.read_csv(layout.tree_anchors("children") / f"{c}.csv", usecols=["x", "y"]).iloc[row]
            with TiledPyramid(layout.slide(c)) as rd:
                ax.imshow(rd.read_patch(xy.x, xy.y, 1, PATCH))
            ax.axis("off")
        fig.suptitle(f"pattern P{j + 1}")
        fig.savefig(out / f"pattern_{j + 1}.png", dpi=80, bbox_inches="tight")
        plt.close(fig)


# ----------------------------------------------------------------------------- survival helpers
def _km(time, event, grid):
    order = np.argsort(time)
    t, e = np.asarray(time, float)[order], np.asarray(event)[order].astype(bool)
    at_risk = len(t) - np.arange(len(t))
    surv = np.cumprod(np.where(e, 1.0 - 1.0 / at_risk, 1.0))
    idx = np.searchsorted(t, grid, side="right") - 1
    return np.where(idx >= 0, surv[np.clip(idx, 0, len(t) - 1)], 1.0)


def _recurrence(time, event, t0):
    """Kaplan-Meier recurrence probability (%) at t0 and the number of patients still at risk."""
    s = float(_km(time, event, np.array([t0]))[0])
    return 100.0 * (1 - s), int((np.asarray(time) >= t0).sum())


def neighbour_km(ref_risk, ref_time, ref_event, risk, k: int = 60) -> dict:
    """Observed recurrence among the k reference patients with the closest score, and in all of them."""
    ref_risk, ref_time, ref_event = map(np.asarray, (ref_risk, ref_time, ref_event))
    nn = np.argsort(np.abs(ref_risk - risk))[:k]
    grid = np.linspace(0, 10, 101)
    out = {"k": int(len(nn)), "grid": grid, "S_nn": _km(ref_time[nn], ref_event[nn], grid),
           "S_all": _km(ref_time, ref_event, grid)}
    for t0 in (5, 10):
        out[f"p{t0}"], out[f"risk{t0}"] = _recurrence(ref_time[nn], ref_event[nn], t0)
        out[f"p{t0}_all"], _ = _recurrence(ref_time, ref_event, t0)
    return out


# ----------------------------------------------------------------------------- one patient
class Reporter:
    """Loads everything shared by the reports of one run once."""

    def __init__(self, layout: Layout, run_name: str, horizon: float = 10.0):
        self.layout, self.run, self.horizon = layout, run_name, horizon
        self.cfg = json.loads(layout.run(run_name).joinpath("config.json").read_text())
        self.sp = load_folds(layout).set_index("case_id")
        folds = sorted(self.sp.fold.unique())
        self.oof = pd.concat([pd.read_csv(layout.run(run_name) / f"fold_{k}" / "test_predictions.csv").assign(fold=k)
                              for k in folds]).set_index("case_id")
        self.expl = layout.report / "explanations" / run_name
        self.atlas = np.load(layout.report / "atlas.npz", allow_pickle=True)
        self.patterns = json.loads((layout.report / "patterns.json").read_text())
        self.pack, self.validation = cc.load(layout)

    def reference(self, case: str):
        coh, fold = self.sp.loc[case, "cohort"], self.sp.loc[case, "fold"]
        ids = self.sp.index[(self.sp.cohort == coh) & (self.sp.fold != fold)]
        r = self.oof.loc[ids]
        return r.risk_score.to_numpy(), r.follow_up_years.to_numpy(), r.event.to_numpy()

    def survival(self, case: str, risk_z: float, member_z):
        ref_r, ref_t, ref_e = self.reference(case)
        horizon = float(min(self.horizon, ref_t.max()))
        br = fit_breslow(ref_r, ref_t, ref_e, horizon)
        thr = risk_group_thresholds(br.rmst(ref_r))
        t_pred = float(br.rmst(risk_z)[0])
        return {"time": t_pred, "group": str(assign_risk_group([t_pred], thr)[0]), "thresholds": thr, "horizon": horizon,
                "percentile": 100.0 * float((ref_r < risk_z).mean()), "n_ref": len(ref_r), "breslow": br,
                "members": [str(assign_risk_group([float(br.rmst(z)[0])], thr)[0]) for z in member_z],
                "km": neighbour_km(ref_r, ref_t, ref_e, risk_z)}

    def pattern_of(self, F):
        return np.argmax(_l2(F) @ self.atlas["centroids"].T, axis=1)


def _wrap(text, width):
    return textwrap.wrap(text, width) or [""]


def _thumb(rd: TiledPyramid):
    lv = min(4, rd.n_levels - 1)
    H, W = rd.shapes[lv]
    return rd.read_region(lv, 0, 0, W, H), 4 ** lv


def _draw_map(ax, thumb, scale, xy, phi, regions=None, vmax=None):
    ax.imshow(thumb)
    vmax = vmax or float(np.quantile(np.abs(phi), 0.995)) or 1.0
    half = S1 / scale / 2
    over = np.zeros(thumb.shape[:2] + (4,), np.float32)
    col = CMAP((np.clip(phi, -vmax, vmax) / vmax + 1) / 2)
    for (x, y), c, p in zip(xy / scale, col, phi):
        x0, x1, y0, y1 = int(max(x - half, 0)), int(x + half) + 1, int(max(y - half, 0)), int(y + half) + 1
        over[y0:y1, x0:x1, :3] = c[:3]
        over[y0:y1, x0:x1, 3] = min(1.0, 0.25 + abs(p) / vmax)
    ax.imshow(over)
    for x, y, num in regions or []:  # key regions: the L2 tile (side 4 L1 patches) and its number
        ax.add_patch(Rectangle((x / scale - 4 * half, y / scale - 4 * half), 8 * half, 8 * half, fill=False, ec="k", lw=0.8))
        ax.text(x / scale, y / scale, f"{num:+d}", color="white", fontsize=6, ha="center", va="center", weight="bold",
                bbox=dict(boxstyle="circle,pad=0.15", fc=RED if num > 0 else BLUE, ec="none"))
    ax.axis("off")
    return vmax


def _region_figure(fig, rd, thumb, scale, region, ch, phi, vmax, title):
    """Position on the slide / the L2 region with its L1 patches / the strongest patch / its centre at level 0."""
    px, py = region.x, region.y
    axs = [fig.add_axes(r) for r in region.axes]
    axs[0].imshow(thumb)
    axs[0].add_patch(Rectangle((px / scale - 2 * S1 / scale, py / scale - 2 * S1 / scale), 4 * S1 / scale,
                               4 * S1 / scale, fill=False, ec=RED, lw=1.2))
    axs[0].set_title("position on the slide", fontsize=6)
    axs[1].imshow(rd.read_patch(px, py, 2, PATCH))
    sub = ch[ch.parent_id == region.parent_id]
    top = sub.phi.abs().idxmax()
    for i, r in sub.iterrows():
        side = PATCH / 4
        x0, y0 = PATCH / 2 + (r.x - px) / 16 - side / 2, PATCH / 2 + (r.y - py) / 16 - side / 2
        axs[1].add_patch(Rectangle((x0, y0), side, side, fill=False, ec=RED if r.phi > 0 else BLUE,
                                   lw=2.2 if i == top else 0.8, alpha=min(1.0, 0.35 + abs(r.phi) / vmax)))
    axs[1].set_title("region (L2) 0.87 mm", fontsize=6)
    t = ch.loc[top]
    axs[2].imshow(rd.read_patch(t.x, t.y, 1, PATCH))
    axs[2].add_patch(Rectangle((PATCH * 3 / 8, PATCH * 3 / 8), PATCH / 4, PATCH / 4, fill=False, ec="k", lw=1))
    axs[2].set_title("strongest patch (L1) 217 um", fontsize=6)
    axs[3].imshow(rd.read_patch(t.x, t.y, 0, PATCH))
    axs[3].set_title("its centre (L0) 54 um", fontsize=6)
    for ax in axs:
        ax.axis("off")
    fig.text(region.axes[0][0], region.axes[0][1] + region.axes[0][3] + 0.035, title, fontsize=7.5, weight="bold", va="bottom")
    return t


def make_report(rep: Reporter, case: str, out_dir: Path, n_raise: int = 4, n_lower: int = 2, figures: bool = False) -> Path:
    lay = rep.layout
    ex = np.load(rep.expl / f"{case}.npz")
    phi, risk_z, base = ex["phi"].astype(float), float(ex["risk_z"]), float(ex["baseline"])
    sv = rep.survival(case, risk_z, ex["member_z"])
    ch = pd.read_csv(lay.tree_anchors("children") / f"{case}.csv")
    ch["phi"] = phi
    par = pd.read_csv(lay.tree_anchors("parents") / f"{case}.csv")
    reg = ch.groupby("parent_id").agg(phi=("phi", "sum"), n=("phi", "size")).join(par.set_index("anchor_id")[["x", "y"]]).reset_index()
    up = reg.sort_values("phi", ascending=False).head(n_raise)
    dn = reg.sort_values("phi").head(n_lower)

    # levels 2 and 3 for every patch: cohort pattern and concept scores
    F = np.load(lay.features(str(rep.atlas["encoder"]), "children") / f"{case}.npy").astype(np.float32)
    ch["pattern"] = rep.pattern_of(F)
    tissue_of = np.array([p["tissue"] for p in rep.patterns])
    ch["tissue"] = tissue_of[ch.pattern.to_numpy()]
    Z = cc.patch_z(rep.pack, np.load(lay.features("quiltnet", "children") / f"{case}.npy"))
    names, desc = rep.pack["names"], rep.pack["descriptions"]
    coh = rep.sp.loc[case, "cohort"]
    ok = [j for j, n in enumerate(names) if n in cc.usable(rep.validation, coh) and n in SUGGESTABLE]
    gl = (ch.tissue == "glandular").to_numpy()

    def match(rows):
        if not len(rows):
            return "no suggestion"
        kinds = ch.tissue.iloc[rows].value_counts()
        if kinds.index[0] != "glandular":
            return f"{kinds.index[0]} pattern - no glandular subtype suggested"
        ref = Z[gl].mean(0) if gl.any() else Z.mean(0)
        d = Z[rows].mean(0) - ref
        js = [j for j in np.argsort(-d) if d[j] >= MIN_CONTRAST and j in ok][:1]
        return "no confident match (abstains)" if not js else f"resembles '{desc[js[0]]}' (automatic match, unconfirmed)"

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{case}.pdf"
    with TiledPyramid(lay.slide(case)) as rd, PdfPages(pdf_path) as pdf:
        thumb, scale = _thumb(rd)
        xy = ch[["x", "y"]].to_numpy(float)
        numbering = list(range(1, len(up) + 1)) + [-i for i in range(1, len(dn) + 1)]
        marks = [(r.x, r.y, num) for r, num in zip(list(up.itertuples()) + list(dn.itertuples()), numbering)]

        # page 1: summary
        fig = plt.figure(figsize=A4)
        fig.text(0.06, 0.965, f"BCR prediction report: {case} ({coh})", fontsize=13, weight="bold", va="top")
        km = sv["km"]
        lines = [f"Risk group: {sv['group'].upper()}    predicted BCR-free time (restricted mean up to {sv['horizon']:.0f} y): "
                 f"{sv['time']:.1f} years",
                 f"Score higher than {sv['percentile']:.0f}% of the {sv['n_ref']} reference patients of the {coh} cohort; "
                 f"risk group of the {len(sv['members'])} models: {', '.join(sv['members'])}.",
                 f"Among the {km['k']} reference patients with the closest score, {km['p5']:.0f}% had recurred by 5 years and "
                 f"{km['p10']:.0f}% by 10 years (all reference patients: {km['p5_all']:.0f}% and {km['p10_all']:.0f}%).",
                 f"Hiding the top-ranked 10% of the tissue removes {100 * float(ex['deletion']):.0f}% of the difference between "
                 f"the score and the reference mean; hiding random tissue of the same area removes "
                 f"{100 * float(ex['deletion_random']):.0f}%."]
        y = 0.935
        for t in lines:
            for ln in _wrap(t, 105):
                fig.text(0.06, y, ln, fontsize=8.5, va="top", color=GROUP_COLOR[sv["group"]] if t is lines[0] else DARK)
                y -= 0.017
            y -= 0.006
        ax = fig.add_axes([0.08, 0.50, 0.38, 0.22])
        ax.step(km["grid"], 100 * (1 - km["S_nn"]), color=RED, where="post", label=f"closest {km['k']} patients")
        ax.step(km["grid"], 100 * (1 - km["S_all"]), color=GREY, where="post", label="all reference patients")
        ax.set_xlabel("years", fontsize=7)
        ax.set_ylabel("recurrence (%)", fontsize=7)
        ax.legend(fontsize=6, frameon=False)
        ax.tick_params(labelsize=6)
        ax2 = fig.add_axes([0.50, 0.30, 0.46, 0.46])
        vmax = _draw_map(ax2, thumb, scale, xy, phi, marks)
        ax2.set_title("attribution map: red raises the score, blue lowers it", fontsize=7)
        yy = 0.26
        for ln in _wrap("Levels of the report: (1) model contribution, the signed contribution of a region to the score; "
                        "(2) cohort pattern of its patches; (3) automatic concept match, which may abstain. Only (1) "
                        "describes the model; (2) and (3) help to read it and have not been reviewed by a pathologist.", 120):
            fig.text(0.06, yy, ln, fontsize=7, va="top")
            yy -= 0.0135
        fig.text(0.06, yy - 0.01, "Not assessed by the model: " + NOT_ASSESSED + ".", fontsize=7, va="top", color=GREY)
        pdf.savefig(fig)
        plt.close(fig)
        if figures:
            f2 = plt.figure(figsize=(6, 6))
            _draw_map(f2.add_axes([0.02, 0.02, 0.8, 0.96]), thumb, scale, xy, phi, marks, vmax=vmax)
            f2.savefig(out_dir / f"{case}_map.pdf", bbox_inches="tight")
            plt.close(f2)

        # pages 2-3: regions
        for title, sel, nums in (("Regions that raise the score", up, numbering[:len(up)]),
                                 ("Regions that lower the score", dn, numbering[len(up):])):
            fig = plt.figure(figsize=A4)
            fig.text(0.06, 0.965, title, fontsize=12, weight="bold", va="top")
            per = 0.225
            for i, (r, num) in enumerate(zip(sel.itertuples(), nums)):
                top = 0.92 - i * per
                r_axes = [[0.06 + 0.23 * j, top - 0.135, 0.21, 0.135] for j in range(4)]
                reg_row = type("R", (), {"x": r.x, "y": r.y, "parent_id": r.parent_id, "axes": r_axes})()
                t = _region_figure(fig, rd, thumb, scale, reg_row, ch, phi, vmax,
                                   f"region {num:+d}: contribution {r.phi:+.4f} z units, {r.n} patches")
                rows = np.flatnonzero(ch.parent_id.to_numpy() == r.parent_id)
                share = ch.iloc[rows].pattern.value_counts(normalize=True)
                pats = ", ".join(f"{rep.patterns[int(p)]['pattern']} {100 * s:.0f}%" for p, s in share.items())
                art = float((Z[rows, names.index('artifact')] >= ARTIFACT_Z).mean())
                txt = [f"cohort patterns of its patches: {pats} ({rep.patterns[int(share.index[0])]['name']})",
                       f"concept match: {match(rows)}"]
                if art > 0.25:
                    txt.append(f"{100 * art:.0f}% of its patches look like artifact: possibly spurious evidence")
                if (ch.tissue.iloc[rows] == "edge").mean() > 0.5:
                    txt.append("mostly tissue edge: possibly spurious evidence")
                for j, ln in enumerate(txt):
                    fig.text(0.06, top - 0.155 - 0.0135 * j, ln, fontsize=6.8, va="top", color=DARK)
                if figures and i == 0 and title.startswith("Regions that raise"):
                    f3 = plt.figure(figsize=(7, 2))
                    _region_figure(f3, rd, thumb, scale, type("R", (), {"x": r.x, "y": r.y, "parent_id": r.parent_id,
                                   "axes": [[0.01 + 0.25 * j, 0.02, 0.23, 0.8] for j in range(4)]})(), ch, phi, vmax, "")
                    f3.savefig(out_dir / f"{case}_region.pdf", bbox_inches="tight")
                    plt.close(f3)
            pdf.savefig(fig)
            plt.close(fig)

        # page 4: patterns and concepts
        fig = plt.figure(figsize=A4)
        fig.text(0.06, 0.965, "Cohort patterns and concepts", fontsize=12, weight="bold", va="top")
        n = max(1, int(round(0.05 * len(phi))))
        y = 0.93
        for what, rows in (("raising", np.argsort(-phi)[:n]), ("lowering", np.argsort(phi)[:n])):
            share = ch.iloc[rows].pattern.value_counts(normalize=True)
            fig.text(0.06, y, f"Patterns of the {n} patches that most {what} the score:", fontsize=8, weight="bold", va="top")
            y -= 0.018
            for p, s in share.head(5).items():
                pt = rep.patterns[int(p)]
                onto = "; ".join(pt["ontology"]) if pt["ontology"] else "no ontology term"
                for ln in _wrap(f"{pt['pattern']} ({pt['name']}) {100 * s:.0f}%: {onto}", 120):
                    fig.text(0.07, y, ln, fontsize=7, va="top")
                    y -= 0.0135
            y -= 0.01
        shown = [f"{desc[j]} [C {rep.validation.loc[names[j], 'C_' + coh]:.2f}]" for j in range(len(names))
                 if names[j] in cc.usable(rep.validation, coh)]
        fig.text(0.06, y, f"Concepts with cohort-level evidence in {coh} (C >= 0.55 or <= 0.45): {len(shown)}", fontsize=8,
                 weight="bold", va="top")
        y -= 0.018
        for s in shown:
            fig.text(0.07, y, s, fontsize=7, va="top")
            y -= 0.0135
        y -= 0.01
        for ln in _wrap("The concept match is an automatic text match with QuiltNet-B-16, shown only for glandular regions and "
                        "only for concepts with cohort-level evidence, otherwise the report abstains. A C-index measures "
                        "prognostic association, not how reliably a description is recognised.", 125):
            fig.text(0.06, y, ln, fontsize=7, va="top", color=GREY)
            y -= 0.0135
        pdf.savefig(fig)
        plt.close(fig)
    return pdf_path

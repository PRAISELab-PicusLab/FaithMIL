"""Deletion and insertion tests of the attributions of a trained run, out of fold, at equal tissue area.

For every patient only the models of its test fold are used (they never saw the patient). The units of an explanation
are the instances of the model: 0.22 mm patches (L1 and context models) or L2 / L3 / L0 tiles (single-level models of
those levels). A unit counts as 1 tissue tile for L1 patches and as its number of L1 children for the L2 tiles of the
tree, so that the removed fraction f of the tissue is comparable between models.

  attributions   shapley          Monte-Carlo Shapley values of the units (256 antithetic permutations)
                 attention        the attention weight
                 attn_x_score     a_c (s_c - b), the exact first-order decomposition of R - b
                 occlusion        change of the risk when the unit alone is removed
                 grad_x_input     gradient of the risk w.r.t. the unit embedding times the embedding
  deletion       units are removed in decreasing order of sign * attribution until the fraction f of the tissue area is
                 removed; drop(f) = sign * (R(all) - R(rest)); sign = +1 if the patient lies above the reference risk b
                 (mean out-of-fold risk of the patients of the same cohort outside the test fold), -1 below
  deletion gap   mean drop over f in {1, 2, 5, 10, 20, 30, 50}% (AOPC) minus the same quantity for random removal of the
                 same area (20 random orders)
  insertion gap  only the top units up to f are kept: sign * (R(top) - b), minus the same for random units
  sanity check   Spearman correlation between the Shapley values of a trained model and of the same model re-initialised
                 (Adebayo et al., 2018)

    faithfulness/<run>/cases.csv    one row per patient
    faithfulness/<run>/table.csv    means with patient-bootstrap 95% intervals, per cohort and attribution method
    faithfulness/<run>/sanity.csv
"""
from __future__ import annotations

import copy
import json
import zlib

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from .bank import load_bank
from .explain import shapley
from .layout import Layout
from .models import build_model
from .utils import get_logger, load_folds

FRACS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
METHODS = ["shapley", "attention", "attn_x_score", "occlusion", "grad_x_input"]


def load_members(fdir, config, device):
    """[(model, mu, sd)] of the seeds of one fold."""
    out = []
    for p in sorted(fdir.glob("model_seed*.pt")):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = build_model({**config, "dims": ck["dims"]})
        m.load_state_dict(ck["state_dict"])
        out.append((m.eval().to(device), float(ck["mu"]), float(ck["sd"])))
    if not out:
        raise FileNotFoundError(f"no model_seed*.pt in {fdir} (train with --save-models)")
    return out


def member_parts(model, X, device):
    """Scores s, attention logits g and gradient x input of the unit embeddings of one patient (numpy, float64)."""
    X = X.to(device)
    with torch.no_grad():
        h = model.embed(X)
    h = h.detach().requires_grad_(True)
    s = model.score(h).squeeze(-1)
    g = model.gate(h)
    (torch.softmax(g, 0) * s).sum().backward()
    gxi = (h.grad * h).sum(1)
    cvt = lambda t: t.detach().double().cpu().numpy()  # noqa: E731
    return cvt(s), cvt(g), cvt(gxi)


def subset_score(parts, keep):
    """Ensemble z-score of the risk computed on the kept units only (attention re-normalised over them)."""
    z = []
    for s, g, mu, sd in parts:
        gg = g[keep]
        e = np.exp(gg - gg.max())
        z.append(((e * s[keep]).sum() / e.sum() - mu) / sd)
    return float(np.mean(z))


def unit_area(layout: Layout, case: str, views, n: int) -> np.ndarray:
    """Tissue area of every unit, in L1 tiles; units are in the order of the bank (sorted by group)."""
    if views[0] == "parents":
        a = pd.read_csv(layout.tree_anchors("parents") / f"{case}.csv", usecols=["n_children"]).n_children
        if len(a) != n:
            raise RuntimeError(f"{case}: {n} units but {len(a)} tree parents")
        return a.to_numpy(float)
    return np.ones(n)


def k_for_area(order, area, f):
    """Number of leading units of `order` whose cumulative area first reaches the fraction f of the total (1 .. n-1)."""
    cum = np.cumsum(area[order])
    return max(1, min(int(np.searchsorted(cum, f * cum[-1]) + 1), len(order) - 1))


def run(layout: Layout, run_name: str, n_permutations: int = 256, n_random: int = 20, device=None,
        max_cases: int | None = None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rdir = layout.run(run_name)
    out = layout.faithfulness(run_name)
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger(out / "log.txt", f"faith_{run_name}")
    cfg = json.loads((rdir / "config.json").read_text())
    if cfg.get("pool"):
        raise ValueError("faithfulness is not defined for runs trained on a tile pool")
    sp = load_folds(layout).set_index("case_id")
    folds = sorted(sp.fold.unique())
    oof = pd.concat([pd.read_csv(rdir / f"fold_{k}" / "test_predictions.csv").assign(fold=k) for k in folds]).set_index("case_id")
    rows, sanity = [], []
    for fold in folds:
        test = oof[oof.fold == fold].index.tolist()[:max_cases]
        members = load_members(rdir / f"fold_{fold}", cfg, device)
        bank, _ = load_bank(layout, test, cfg["encoder"], cfg["views"])
        for ci, c in enumerate(test):
            coh = sp.loc[c, "cohort"]
            base_r = oof[(oof.fold != fold) & (sp.loc[oof.index, "cohort"].to_numpy() == coh)].risk_score.mean()
            mp, attrs = [], {k: [] for k in METHODS}
            for m, mu, sd in members:
                s, g, gxi = member_parts(m, bank[c].X.float(), device)
                mp.append((s, g, mu, sd))
                e = np.exp(g - g.max())
                a = e / e.sum()
                R = float((a * s).sum())
                b = mu + base_r * sd  # reference risk in the raw units of this model
                attrs["shapley"].append(shapley(s, g, baseline=b, n_permutations=n_permutations) / sd)
                attrs["attention"].append(a)
                attrs["attn_x_score"].append(a * (s - b) / sd)
                Z, N = e.sum(), (e * s).sum()
                attrs["occlusion"].append(np.where(len(s) > 1, R - (N - e * s) / np.maximum(Z - e, 1e-12), 0.0) / sd)
                attrs["grad_x_input"].append(gxi / sd)
            n = len(mp[0][0])
            area = unit_area(layout, c, cfg["views"], n)
            full = subset_score(mp, np.ones(n, bool))
            sign = 1.0 if full >= base_r else -1.0
            rng = np.random.default_rng(zlib.crc32(c.encode()))
            rand_orders = [rng.permutation(n) for _ in range(n_random)]

            def curves(order):
                dele, ins = [], []
                for f in FRACS:
                    k = k_for_area(order, area, f)
                    keep = np.ones(n, bool)
                    keep[order[:k]] = False
                    dele.append(sign * (full - subset_score(mp, keep)))
                    keep2 = np.zeros(n, bool)
                    keep2[order[:k]] = True
                    ins.append(sign * (subset_score(mp, keep2) - base_r))
                return np.array(dele), np.array(ins)

            rnd = [curves(o) for o in rand_orders]
            rd, ri = np.mean([r[0] for r in rnd], 0), np.mean([r[1] for r in rnd], 0)
            row = {"case_id": c, "cohort": coh, "fold": fold, "n_units": n, "tissue_tiles": float(area.sum()),
                   "AOPC_random": float(rd.mean())}
            for k in METHODS:
                v = np.mean(attrs[k], 0)
                order = np.argsort(-v) if k == "attention" else np.argsort(-sign * v)
                d_, i_ = curves(order)
                row[f"AOPC_{k}"] = float(d_.mean())
                row[f"gap_{k}"] = float(d_.mean() - rd.mean())
                row[f"ins_{k}"] = float((i_ - ri).mean())
                row[f"drop10_{k}"] = float(d_[FRACS.index(0.10)])
            rows.append(row)
            if len(sanity) < 50 and ci % 8 == 0:  # trained vs re-initialised weights
                m0 = members[0][0]
                mr = copy.deepcopy(m0)
                for mod in mr.modules():
                    if hasattr(mod, "reset_parameters"):
                        mod.reset_parameters()
                st, gt, _ = member_parts(m0, bank[c].X.float(), device)
                sr, gr, _ = member_parts(mr, bank[c].X.float(), device)
                phi = shapley(st, gt, baseline=float(st.mean()), n_permutations=n_permutations // 2)
                phr = shapley(sr, gr, baseline=float(sr.mean()), n_permutations=n_permutations // 2)
                sanity.append({"case_id": c, "spearman_trained_vs_random_weights": float(spearmanr(phi, phr).statistic)})
        log.info(f"fold {fold}: {len(test)} cases")
    D = pd.DataFrame(rows)
    D.to_csv(out / "cases.csv", index=False)
    pd.DataFrame(sanity).to_csv(out / "sanity.csv", index=False)
    T = summarise(D)
    T.to_csv(out / "table.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(T.to_string(index=False))
    return D, T


def summarise(D: pd.DataFrame, n_boot: int = 1000, seed: int = 1) -> pd.DataFrame:
    """Mean gaps with patient-bootstrap 95% intervals, for all patients and per cohort."""
    rng = np.random.default_rng(seed)
    out = []
    for coh in ["all"] + sorted(D.cohort.unique()):
        d = D if coh == "all" else D[D.cohort == coh]
        boot = [rng.integers(0, len(d), len(d)) for _ in range(n_boot)]
        for k in METHODS:
            r = {"cohort": coh, "method": k}
            for col in ["gap", "ins", "drop10"]:
                v = d[f"{col}_{k}"].to_numpy()
                bs = [v[i].mean() for i in boot]
                r[col], r[f"{col}_lo"], r[f"{col}_hi"] = v.mean(), np.quantile(bs, .025), np.quantile(bs, .975)
            out.append(r)
    return pd.DataFrame(out)

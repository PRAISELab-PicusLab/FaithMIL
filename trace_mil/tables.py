"""The numbers of the tables of the paper, recomputed from the run outputs, with 95% bootstrap confidence intervals.

C-index        mean over the 10 test folds of Harrell's C-index of the fold (ties in risk count 1/2); rows with several
               trainings (the faithfulness-aware models are trained twice) average the per-fold C-indices of the trainings
               CI: 2000 bootstrap resamples of the patients within each fold (the same resamples for every run)
Delta vs L1/L2 paired difference to the L1 / L2 model without faithfulness-aware training, mean over the folds, with its
               95% interval from the same resamples, and the number of folds in which the row is better
Faithfulness   mean over the patients of the deletion and insertion gap of the Shapley attributions (seed-averaged per
               patient), with a patient-bootstrap 95% interval, and paired differences to the L1 / L2 models
* in the printed tables: the interval of the difference excludes 0
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .layout import Layout
from .utils import load_folds

B = 2000
REFS = ("L1", "L2")


def harrell(t, e, r) -> float:
    """Harrell's C (comparable pair = shorter time with an event; ties in risk count 1/2)."""
    t, e, r = map(np.asarray, (t, e, r))
    comp = (t[:, None] < t[None, :]) & (e[:, None] == 1)
    if not comp.any():
        return np.nan
    d = r[:, None] - r[None, :]
    return float(((d > 0) + 0.5 * (d == 0))[comp].sum() / comp.sum())


class CIndex:
    """Per-fold and bootstrap C-indices of runs, on fixed per-fold patient resamples."""

    def __init__(self, layout: Layout, seed: int = 0):
        self.layout = layout
        sp = load_folds(layout)
        self.folds = [g.sort_values("case_id") for _, g in sp.groupby("fold")]
        rng = np.random.default_rng(seed)
        self.boot = [rng.integers(0, len(g), size=(B, len(g))) for g in self.folds]
        self._cache: dict = {}

    def run(self, run: str):
        """(C per fold (n_folds,), bootstrap C (B, n_folds)) of one run."""
        if run not in self._cache:
            d = self.layout.run(run)
            r = pd.concat([pd.read_csv(f) for f in sorted(d.glob("fold_[0-9]*/test_predictions.csv"))])
            r = r.set_index("case_id").risk_score
            c, cb = [], []
            for g, bi in zip(self.folds, self.boot):
                t, e, x = g.follow_up_years.to_numpy(), g.event.to_numpy(), r.loc[g.case_id].to_numpy()
                c.append(harrell(t, e, x))
                cb.append([harrell(t[i], e[i], x[i]) for i in bi])
            self._cache[run] = (np.array(c), np.array(cb).T)
        return self._cache[run]

    def runs(self, runs: list):
        cs = [self.run(r) for r in runs]
        return np.mean([c for c, _ in cs], 0), np.mean([b for _, b in cs], 0)


def ci_mean(res):
    return tuple(np.nanpercentile(np.nanmean(res[1], 1), [2.5, 97.5]))


def paired(a, b):
    d = np.nanmean(a[1] - b[1], 1)  # a resampled fold can have no comparable pair (few events)
    return float(np.nanmean(a[0] - b[0])), *np.nanpercentile(d, [2.5, 97.5]), int((a[0] > b[0]).sum())


def level_table(ci: CIndex, layout: Layout, spec: dict) -> pd.DataFrame:
    rows = []
    for lvl_label, lvl in spec["levels"].items():
        for enc_label, enc in spec["encoders"].items():
            run = spec["run_pattern"].format(level=lvl, encoder=enc)
            if not layout.run(run).exists():
                continue
            r = ci.runs([run])
            lo, hi = ci_mean(r)
            rows.append({"level": lvl_label, "encoder": enc_label, "c_index": float(np.nanmean(r[0])), "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def cindex_table(ci: CIndex, groups: dict, refs: dict) -> pd.DataFrame:
    ref = {k: ci.runs(v) for k, v in refs.items()}
    out = []
    for group, rows in groups.items():
        for label, runs in rows:
            c = ci.runs(runs)
            lo, hi = ci_mean(c)
            r = {"group": group, "model": label, "n_runs": len(runs), "c_index": float(np.nanmean(c[0])), "lo": lo, "hi": hi}
            for k, rc in ref.items():
                if runs == refs[k]:
                    continue
                p = paired(c, rc)
                r |= {f"d_{k}": p[0], f"lo_{k}": p[1], f"hi_{k}": p[2], f"better_folds_{k}": p[3]}
            out.append(r)
    return pd.DataFrame(out)


def faith_cases(layout: Layout, runs: list, method: str) -> pd.DataFrame:
    dfs = [pd.read_csv(layout.faithfulness(r) / "cases.csv").set_index("case_id").sort_index() for r in runs]
    return pd.DataFrame({k: np.mean([d[f"{k}_{method}"] for d in dfs], 0) for k in ("gap", "ins")}, index=dfs[0].index)


def faithfulness_table(layout: Layout, groups: dict, refs: dict, method: str = "shapley", seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    boot = lambda v: tuple(np.percentile(v[rng.integers(0, len(v), size=(B, len(v)))].mean(1), [2.5, 97.5]))  # noqa: E731
    ref = {k: faith_cases(layout, v, method) for k, v in refs.items()}
    out = []
    for group, rows in groups.items():
        for label, runs in rows:
            f = faith_cases(layout, runs, method)
            r = {"group": group, "model": label}
            for kind, name in (("gap", "deletion"), ("ins", "insertion")):
                v = f[kind].to_numpy()
                r[name], (r[f"{name}_lo"], r[f"{name}_hi"]) = float(v.mean()), boot(v)
                for k, rf in ref.items():
                    if runs == refs[k]:
                        continue
                    d = (f[kind] - rf.loc[f.index, kind]).to_numpy()
                    r[f"{name}_d_{k}"], (r[f"{name}_lo_{k}"], r[f"{name}_hi_{k}"]) = float(d.mean()), boot(d)
            out.append(r)
    return pd.DataFrame(out)


def _excl0(lo, hi) -> bool:
    return bool(lo > 0 or hi < 0)


def _fmt_ci(t: pd.DataFrame, value: str, prefix: str = "") -> pd.Series:
    """'.624 [.55,.70]' (+ deltas with * when their interval excludes 0) for printing."""
    def one(r):
        s = f"{r[value]:.3f} [{r[f'{prefix}lo']:.2f},{r[f'{prefix}hi']:.2f}]"
        for k in REFS:
            col = f"{prefix}d_{k}"
            if col in r and pd.notna(r[col]):
                s += f"  d{k} {r[col]:+.3f} [{r[f'{prefix}lo_{k}']:+.2f},{r[f'{prefix}hi_{k}']:+.2f}]"
                s += "*" if _excl0(r[f"{prefix}lo_{k}"], r[f"{prefix}hi_{k}"]) else " "
        return s
    return t.apply(one, axis=1)


def build(layout: Layout, config_path: Path) -> None:
    cfg = json.loads(Path(config_path).read_text())
    out = layout.tables
    out.mkdir(parents=True, exist_ok=True)
    ci = CIndex(layout)
    exists = lambda runs: all(layout.run(x).exists() for x in runs)  # noqa: E731

    t1 = level_table(ci, layout, cfg["table1"])
    t1.to_csv(out / "table1_levels.csv", index=False)

    spec = cfg["table2"]
    refs = spec["references"]
    groups = {g: [r for r in spec[g] if exists(r[1])] for g in ("rows", "faithfulness_aware", "baselines") if g in spec}
    t2 = cindex_table(ci, groups, refs)
    t2.to_csv(out / "table2_cindex.csv", index=False)

    has_faith = lambda runs: all((layout.faithfulness(x) / "cases.csv").exists() for x in runs)  # noqa: E731
    fgroups = {g: [r for r in groups[g] if has_faith(r[1])] for g in ("rows", "faithfulness_aware") if g in groups}
    t3 = faithfulness_table(layout, fgroups, refs) if all(has_faith(v) for v in refs.values()) else pd.DataFrame()
    t3.to_csv(out / "table3_faithfulness.csv", index=False)

    print("\nTable 1: single levels (C-index [95% CI])")
    if len(t1):
        print(t1.assign(c=_fmt_ci(t1, "c_index")).pivot(index="level", columns="encoder", values="c")
              .reindex(list(cfg["table1"]["levels"])).to_string())
    print("\nTable 2: C-index [95% CI], paired difference to L1 / L2 (* = interval excludes 0)")
    for _, r in t2.assign(c=_fmt_ci(t2, "c_index")).iterrows():
        print(f"  {r.group:<18} {r.model:<24} {r.c}")
    if len(t3):
        print("\nTable 3: Shapley deletion / insertion gaps [95% CI], paired difference to L1 / L2")
        t3 = t3.assign(dl=_fmt_ci(t3, "deletion", "deletion_"), ins=_fmt_ci(t3, "insertion", "insertion_"))
        for _, r in t3.iterrows():
            print(f"  {r.group:<18} {r.model:<24} deletion {r.dl}\n  {'':<43} insertion {r.ins}")

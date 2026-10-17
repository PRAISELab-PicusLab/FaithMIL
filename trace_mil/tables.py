"""The numbers of the tables of the paper, recomputed from the run outputs.

C-index        mean over the test folds of Harrell's C-index of the fold, computed within each cohort
Delta vs L1    paired difference to the reference run, mean over the folds
Faithfulness   mean over the patients of the deletion and insertion gap of the Shapley attributions; for rows with several
               trainings (the faithfulness-aware models are trained twice) the per-patient gaps are averaged first, and the
               C-index is the mean of the trainings
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .layout import Layout
from .utils import c_index, load_folds

COHORTS = ["RUMC", "TCGA"]


def fold_c(layout: Layout, run: str) -> pd.DataFrame:
    """C-index of every (fold, cohort) of a run."""
    sp = load_folds(layout).set_index("case_id")
    rows = []
    for d in sorted(layout.run(run).glob("fold_[0-9]*")):
        t = pd.read_csv(d / "test_predictions.csv")
        t["cohort"] = sp.loc[t.case_id, "cohort"].to_numpy()
        for coh, g in t.groupby("cohort"):
            if g.event.sum() > 0:
                rows.append({"fold": int(d.name.split("_")[1]), "cohort": coh,
                             "c": c_index(g.follow_up_years, g.risk_score, g.event)})
    return pd.DataFrame(rows)


def mean_c(layout: Layout, runs: list) -> dict:
    """{cohort: (fold-mean C averaged over the trainings, per-fold C averaged over the trainings)}."""
    per = [fold_c(layout, r).pivot(index="fold", columns="cohort", values="c") for r in runs]
    avg = sum(per) / len(per)
    return {coh: (float(avg[coh].mean()), avg[coh]) for coh in COHORTS if coh in avg}


def cindex_table(layout: Layout, rows, reference: list | None = None) -> pd.DataFrame:
    ref = mean_c(layout, reference) if reference else None
    out = []
    for label, runs in rows:
        m = mean_c(layout, runs)
        r = {"model": label}
        for coh in COHORTS:
            r[coh] = m[coh][0]
            if ref is not None:
                d = m[coh][1] - ref[coh][1]
                r[f"delta_{coh}"] = float(d.mean())
                r[f"better_folds_{coh}"] = int((d > 0).sum())
        out.append(r)
    return pd.DataFrame(out)


def faithfulness_table(layout: Layout, rows, method: str = "shapley") -> pd.DataFrame:
    out = []
    for label, runs in rows:
        cases = [pd.read_csv(layout.faithfulness(r) / "cases.csv").set_index("case_id") for r in runs]
        r = {"model": label}
        for coh in COHORTS:
            sel = [c[c.cohort == coh] for c in cases]
            for col, name in (("gap", "deletion"), ("ins", "insertion")):
                r[f"{name}_{coh}"] = float(np.mean([s[f"{col}_{method}"] for s in sel], axis=0).mean())
        out.append(r)
    return pd.DataFrame(out)


def level_table(layout: Layout, spec: dict) -> pd.DataFrame:
    rows = []
    for lvl_label, lvl in spec["levels"].items():
        r = {"level": lvl_label}
        for enc_label, enc in spec["encoders"].items():
            run = spec["run_pattern"].format(level=lvl, encoder=enc)
            if not layout.run(run).exists():
                continue
            for coh, (v, _) in mean_c(layout, [run]).items():
                r[f"{enc_label} {coh}"] = v
        rows.append(r)
    return pd.DataFrame(rows)


def build(layout: Layout, config_path: Path) -> None:
    cfg = json.loads(Path(config_path).read_text())
    out = layout.tables
    out.mkdir(parents=True, exist_ok=True)
    t1 = level_table(layout, cfg["table1"])
    t1.to_csv(out / "table1_levels.csv", index=False)
    spec = cfg["table2"]
    ref = [spec["reference"]]
    rows = [r for r in spec["rows"] + spec["faithfulness_aware"] if all(layout.run(x).exists() for x in r[1])]
    t2 = cindex_table(layout, rows, ref)
    t2.to_csv(out / "table2_cindex.csv", index=False)
    frows = [r for r in rows if all((layout.faithfulness(x) / "cases.csv").exists() for x in r[1])]
    t3 = faithfulness_table(layout, frows)
    t3.to_csv(out / "table3_faithfulness.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        for name, t in (("Table 1", t1), ("Table 2", t2), ("Table 3", t3)):
            print(f"\n{name}\n{t.to_string(index=False)}")

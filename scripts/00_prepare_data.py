#!/usr/bin/env python3
"""Build metadata/cases.csv and splits/folds.csv in the data root.

Inputs (see the README for where to get them):
    images/<case>.tif, tissue-masks/<case>_tissue.tif
    one or more label CSVs with the columns case_id, event, follow_up_years (and optionally cohort); a trailing '.tif' of the
    case id is removed, and a missing cohort is inferred from the id ('case_radboud_*' = RUMC, 'TCGA-*' = TCGA)

    python scripts/00_prepare_data.py --labels labels_rumc.csv labels_tcga.csv

The folds are the ones of the paper (data/folds.csv: 10 folds stratified by event; RUMC and TCGA-PRAD patients are
spread over the same folds). `--make-folds N` draws new ones instead: StratifiedKFold on the event within each cohort.
Cases without a slide or a mask are reported and dropped.
"""
import _repo  # noqa: F401
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from trace_mil.layout import Layout

REPO = Path(__file__).resolve().parents[1]


def cohort_of(case: str) -> str:
    if case.startswith("case_radboud_"):
        return "RUMC"
    if case.startswith("TCGA-"):
        return "TCGA"
    raise ValueError(f"cannot infer the cohort of {case!r}; add a 'cohort' column to the labels")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--folds", default=str(REPO / "data" / "folds.csv"), help="case_id, cohort, fold")
    ap.add_argument("--make-folds", type=int, default=0, help="draw this many new folds instead of using --folds")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    lay = Layout(a.root)

    lab = pd.concat([pd.read_csv(p) for p in a.labels], ignore_index=True)
    lab["case_id"] = lab["case_id"].astype(str).str.replace(r"\.tif$", "", regex=True)
    if "cohort" not in lab:
        lab["cohort"] = [cohort_of(c) for c in lab.case_id]
    lab = lab[["case_id", "cohort", "event", "follow_up_years"]].drop_duplicates("case_id")
    ok = [lay.slide(c).exists() and lay.mask(c).exists() for c in lab.case_id]
    if not all(ok):
        print(f"WARNING: {len(ok) - sum(ok)} cases have no slide or mask in {lay.root} and are dropped, e.g. "
              f"{list(lab.case_id[~np.array(ok)][:5])}")
    lab = lab[ok].sort_values("case_id").reset_index(drop=True)
    lay.cases_csv.parent.mkdir(parents=True, exist_ok=True)
    lab.to_csv(lay.cases_csv, index=False)

    if a.make_folds:
        fold = pd.Series(-1, index=lab.index)
        for _, g in lab.groupby("cohort"):
            skf = StratifiedKFold(a.make_folds, shuffle=True, random_state=a.seed)
            for k, (_, te) in enumerate(skf.split(g, g.event)):
                fold[g.index[te]] = k
        folds = lab[["case_id", "cohort"]].assign(fold=fold)
    else:
        folds = pd.read_csv(a.folds)[["case_id", "fold"]]
        missing = set(lab.case_id) - set(folds.case_id)
        if missing:
            raise SystemExit(f"{len(missing)} cases are not in {a.folds}, e.g. {sorted(missing)[:5]}")
        folds = lab[["case_id", "cohort"]].merge(folds, on="case_id")
    out = folds.merge(lab[["case_id", "event", "follow_up_years"]], on="case_id")
    lay.folds_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(lay.folds_csv, index=False)
    print(out.groupby("cohort").agg(patients=("case_id", "size"), events=("event", "sum"),
                                    median_follow_up_years=("follow_up_years", "median")).to_string())
    print(f"{lay.cases_csv}\n{lay.folds_csv}")


if __name__ == "__main__":
    main()

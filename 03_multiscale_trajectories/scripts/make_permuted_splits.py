#!/usr/bin/env python3
"""Label-permutation null for the linear probe: permuted copies of folds.csv.

Within each outer fold, the (event, follow_up_years) pairs are shuffled across
patients. Features, fold membership and the number of events per fold stay
exactly as in the real experiment, so running scripts/linear_probe_survival.py
unchanged on these files gives the null distribution of THIS protocol on THESE
features. p = (1 + #{perm C >= real C}) / (1 + n_perm).

Example:
    python scripts/make_permuted_splits.py --n 30 --out-dir splits_perm
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits-csv", default="splits/folds.csv")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out-dir", default="splits_perm")
    args = p.parse_args()

    df = pd.read_csv(args.splits_csv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for i in range(args.n):
        perm = df.copy()
        for _, idx in df.groupby("fold").groups.items():
            idx = np.asarray(idx)
            src = rng.permutation(idx)
            perm.loc[idx, ["event", "follow_up_years"]] = df.loc[src, ["event", "follow_up_years"]].to_numpy()
        assert (perm.groupby("fold")["event"].sum() == df.groupby("fold")["event"].sum()).all()
        perm.to_csv(out / f"perm_{i:03d}.csv", index=False)
    print(f"{args.n} permuted splits -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Raw tiles of one level for the DINO pre-training of that level (see trace_mil/tiles.py).

    python scripts/02_extract_tiles.py --level 2
    python scripts/02_extract_tiles.py --level 1 --cap 3000
    python scripts/02_extract_tiles.py --level 3 --cap 1500 --per-cell 25
    python scripts/02_extract_tiles.py --level 0 --cap 3000

`--shard i --num-shards n` splits the cases between n independent processes.
"""
import _repo  # noqa: F401
import argparse

import pandas as pd

from trace_mil import tiles
from trace_mil.layout import Layout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--level", type=int, choices=[0, 1, 2, 3], required=True)
    ap.add_argument("--cap", type=int, default=3000, help="levels 0, 1, 3: maximum number of tiles per case")
    ap.add_argument("--per-cell", type=int, default=25, help="level 3: maximum number of crops per grid cell")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cases", nargs="*", default=None)
    a = ap.parse_args()
    lay = Layout(a.root)
    cases = (a.cases or sorted(pd.read_csv(lay.folds_csv).case_id))[a.shard::a.num_shards]
    print(f"level {a.level}: {len(cases)} cases (shard {a.shard}/{a.num_shards})", flush=True)
    tiles.run(lay, a.level, cases, a.cap, a.per_cell, a.workers)


if __name__ == "__main__":
    main()

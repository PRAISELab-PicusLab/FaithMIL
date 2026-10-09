#!/usr/bin/env python3
"""Per-slide pooling of cached trajectory features, in the linear-probe format.

For every level L and pooling P, writes <out-dir>/<backbone>_L<L>_<P>.npz with
    case_id: str[N], features: float32[N, D]
exactly what scripts/linear_probe_survival.py reads, so the trajectory
diagnostic runs the SAME nested-CV ridge-Cox protocol as the thumbnail probe.

Poolings (over the ~1300 trajectories of a slide, per feature dimension):
    mean     average
    max      maximum
    top<k>   mean of the k largest values (a softer max; default k=16, ~1%)

--null-seeds adds <backbone>_null<s>.npz: i.i.d. Gaussian features of the same
shape. Run through the probe they give the protocol's behaviour when there is
NO signal, the reference any real configuration must beat.

Example:
    python scripts/pool_trajectory_features.py --backbone imagenet --levels 5 4 3 2 1 0 --null-seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.trajectory.io import DEFAULT_TRAJ_ROOT, load_features  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default="imagenet")
    p.add_argument("--anchor-set", default="grid_L2_p224")
    p.add_argument("--levels", type=int, nargs="+", default=[5, 4, 3, 2, 1, 0])
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--null-seeds", type=int, nargs="*", default=[])
    p.add_argument("--traj-root", default=str(DEFAULT_TRAJ_ROOT))
    p.add_argument("--splits-csv", default="splits/folds.csv")
    p.add_argument("--out-dir", default="features_traj")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(args.splits_csv)["case_id"].tolist()
    ids = np.array(cases)
    poolings = ["mean", "max", f"top{args.topk}"]

    for lvl in args.levels:
        t0 = time.time()
        pooled = {pl: [] for pl in poolings}
        for c in cases:
            f = load_features(args.traj_root, args.anchor_set, args.backbone, lvl, c).astype(np.float32)
            k = min(args.topk, f.shape[0])
            pooled["mean"].append(f.mean(0))
            pooled["max"].append(f.max(0))
            pooled[f"top{args.topk}"].append(np.sort(f, axis=0)[-k:].mean(0))
        for pl, rows in pooled.items():
            X = np.stack(rows)
            if not np.isfinite(X).all():
                raise RuntimeError(f"non-finite pooled features for L{lvl} {pl}")
            np.savez_compressed(out / f"{args.backbone}_L{lvl}_{pl}.npz", case_id=ids, features=X)
        print(f"L{lvl}: {len(cases)} slides pooled ({', '.join(poolings)}) in {time.time() - t0:.0f}s", flush=True)

    dim = X.shape[1] if args.levels else 2048
    for s in args.null_seeds:
        X = np.random.default_rng(1000 + s).standard_normal((len(cases), dim)).astype(np.float32)
        np.savez_compressed(out / f"{args.backbone}_null{s}.npz", case_id=ids, features=X)
    if args.null_seeds:
        print(f"null controls: seeds {args.null_seeds}, D={dim}", flush=True)


if __name__ == "__main__":
    main()

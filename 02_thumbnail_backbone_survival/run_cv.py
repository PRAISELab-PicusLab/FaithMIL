#!/usr/bin/env python3
"""Drive train.py across folds (and optionally across the full experiment
matrix), reusing the exact same folds/config for every run.

Does NOT run anything by itself beyond what you ask for:
  - one task/backbone across all 10 folds (the common case):
        python run_cv.py --task classification --backbone barlow_twins
  - a subset of folds:
        python run_cv.py --task survival --backbone imagenet --folds 0,1,2
  - the full 4 backbones x 3 tasks x 10 folds matrix (120 runs) requires an
    explicit extra confirmation flag, since this is expensive:
        python run_cv.py --all --confirm-full-matrix
  - --dry-run prints the train.py commands without executing them.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.models import BACKBONE_NAMES

TASKS = ["classification", "regression", "survival"]

PASSTHROUGH_FLAGS = [
    "config",
    "splits_csv",
    "pretrained_dir",
    "results_dir",
    "image_size",
    "val_fraction",
    "lr",
    "weight_decay",
    "batch_size",
    "max_epochs",
    "patience",
    "num_workers",
    "lr_scheduler",
    "seed",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=TASKS, default=None)
    p.add_argument("--backbone", choices=BACKBONE_NAMES, default=None)
    p.add_argument("--folds", default=None, help="Comma-separated fold indices, e.g. 0,1,2. Default: all 10.")
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--all", action="store_true", help="Run the full task x backbone matrix")
    p.add_argument("--confirm-full-matrix", action="store_true", help="Required alongside --all to actually launch 120 runs")
    p.add_argument("--dry-run", action="store_true", help="Print the commands instead of running them")
    p.add_argument("--freeze-backbone", action="store_true", default=None)
    p.add_argument("--amp", action="store_true", default=None)
    p.add_argument("--smoke-test", action="store_true")

    for flag in PASSTHROUGH_FLAGS:
        p.add_argument(f"--{flag.replace('_', '-')}", default=None)

    return p.parse_args()


def build_command(task: str, backbone: str, fold: int, args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "train.py"), "--task", task, "--backbone", backbone, "--fold", str(fold)]
    for flag in PASSTHROUGH_FLAGS:
        value = getattr(args, flag)
        if value is not None:
            cmd += [f"--{flag.replace('_', '-')}", str(value)]
    if args.freeze_backbone:
        cmd.append("--freeze-backbone")
    if args.amp:
        cmd.append("--amp")
    if args.smoke_test:
        cmd.append("--smoke-test")
    return cmd


def main() -> None:
    args = parse_args()
    folds = [int(f) for f in args.folds.split(",")] if args.folds else list(range(args.n_folds))

    if args.all:
        if not args.confirm_full_matrix:
            print(
                "Refusing to launch the full task x backbone x fold matrix "
                f"({len(TASKS)} tasks x {len(BACKBONE_NAMES)} backbones x {len(folds)} folds = "
                f"{len(TASKS) * len(BACKBONE_NAMES) * len(folds)} runs) without --confirm-full-matrix.",
                file=sys.stderr,
            )
            sys.exit(1)
        combos = [(t, b) for t in TASKS for b in BACKBONE_NAMES]
    else:
        if not args.task or not args.backbone:
            print("Specify --task and --backbone (or use --all --confirm-full-matrix)", file=sys.stderr)
            sys.exit(1)
        combos = [(args.task, args.backbone)]

    commands = [build_command(t, b, f, args) for (t, b) in combos for f in folds]
    print(f"Planned runs: {len(commands)}")

    if args.dry_run:
        for cmd in commands:
            print(" ".join(cmd))
        return

    for i, cmd in enumerate(commands, 1):
        print(f"\n=== [{i}/{len(commands)}] {' '.join(cmd)} ===")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Command failed (exit={result.returncode}): {' '.join(cmd)}", file=sys.stderr)
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()

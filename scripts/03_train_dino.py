#!/usr/bin/env python3
"""DINO pre-training of the encoder of one level (see trace_mil/dino.py). Needs one GPU; resumes from its last epoch.

    python scripts/03_train_dino.py --level 2          # 18 epochs (paper); level 1: 6, level 0: 6, level 3: 15
    python scripts/03_train_dino.py --level 2 --debug  # ~40 cases, 100 steps, saves an augmentation grid
"""
import _repo  # noqa: F401
import argparse

from trace_mil.dino import EPOCHS, train
from trace_mil.layout import Layout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--level", type=int, choices=[0, 1, 2, 3], required=True)
    ap.add_argument("--epochs", type=int, default=None, help=f"default per level: {EPOCHS}")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4, help="base learning rate for batch 256 (scaled linearly)")
    ap.add_argument("--num-workers", type=int, default=64, help="tile reader threads")
    ap.add_argument("--time-budget-h", type=float, default=1e9, help="stop and checkpoint after this many hours")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    train(Layout(a.root), a.level, a.epochs, a.batch_size, lr=a.lr, num_workers=a.num_workers,
          time_budget_h=a.time_budget_h, seed=a.seed, max_steps=a.max_steps, debug=a.debug)


if __name__ == "__main__":
    main()

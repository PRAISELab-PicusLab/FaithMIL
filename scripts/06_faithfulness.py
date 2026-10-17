#!/usr/bin/env python3
"""Deletion / insertion tests of a trained run (see trace_mil/faithfulness.py). The run must be trained with --save-models.

    python scripts/06_faithfulness.py --run ctx3
"""
import _repo  # noqa: F401
import argparse

from trace_mil import faithfulness
from trace_mil.layout import Layout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-permutations", type=int, default=256, help="Monte-Carlo permutations of the Shapley values")
    ap.add_argument("--n-random", type=int, default=20, help="random orders of the reference")
    ap.add_argument("--max-cases", type=int, default=None, help="only the first n test patients of every fold")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    faithfulness.run(Layout(a.root), a.run, a.n_permutations, a.n_random, a.device, a.max_cases)


if __name__ == "__main__":
    main()

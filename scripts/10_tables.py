#!/usr/bin/env python3
"""Tables 1-3 of the paper from the run outputs (see trace_mil/tables.py and configs/paper_runs.json).

    python scripts/10_tables.py
"""
import _repo  # noqa: F401
import argparse
from pathlib import Path

from trace_mil import tables
from trace_mil.layout import Layout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "paper_runs.json"))
    a = ap.parse_args()
    tables.build(Layout(a.root), a.config)


if __name__ == "__main__":
    main()

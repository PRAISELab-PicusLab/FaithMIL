#!/usr/bin/env python3
"""PDF report of patients (see trace_mil/report.py). Needs scripts/08_explain.py.

    python scripts/09_make_reports.py --run ctx3_dino --cases case_radboud_0382 --figures
    python scripts/09_make_reports.py --run ctx3_dino                      # every patient with explanations

--figures also writes the attribution map and the strongest region as separate PDFs (figures of the paper).
"""
import _repo  # noqa: F401
import argparse

from trace_mil import report
from trace_mil.layout import Layout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--run", required=True)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--horizon", type=float, default=10.0, help="years of the restricted mean survival time")
    a = ap.parse_args()
    lay = Layout(a.root)
    rep = report.Reporter(lay, a.run, a.horizon)
    cases = a.cases or sorted(p.stem for p in rep.expl.glob("*.npz"))
    out = lay.report / "reports" / a.run
    for c in cases:
        print(report.make_report(rep, c, out, figures=a.figures), flush=True)


if __name__ == "__main__":
    main()

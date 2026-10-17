#!/usr/bin/env python3
"""Out-of-fold Shapley values of every patient, cohort patterns and their description (see trace_mil/report.py).
The run must be trained with --save-models, and scripts/07_ontology_concepts.py must have been run.

    python scripts/08_explain.py --run ctx3_dino
    python scripts/08_explain.py --run ctx3_dino --skip-explain    # refit only the patterns of existing explanations

report/patterns.json is written automatically (tissue class and ontology terms of every pattern) and can be edited by hand;
report/patterns/pattern_<k>.png shows the patches closest to every pattern to inspect.
"""
import _repo  # noqa: F401
import argparse

from trace_mil import report
from trace_mil.layout import Layout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--run", required=True)
    ap.add_argument("--encoder", default="dino", help="encoder whose L1 features are clustered into patterns")
    ap.add_argument("--k", type=int, default=8, help="number of patterns")
    ap.add_argument("--top-frac", type=float, default=0.02, help="share of the most raising / lowering patches per patient")
    ap.add_argument("--n-permutations", type=int, default=256)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--skip-explain", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    lay = Layout(a.root)
    if not a.skip_explain:
        report.explain(lay, a.run, a.n_permutations, device=a.device, max_cases=a.max_cases)
    report.fit_atlas(lay, a.run, a.encoder, a.k, a.top_frac)
    for p in report.describe_patterns(lay):
        print(p)
    report.save_prototypes(lay)


if __name__ == "__main__":
    main()

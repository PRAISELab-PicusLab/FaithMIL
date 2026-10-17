#!/usr/bin/env python3
"""Ontology terms and QuiltNet concepts of the report (see trace_mil/concepts.py).

Needs the QuiltNet image embeddings of the L1 tiles (scripts/04_extract_features.py --encoder quiltnet --view children)
and the two ontology files:
    NCI Thesaurus  Thesaurus.FLAT.zip   (https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/)
    UBERON         uberon-basic.obo     (https://obophenotype.github.io/uberon/)

    python scripts/07_ontology_concepts.py --ncit Thesaurus.FLAT.zip --ncit-version 26.09d --uberon uberon-basic.obo
"""
import _repo  # noqa: F401
import argparse
from pathlib import Path

from trace_mil import concepts
from trace_mil.layout import Layout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--ncit", required=True, help="Thesaurus.FLAT.zip")
    ap.add_argument("--ncit-version", default="unknown", help="version string of the NCI Thesaurus release")
    ap.add_argument("--uberon", required=True, help="uberon-basic.obo")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    lay = Layout(a.root)
    terms = concepts.build_terms(Path(a.ncit), Path(a.uberon), a.ncit_version, lay.report)
    pack = concepts.build(lay, terms, a.device)
    print(open(lay.report / "concept_validation.csv").read())


if __name__ == "__main__":
    main()

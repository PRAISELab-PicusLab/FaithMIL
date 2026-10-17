"""Directory layout of the data root.

Every script takes ``--root`` (default: ``$TRACE_DATA_ROOT`` or the current directory) and reads and writes
only below it:

    images/<case>.tif                     input: pyramidal whole-slide images (level k = 4**k x the pixel size of level 0)
    tissue-masks/<case>_tissue.tif        input: binary tissue masks
    metadata/cases.csv                    input: case_id, cohort, event, follow_up_years
    splits/folds.csv                      case_id, cohort, fold, event, follow_up_years
    anchors/tree/{parents,children}/      L2 tiles with tissue and their L1 sub-tiles (one CSV per case)
    anchors/grid_L<k>/                    tissue grid of level k (single-level models)
    tiles/L<k>/<case>.npy                 raw 224-px tiles for DINO pre-training
    encoders/dino_L<k>/                   DINO pre-training of level k (teacher_backbone.pth = the encoder)
    features/<encoder>/<view>/<case>.npy  frozen features (see FEATURE_VIEWS)
    runs/<run>/                           cross-validation runs
    faithfulness/<run>/                   deletion / insertion results
    report/                               cohort patterns, concepts, reports, figures
    tables/                               tables of the paper
"""
from __future__ import annotations

import os
from pathlib import Path

PATCH = 224  # tile side in pixels at every level

# a "view" is a set of tiles of one case, all encoded with the same encoder; its rows follow the anchor file
FEATURE_VIEWS = {
    "children": ("tree children", 1),   # L1 tiles of the tree          rows = anchors/tree/children/<case>.csv
    "parents": ("tree parents", 2),     # L2 tiles of the tree          rows = anchors/tree/parents/<case>.csv
    "ctx2": ("centred L2 context of every L1 child", 2),  # rows = anchors/tree/children/<case>.csv
    "ctx3": ("centred L3 context of every L1 child", 3),  # rows = anchors/tree/children/<case>.csv
    "grid0": ("L0 grid", 0),            # rows = anchors/grid_L0/<case>.csv
    "grid3": ("L3 grid", 3),            # rows = anchors/grid_L3/<case>.csv
}


class Layout:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or os.environ.get("TRACE_DATA_ROOT") or ".").resolve()

    def p(self, *parts) -> Path:
        return self.root.joinpath(*parts)

    @property
    def images(self) -> Path:
        return self.p("images")

    @property
    def masks(self) -> Path:
        return self.p("tissue-masks")

    @property
    def cases_csv(self) -> Path:
        return self.p("metadata", "cases.csv")

    @property
    def folds_csv(self) -> Path:
        return self.p("splits", "folds.csv")

    def slide(self, case: str) -> Path:
        return self.images / f"{case}.tif"

    def mask(self, case: str) -> Path:
        return self.masks / f"{case}_tissue.tif"

    def tree_anchors(self, role: str) -> Path:  # role: parents | children
        return self.p("anchors", "tree", role)

    def grid_anchors(self, level: int) -> Path:
        return self.p("anchors", f"grid_L{level}")

    def anchors_of_view(self, view: str) -> Path:
        if view in ("children", "ctx2", "ctx3"):
            return self.tree_anchors("children")
        if view == "parents":
            return self.tree_anchors("parents")
        return self.grid_anchors(FEATURE_VIEWS[view][1])

    def tiles(self, level: int) -> Path:
        return self.p("tiles", f"L{level}")

    def encoder(self, level: int) -> Path:
        return self.p("encoders", f"dino_L{level}")

    def features(self, encoder: str, view: str) -> Path:
        return self.p("features", encoder, view)

    def run(self, name: str) -> Path:
        return self.p("runs", name)

    def faithfulness(self, name: str) -> Path:
        return self.p("faithfulness", name)

    @property
    def report(self) -> Path:
        return self.p("report")

    @property
    def tables(self) -> Path:
        return self.p("tables")

#!/usr/bin/env python3
"""Train and evaluate one model with cross-validation (see trace_mil/train.py).

The model is defined by the views of its instances (their features are concatenated):

    L1 single level        --views children
    L2 single level        --views parents
    L3 / L0 single level   --views grid3  /  --views grid0 --pool 8192
    L1 + L2 context        --views children ctx2
    L1 + L2 + L3 context   --views children ctx2 ctx3

and, optionally, by the faithfulness-aware terms of the paper (lambda_s = lambda_n = 0.5, lambda_e = 0.3, q = 10%):

    --suff-cox 0.5 --nec-var 0.5 --ent 0.3 --topk 0.1 [--faith-start-epoch 10]

MIL baselines (C-index only, same protocol; see trace_mil/models.py):

    --arch mean --views children      --arch clam_sb --views children      --arch transmil --views parents

    python scripts/05_train.py --run-name ctx3 --views children ctx2 ctx3 --save-models
"""
import _repo  # noqa: F401
import argparse
from dataclasses import fields

from trace_mil.faith import FaithConfig
from trace_mil.layout import FEATURE_VIEWS, Layout
from trace_mil.train import TrainConfig, run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--arch", default="additive", choices=["additive", "mean", "clam_sb", "transmil"],
                    help="aggregator: the additive attention MIL of the paper or a MIL baseline")
    ap.add_argument("--views", nargs="+", default=["children"], choices=list(FEATURE_VIEWS))
    ap.add_argument("--encoder", default="dino", choices=["dino", "prost40m", "hoptimus0"])
    ap.add_argument("--max-groups", type=int, default=64, help="random L2 groups per patient and training step")
    ap.add_argument("--pool", type=int, default=None, help="fixed random pool of tiles per patient for training; "
                    "the test patients are scored on all their tiles (level 0)")
    ap.add_argument("--folds", default="all", help="'all' or a comma list, e.g. 0,1")
    ap.add_argument("--seed", type=int, default=42, help="base seed (the seeds of a fold are seed + 100 * fold + s)")
    ap.add_argument("--n-seeds", type=int, default=3, help="models per fold whose z-scored risks are averaged")
    ap.add_argument("--min-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--max-epochs", type=int, default=150)
    ap.add_argument("--save-models", action="store_true", help="keep the weights (needed for faithfulness and report)")
    ap.add_argument("--suff-cox", type=float, default=0.0, help="lambda_s: sufficiency of the top patches")
    ap.add_argument("--nec-var", type=float, default=0.0, help="lambda_n: necessity of the top patches")
    ap.add_argument("--ent", type=float, default=0.0, help="lambda_e: attention entropy")
    ap.add_argument("--topk", type=float, default=0.1, help="fraction q of the top patches by attention")
    ap.add_argument("--faith-start-epoch", type=int, default=0, help="Cox-only warm-up epochs")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    fc = FaithConfig(a.suff_cox, a.nec_var, a.ent, a.topk, a.faith_start_epoch)
    cfg = TrainConfig(run_name=a.run_name, arch=a.arch, views=a.views, encoder=a.encoder, max_groups=a.max_groups,
                      pool=a.pool, folds=a.folds, seed=a.seed, n_seeds=a.n_seeds, min_epochs=a.min_epochs, patience=a.patience,
                      max_epochs=a.max_epochs, save_models=a.save_models, faith=fc.asdict())
    run(Layout(a.root), cfg, a.device)


if __name__ == "__main__":
    main()

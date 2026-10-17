"""Seeding, logging, metrics and the inner train / validation split."""
from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(path: Path | str | None = None, name: str = "trace") -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, mode="w")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log


def c_index(time, risk, event) -> float:
    """Harrell's C-index; higher risk = shorter survival. NaN when no pair is comparable."""
    from lifelines.utils import concordance_index
    try:
        return float(concordance_index(np.asarray(time), -np.asarray(risk, dtype=float), np.asarray(event)))
    except ZeroDivisionError:
        return float("nan")


def cohort_c_index(frame: pd.DataFrame, risk) -> float:
    """C-index within each cohort, averaged with the number of events as weights (pairs across cohorts are never
    formed: follow-up differs strongly between cohorts, and a mixed C-index would reward separating them)."""
    num, den = 0.0, 0
    for _, g in frame.assign(risk=np.asarray(risk)).groupby("cohort"):
        ev = int(g.event.sum())
        if ev == 0 or len(g) < 2:
            continue
        c = c_index(g.follow_up_years.to_numpy(), g.risk.to_numpy(), g.event.to_numpy())
        if np.isfinite(c):
            num, den = num + c * ev, den + ev
    return num / den if den else float("nan")


def train_val_split(df: pd.DataFrame, val_fraction: float, seed: int):
    """Stratified (by event) split of the training folds; used only for model selection."""
    from sklearn.model_selection import train_test_split
    tr, va = train_test_split(df, test_size=val_fraction, random_state=seed, stratify=df["event"])
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def load_folds(layout) -> pd.DataFrame:
    df = pd.read_csv(layout.folds_csv)
    need = {"case_id", "cohort", "fold", "event", "follow_up_years"}
    if need - set(df.columns):
        raise ValueError(f"{layout.folds_csv}: missing columns {need - set(df.columns)} (run scripts/00_prepare_data.py)")
    return df

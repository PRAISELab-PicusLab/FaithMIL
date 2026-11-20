"""Cross-validated training of the additive attention MIL (the protocol of the paper).

For every outer fold: 15% of the training patients (stratified by event) are held out for model selection; `n_seeds`
models are trained with different initialisations and bag samplings; the checkpoint of a model is the epoch at the peak
of the 5-epoch moving average of the validation C-index; training stops after `patience` epochs without improvement (not before `min_epochs`). The risk of
each model is z-scored with the mean and standard deviation of its own training risks and the z-scores are averaged
(the average of sums over patches is again a sum over patches, so the decomposition is preserved).

    runs/<run>/config.json, run_log.txt, fold_metrics.csv, summary.json
    runs/<run>/fold_<k>/test_predictions.csv          ensemble risk of the test patients (column risk_score)
                        test_predictions_seed<s>.csv, history_seed<s>.csv, seeds.csv, metrics.json
                        model_seed<s>.pt, fit_risks.csv      (with save_models: the weights and the ensemble risks of the
                                                              training / validation patients, needed for the report)
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .bank import assemble, load_bank, pool_rows
from .faith import FaithConfig, faith_loss
from .layout import Layout
from .losses import cox_ph_loss
from .models import BASELINES, AttentionMIL, build_model
from .utils import c_index, cohort_c_index, get_logger, load_folds, set_seed, train_val_split


@dataclass
class TrainConfig:
    run_name: str
    arch: str = "additive"        # additive attention MIL (ours) or a MIL baseline: mean, clam_sb, transmil
    views: list = field(default_factory=lambda: ["children"])
    encoder: str = "dino"
    max_groups: int = 64          # random groups (L2 tiles) per patient and step during training
    pool: int | None = None       # fixed random pool of tiles per patient for training (L0); testing uses all tiles
    hidden_dim: int = 64
    attention_dim: int = 32
    dropout: float = 0.25
    input_dropout: float = 0.25
    lr: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 64
    warmup_epochs: int = 3
    val_fraction: float = 0.15
    seed: int = 42
    min_epochs: int = 30
    patience: int = 30
    max_epochs: int = 150
    ma_window: int = 5
    n_seeds: int = 3
    faith: dict = field(default_factory=lambda: FaithConfig().asdict())
    folds: str = "all"
    save_models: bool = False


@torch.no_grad()
def predict(model: AttentionMIL, bank, cases, device) -> np.ndarray:
    """Risk of every case, from all its instances."""
    model.eval()
    out = []
    for case in cases:
        X, seg = assemble(bank, [case], None, None, device)
        out.append(float(model(X, seg, 1)[0][0]))
    return np.array(out)


@torch.no_grad()
def predict_streaming(model: AttentionMIL, feats, device, chunk: int = 65536) -> float:
    """Risk of one patient from a (memory-mapped) feature array that is read in row chunks (single-view models)."""
    model.eval()
    g, s = [], []
    for a in range(0, feats.shape[0], chunk):
        x = torch.from_numpy(np.ascontiguousarray(feats[a:a + chunk])).to(device)
        h = model.embed(x)
        g.append(model.gate(h))
        s.append(model.score(h).squeeze(-1))
    w = torch.softmax(torch.cat(g), 0)
    return float((w * torch.cat(s)).sum())


def train_seed(cfg: TrainConfig, bank, train_df, val_df, device, logger, seed: int, dims):
    set_seed(seed)
    fc = FaithConfig(**cfg.faith)
    model = build_model({**asdict(cfg), "dims": dims}).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ids = train_df["case_id"].to_numpy()
    t = torch.tensor(train_df["follow_up_years"].to_numpy(np.float32), device=device)
    e = torch.tensor(train_df["event"].to_numpy(np.float32), device=device)
    n_batches = max(1, int(round(len(ids) / cfg.batch_size)))
    total, warm = cfg.max_epochs * n_batches, cfg.warmup_epochs * n_batches
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: (k + 1) / warm if k < warm else
                                              0.5 * (1 + math.cos(math.pi * min(1.0, (k - warm) / max(1, total - warm)))))
    gen, rng = torch.Generator().manual_seed(seed), np.random.default_rng(seed)
    vt = torch.tensor(val_df.follow_up_years.to_numpy(np.float32))
    ve = torch.tensor(val_df.event.to_numpy(np.float32))

    hist, window = [], []
    best_ma, best_state, best_epoch, bad = -np.inf, None, -1, 0
    for epoch in range(cfg.max_epochs):
        model.train()
        t0, losses = time.time(), []
        faith_on = fc.any and epoch >= fc.start_epoch
        for bi in np.array_split(rng.permutation(len(ids)), n_batches):
            X, seg = assemble(bank, ids[bi], cfg.max_groups, gen, device)
            risk, out = model(X, seg, len(bi))
            idx = torch.as_tensor(bi, device=device)
            loss = cox_ph_loss(risk, t[idx], e[idx])
            if "inst_loss" in out:  # CLAM-SB instance clustering term
                loss = loss + out["inst_loss"]
            if faith_on:
                loss = loss + faith_loss(fc, out["g"], out["s"], out["w"], seg, risk, len(bi), t[idx], e[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        vrisk = predict(model, bank, val_df["case_id"], device)
        vc = cohort_c_index(val_df, vrisk) if epoch >= fc.start_epoch else float("nan")  # warm-up: not selectable
        vloss = float(cox_ph_loss(torch.tensor(vrisk), vt, ve))
        window = (window + [vc])[-cfg.ma_window:]
        with np.errstate(all="ignore"):
            ma = float(np.nanmean(window)) if len(window) == cfg.ma_window and not np.isnan(window).all() else -np.inf
        improved = ma > best_ma + 1e-4
        if improved:
            best_ma, best_state, best_epoch, bad = ma, copy.deepcopy(model.state_dict()), epoch, 0
        elif epoch >= cfg.ma_window - 1:
            bad += 1
        hist.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": vloss, "val_c_cohort": vc,
                     "val_c_ma": ma if np.isfinite(ma) else np.nan, "is_best": improved, "seconds": time.time() - t0})
        if epoch % 10 == 0 or improved:
            logger.info(f"  seed {seed} ep {epoch:03d} train {hist[-1]['train_loss']:.4f} val_loss {vloss:.4f} "
                        f"val_c {vc:.3f} ma {ma:.3f} best={improved}")
        if epoch + 1 >= cfg.min_epochs and bad >= cfg.patience:
            break
    if best_state is None:  # no epoch had a full moving-average window with a defined C-index
        best_state, best_epoch = copy.deepcopy(model.state_dict()), len(hist) - 1
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist), best_epoch, best_ma


def run(layout: Layout, cfg: TrainConfig, device=None) -> Path:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = layout.run(cfg.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(run_dir / "run_log.txt", cfg.run_name)
    df = load_folds(layout)[["case_id", "cohort", "fold", "event", "follow_up_years"]]
    folds = sorted(df["fold"].unique()) if cfg.folds == "all" else [int(f) for f in str(cfg.folds).split(",")]
    bank, dims = load_bank(layout, df["case_id"].tolist(), cfg.encoder, cfg.views, cfg.pool, logger=logger)
    if cfg.pool and len(cfg.views) != 1:
        raise ValueError("a training pool is only supported for single-view models")
    if cfg.arch != "additive":
        if list(cfg.views) != BASELINES[cfg.arch] or cfg.pool or FaithConfig(**cfg.faith).any:
            raise ValueError(f"{cfg.arch} needs --views {' '.join(BASELINES[cfg.arch])}, no pool and no faithfulness terms")
    (run_dir / "config.json").write_text(json.dumps({**asdict(cfg), "dims": dims}, indent=2))
    logger.info(f"config={asdict(cfg)}")

    rows = []
    for fold in folds:
        test_df = df[df["fold"] == fold].reset_index(drop=True)
        train_df, val_df = train_val_split(df[df["fold"] != fold].reset_index(drop=True), cfg.val_fraction, cfg.seed)
        logger.info(f"fold {fold}: train {len(train_df)} ({int(train_df.event.sum())} ev) val {len(val_df)} "
                    f"({int(val_df.event.sum())} ev) test {len(test_df)} ({int(test_df.event.sum())} ev)")
        fdir = run_dir / f"fold_{fold}"
        fdir.mkdir(exist_ok=True)
        fit_df = pd.concat([train_df.assign(split="train"), val_df.assign(split="val")], ignore_index=True)
        z_test, z_fit, seed_rows, members = [], [], [], []
        for s in range(cfg.n_seeds):
            seed = cfg.seed + 100 * int(fold) + s
            model, hist, best_epoch, best_ma = train_seed(cfg, bank, train_df, val_df, device, logger, seed, dims)
            hist.to_csv(fdir / f"history_seed{s}.csv", index=False)
            tr_risk = predict(model, bank, train_df["case_id"], device)
            mu, sd = float(tr_risk.mean()), float(tr_risk.std()) or 1.0
            if cfg.pool:  # test patients are scored on ALL their tiles, streamed from disk
                feats = lambda c: np.load(layout.features(cfg.encoder, cfg.views[0]) / f"{c}.npy", mmap_mode="r")  # noqa: E731
                te_risk = np.array([predict_streaming(model, feats(c), device) for c in test_df["case_id"]])
            else:
                te_risk = predict(model, bank, test_df["case_id"], device)
            z_test.append((te_risk - mu) / sd)
            if cfg.save_models:
                va_risk = predict(model, bank, val_df["case_id"], device)
                z_fit.append((np.concatenate([tr_risk, va_risk]) - mu) / sd)
                torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "mu": mu, "sd": sd,
                            "seed": seed, "best_epoch": best_epoch, "dims": dims}, fdir / f"model_seed{s}.pt")
            seed_rows.append({"seed": seed, "best_epoch": best_epoch, "epochs_run": len(hist), "val_c_ma_best": best_ma,
                              "test_c_index_cohort": cohort_c_index(test_df, te_risk)})
            pd.DataFrame({"case_id": test_df["case_id"], "fold": fold, "event": test_df.event,
                          "follow_up_years": test_df.follow_up_years, "risk_score": te_risk,
                          "risk_z": z_test[-1]}).to_csv(fdir / f"test_predictions_seed{s}.csv", index=False)
            logger.info(f"fold {fold} seed {s}: best epoch {best_epoch} ({len(hist)} run), val C ma {best_ma:.3f}, "
                        f"test C (per cohort) {seed_rows[-1]['test_c_index_cohort']:.3f}")
        ens = np.mean(z_test, axis=0)
        if cfg.save_models:
            fit_df.assign(risk_score=np.mean(z_fit, axis=0))[
                ["case_id", "split", "cohort", "event", "follow_up_years", "risk_score"]].to_csv(fdir / "fit_risks.csv", index=False)
        pd.DataFrame({"case_id": test_df["case_id"], "fold": fold, "event": test_df.event,
                      "follow_up_years": test_df.follow_up_years, "risk_score": ens}).to_csv(
            fdir / "test_predictions.csv", index=False)
        pd.DataFrame(seed_rows).to_csv(fdir / "seeds.csv", index=False)
        m = {"fold": int(fold), "test_c_index": c_index(test_df.follow_up_years, ens, test_df.event),
             "test_c_index_cohort": cohort_c_index(test_df, ens),
             "seed_test_c_index_cohort": [r["test_c_index_cohort"] for r in seed_rows],
             "best_epochs": [r["best_epoch"] for r in seed_rows], "epochs_run": [r["epochs_run"] for r in seed_rows]}
        (fdir / "metrics.json").write_text(json.dumps(m, indent=2))
        rows.append(m)
        logger.info(f"FOLD {fold} ENSEMBLE test C {m['test_c_index']:.4f} (per cohort {m['test_c_index_cohort']:.4f})")

    fm = pd.DataFrame(rows)
    fm.to_csv(run_dir / "fold_metrics.csv", index=False)
    summary = {"run": cfg.run_name, "n_folds": len(rows), "fold_c_index_cohort_mean": float(np.nanmean(fm.test_c_index_cohort)),
               "fold_c_index_mean": float(np.nanmean(fm.test_c_index))}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"SUMMARY {json.dumps(summary)}")
    return run_dir

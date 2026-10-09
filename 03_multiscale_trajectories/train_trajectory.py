#!/usr/bin/env python3
"""Train the multi-scale trajectory-MIL survival model on cached features.

Encoders are frozen (features cached by extract_trajectory_features.py); what
is trained here is ~0.3-1.5M parameters: per-level adapters, the trajectory
aggregator, the local score head and the reliability attention. An epoch
takes seconds, so the full 10-fold CV stays affordable.

Examples:
    python train_trajectory.py --fold 0                       # one outer CV fold
    python train_trajectory.py --folds all                    # all 10 folds, features loaded once
    python train_trajectory.py --split-mode holdout           # folds 0-6 / 7 / 8-9
    python train_trajectory.py --folds all --levels 5 4 3 --aggregator gru
    python train_trajectory.py --folds all --levels 5 4 3 2 1 0 --backbone mocov2 \
        --level-backbones 0:prost40m,1:prost40m
"""
from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.dataset import make_train_val_split
from src.losses import cox_ph_loss
from src.metrics import survival_metrics
from src.trajectory.bank import FeatureBank
from src.trajectory.explain import attention_contributions, level_occlusion, shapley_trajectories
from src.trajectory.io import features_dir
from src.trajectory.model import build_trajectory_model
from src.trajectory.survival import assign_risk_group, fit_breslow, risk_group_thresholds
from src.utils import EarlyStopping, count_parameters, get_device, save_config, save_json, set_global_seed, setup_logger


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(Path(__file__).resolve().parent / "configs" / "trajectory.yaml"))
    p.add_argument("--fold", type=int, default=None, help="one outer CV fold (cv mode)")
    p.add_argument("--folds", default=None, help="'all' or comma-separated folds, trained sequentially")
    p.add_argument("--split-mode", choices=["cv", "holdout"], default=None)
    p.add_argument("--levels", type=int, nargs="+", default=None)
    p.add_argument("--backbone", default=None)
    p.add_argument("--level-backbones", default=None, help="e.g. 0:prost40m,1:prost40m")
    p.add_argument("--anchor-set", default=None)
    p.add_argument("--traj-root", default=None)
    p.add_argument("--aggregator", choices=["mlp", "transformer", "gru", "lstm"], default=None)
    p.add_argument("--pooling", choices=["attention", "mean"], default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--bag-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, help="patients per step; 0 = whole training set")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--monitor", choices=["val_loss", "val_cindex"], default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--results-dir", default="results/trajectory")
    p.add_argument("--only-available", action="store_true",
                   help="restrict to cases whose features exist (staged/smoke runs); logged loudly")
    p.add_argument("--smoke-test", action="store_true", help="3 epochs, for pipeline checks only")
    return p.parse_args()


def build_config(args) -> dict:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    d, m, t, s = cfg["data"], cfg["model"], cfg["train"], cfg["split"]
    for key, section, name in [
        ("levels", d, "levels"), ("backbone", d, "backbone"), ("anchor_set", d, "anchor_set"),
        ("traj_root", d, "traj_root"), ("aggregator", m, "aggregator"), ("pooling", m, "pooling"),
        ("hidden_dim", m, "hidden_dim"), ("n_layers", m, "n_layers"), ("dropout", m, "dropout"),
        ("bag_size", t, "bag_size"), ("max_epochs", t, "max_epochs"), ("lr", t, "lr"),
        ("weight_decay", t, "weight_decay"), ("patience", t, "early_stopping_patience"),
        ("monitor", t, "monitor"), ("split_mode", s, "mode"),
    ]:
        value = getattr(args, key)
        if value is not None:
            section[name] = value
    if args.batch_size is not None:
        t["batch_size"] = args.batch_size or None
    if args.level_backbones:
        d["level_backbones"] = {int(k): v for k, v in (x.split(":") for x in args.level_backbones.split(","))}
    if args.seed is not None:
        cfg["seed"] = args.seed
    d["levels"] = sorted({int(l) for l in d["levels"]}, reverse=True)
    d["level_backbones"] = {int(k): v for k, v in (d.get("level_backbones") or {}).items()}
    if args.smoke_test:
        t["max_epochs"] = 3
        cfg["outputs"]["shapley_permutations"] = 16
        cfg["smoke_test"] = True

    if args.run_name:
        run = args.run_name
    else:
        tag = d["backbone"] + ("-mix" if d["level_backbones"] else "")
        run = f"{tag}_{m['aggregator']}_{m['pooling']}_L{''.join(map(str, d['levels']))}"
    cfg["run_name"] = run
    return cfg


def level_backbones_of(cfg) -> dict[int, str]:
    d = cfg["data"]
    return {l: d["level_backbones"].get(l, d["backbone"]) for l in d["levels"]}


def split_frames(df: pd.DataFrame, cfg: dict, fold: int | None):
    s = cfg["split"]
    if s["mode"] == "holdout":
        pick = lambda folds: df[df["fold"].isin(folds)].reset_index(drop=True)  # noqa: E731
        return pick(s["holdout_train_folds"]), pick(s["holdout_val_folds"]), pick(s["holdout_test_folds"])
    test = df[df["fold"] == fold].reset_index(drop=True)
    train, val = make_train_val_split(df[df["fold"] != fold].reset_index(drop=True), s["val_fraction"], cfg["seed"])
    return train, val, test


@torch.no_grad()
def predict(model, bank: FeatureBank, case_ids, device, chunk: int) -> dict:
    model.eval()
    out = {}
    for c in case_ids:
        s, g = model.trajectory_outputs(bank.bag(c, device), chunk)
        mask = torch.ones_like(s, dtype=torch.bool)
        risk, w = model.pool(s[None], g[None], mask[None])
        out[c] = {"risk": float(risk), "scores": s.cpu().numpy(), "logits": g.cpu().numpy(), "weights": w[0].cpu().numpy()}
    return out


def eval_split(model, bank, frame, device, chunk):
    preds = predict(model, bank, frame["case_id"], device, chunk)
    risk = np.array([preds[c]["risk"] for c in frame["case_id"]])
    time_ = frame["follow_up_years"].to_numpy(np.float32)
    event = frame["event"].to_numpy(np.float32)
    loss = float(cox_ph_loss(torch.tensor(risk), torch.tensor(time_), torch.tensor(event)))
    cidx = survival_metrics(time_, risk, event)["c_index"]
    return preds, risk, loss, cidx


def make_scheduler(optimizer, cfg, steps_per_epoch):
    t = cfg["train"]
    if t["scheduler"] == "none":
        return None
    total = max(1, t["max_epochs"] * steps_per_epoch)
    warm = t["warmup_epochs"] * steps_per_epoch

    def factor(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_one(cfg, bank, train_df, val_df, device, logger, seed):
    t = cfg["train"]
    model = build_trajectory_model(bank.level_dims, cfg).to(device)
    logger.info(f"model: {count_parameters(model)} | aggregator={cfg['model']['aggregator']} "
                f"pooling={cfg['model']['pooling']} levels={model.levels}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])

    ids = train_df["case_id"].to_numpy()
    times = torch.tensor(train_df["follow_up_years"].to_numpy(np.float32), device=device)
    events = torch.tensor(train_df["event"].to_numpy(np.float32), device=device)
    bs = t["batch_size"] or len(ids)
    n_batches = max(1, int(round(len(ids) / bs)))
    scheduler = make_scheduler(optimizer, cfg, n_batches)
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    monitor = t["monitor"]
    stopper = EarlyStopping(patience=t["early_stopping_patience"], mode="min" if monitor == "val_loss" else "max")

    best_state, best_epoch, history = None, -1, []
    for epoch in range(t["max_epochs"]):
        model.train()
        t0 = time.time()
        losses = []
        # equal-sized batches: no tiny last batch with an ill-defined risk set
        for batch_idx in np.array_split(rng.permutation(len(ids)), n_batches):
            feats, mask = bank.sample_batch(ids[batch_idx], t["bag_size"], gen, device)
            out = model(feats, mask)
            bi = torch.as_tensor(batch_idx, device=device)
            loss = cox_ph_loss(out["risk"], times[bi], events[bi])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if t["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(loss.item())

        _, _, val_loss, val_c = eval_split(model, bank, val_df, device, t["eval_chunk"])
        score = val_loss if monitor == "val_loss" else (val_c if np.isfinite(val_c) else -np.inf)
        is_best = stopper.step(score)
        if is_best:
            best_state, best_epoch = copy.deepcopy(model.state_dict()), epoch
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "val_c_index": val_c,
               "lr": optimizer.param_groups[0]["lr"], "is_best": is_best, "seconds": time.time() - t0}
        history.append(row)
        logger.info(f"epoch {epoch:03d} | train_loss={row['train_loss']:.4f} val_loss={val_loss:.4f} "
                    f"val_c={val_c:.3f} lr={row['lr']:.2e} best={is_best} ({row['seconds']:.1f}s)")
        if stopper.should_stop:
            logger.info(f"early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch


def run_split(cfg, bank, df, fold, device, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / "log.txt", name=f"traj_{out_dir.name}")
    seed = cfg["seed"] + (fold or 0)
    set_global_seed(seed, deterministic=cfg["deterministic"])
    train_df, val_df, test_df = split_frames(df, cfg, fold)
    for name, f in [("train", train_df), ("val", val_df), ("test", test_df)]:
        logger.info(f"{name}: {len(f)} cases, {int(f['event'].sum())} events")

    model, history, best_epoch = train_one(cfg, bank, train_df, val_df, device, logger, seed)
    t = cfg["train"]
    means = bank.level_means(train_df["case_id"])
    for l, m in means.items():
        getattr(model, f"level_mean_L{l}").copy_(m.to(device))

    splits = {"train": train_df, "val": val_df, "test": test_df}
    results, rows = {}, []
    for name, frame in splits.items():
        preds, risk, loss, cidx = eval_split(model, bank, frame, device, t["eval_chunk"])
        results[name] = {"preds": preds, "risk": risk, "loss": loss, "c_index": cidx}

    o = cfg["outputs"]
    tr = results["train"]
    breslow = fit_breslow(tr["risk"], train_df["follow_up_years"], train_df["event"], o["rmst_horizon_years"])
    thresholds = risk_group_thresholds(breslow.rmst(tr["risk"]), o["risk_group_quantiles"])
    baseline = float(tr["risk"].mean())

    metrics = {"best_epoch": best_epoch, "epochs_run": len(history)}
    for name, frame in splits.items():
        risk = results[name]["risk"]
        pred_time = breslow.rmst(risk)
        weights = [results[name]["preds"][c]["weights"] for c in frame["case_id"]]
        rows.append(pd.DataFrame({
            "case_id": frame["case_id"], "fold": fold if fold is not None else frame["fold"], "split": name,
            "event": frame["event"], "follow_up_years": frame["follow_up_years"],
            "risk_score": risk, "predicted_time_years": pred_time, "median_time_years": breslow.median_time(risk),
            "risk_group": assign_risk_group(pred_time, thresholds),
            "n_trajectories": [len(w) for w in weights],
            "attention_eff_n": [1.0 / float(np.sum(w ** 2)) for w in weights],
        }))
        metrics[f"{name}_c_index"] = results[name]["c_index"]
        metrics[f"{name}_loss"] = results[name]["loss"]
        # C-index of the time prediction (higher time = lower risk): must equal the risk C-index
        metrics[f"{name}_c_index_from_time"] = survival_metrics(
            frame["follow_up_years"].to_numpy(), -pred_time, frame["event"].to_numpy())["c_index"]

    preds_df = pd.concat(rows, ignore_index=True)
    preds_df.to_csv(out_dir / "all_predictions.csv", index=False)
    preds_df[preds_df["split"] == "test"].to_csv(out_dir / "test_predictions.csv", index=False)
    history.to_csv(out_dir / "history.csv", index=False)

    # explanations for the requested splits (per-case npz, used by make_challenge_outputs.py)
    exp_dir = out_dir / "trajectory_outputs"
    exp_dir.mkdir(exist_ok=True)
    level_rows = []
    for name in o["explain_splits"]:
        for c in splits[name]["case_id"]:
            p = results[name]["preds"][c]
            phi = shapley_trajectories(p["scores"], p["logits"], baseline, o["shapley_permutations"],
                                       seed=seed, pooling=cfg["model"]["pooling"])
            occl = level_occlusion(model, bank.bag(c, device), t["eval_chunk"])
            np.savez_compressed(exp_dir / f"{c}.npz", scores=p["scores"], logits=p["logits"], weights=p["weights"],
                                shapley=phi, contributions=attention_contributions(p["scores"], p["weights"], baseline),
                                risk=p["risk"], baseline=baseline,
                                occlusion_levels=np.array(list(occl)), occlusion_delta=np.array(list(occl.values())))
            level_rows.append({"case_id": c, "split": name, "risk": p["risk"],
                               **{f"delta_L{l}": v for l, v in occl.items()}})
    pd.DataFrame(level_rows).to_csv(out_dir / "level_occlusion.csv", index=False)

    torch.save({
        "state_dict": model.state_dict(), "level_dims": bank.level_dims, "levels": model.levels,
        "level_backbones": level_backbones_of(cfg), "config": cfg, "breslow": breslow.to_dict(),
        "risk_group_thresholds": thresholds, "baseline_risk": baseline,
        "train_risks": np.sort(tr["risk"]).tolist(), "fold": fold,
        "n_train_cases": len(train_df), "n_train_events": int(train_df["event"].sum()),
    }, out_dir / "model.pt")
    save_json(metrics, out_dir / "metrics.json")
    save_config(cfg, out_dir / "config.yaml")
    logger.info(f"TEST c_index={metrics['test_c_index']:.4f} (from time: {metrics['test_c_index_from_time']:.4f}) | "
                f"val {metrics['val_c_index']:.4f} | train {metrics['train_c_index']:.4f} | saved to {out_dir}")
    return metrics


def main():
    args = parse_args()
    cfg = build_config(args)
    device = get_device()
    run_dir = Path(args.results_dir) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir / "run_log.txt", name="traj_run")
    logger.info(f"run={cfg['run_name']} device={device}\nconfig={cfg}")

    df = pd.read_csv(cfg["data"]["splits_csv"])[["case_id", "fold", "event", "follow_up_years", "thumbnail_level"]]
    lb = level_backbones_of(cfg)
    d = cfg["data"]
    if args.only_available:
        have = df["case_id"].map(lambda c: all(
            (features_dir(d["traj_root"], d["anchor_set"], lb[l], l) / f"{c}.npy").exists() for l in d["levels"]))
        logger.warning(f"--only-available: using {int(have.sum())}/{len(df)} cases with features for all levels "
                       "(NOT a valid benchmark unless this is all 508)")
        df = df[have].reset_index(drop=True)
        cfg["only_available_n_cases"] = int(have.sum())

    bank = FeatureBank(d["traj_root"], d["anchor_set"], df["case_id"], d["levels"], lb, logger)

    if cfg["split"]["mode"] == "holdout":
        run_split(cfg, bank, df, None, device, run_dir / "holdout")
        return
    if args.folds:
        folds = sorted(df["fold"].unique()) if args.folds == "all" else [int(f) for f in args.folds.split(",")]
    elif args.fold is not None:
        folds = [args.fold]
    else:
        raise SystemExit("cv mode needs --fold k or --folds all|k1,k2,...")
    for fold in folds:
        run_split(cfg, bank, df, int(fold), device, run_dir / f"fold_{fold}")


if __name__ == "__main__":
    main()

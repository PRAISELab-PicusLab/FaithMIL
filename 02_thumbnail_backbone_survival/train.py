#!/usr/bin/env python3
"""Train ResNet50 on ONE (task, backbone, fold) combination.

Fully offline: reads only local thumbnails/, splits/folds.csv and
pretrained/*.pt(orch) files. Run scripts/download_pretrained_models.py from
the login node first, and scripts/create_folds.py once before any of this.

Examples:
    python train.py --task classification --backbone barlow_twins --fold 0
    python train.py --task regression      --backbone imagenet     --fold 3
    python train.py --task survival        --backbone swav         --fold 7
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from src import metrics as metrics_mod
from src.dataset import (
    LeopardThumbnailDataset,
    input_size_for_backbone,
    load_folds,
    make_train_val_split,
)
from src.losses import get_loss
from src.models import BACKBONE_NAMES, build_model
from src.utils import (
    EarlyStopping,
    count_parameters,
    get_device,
    save_config,
    save_json,
    set_global_seed,
    setup_logger,
)

TASKS = ["classification", "regression", "survival"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--backbone", required=True, choices=BACKBONE_NAMES)
    p.add_argument("--fold", required=True, type=int)

    p.add_argument("--config", default=str(Path(__file__).resolve().parent / "configs" / "default.yaml"))
    p.add_argument("--splits-csv", default=None)
    p.add_argument("--pretrained-dir", default="pretrained")
    p.add_argument("--results-dir", default="results")

    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--val-fraction", type=float, default=None)
    p.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true", default=None)
    p.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    p.add_argument("--amp", dest="amp", action="store_true", default=None,
                   help="fp16 autocast for the backbone forward (recommended for the large ViTs)")

    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--lr-scheduler", choices=["plateau", "cosine", "none"], default=None)

    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--smoke-test", action="store_true", help="Cap epochs/data for a fast pipeline sanity check")
    return p.parse_args()


def build_effective_config(args: argparse.Namespace) -> dict:
    import yaml

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.splits_csv is not None:
        config["data"]["splits_csv"] = args.splits_csv
    if args.image_size is not None:
        config["data"]["image_size"] = args.image_size
    if args.val_fraction is not None:
        config["data"]["val_fraction"] = args.val_fraction
    if args.freeze_backbone is not None:
        config["model"]["freeze_backbone"] = args.freeze_backbone
    if args.amp is not None:
        config["model"]["amp"] = args.amp
    if args.lr is not None:
        config["optim"]["lr"] = args.lr
    if args.weight_decay is not None:
        config["optim"]["weight_decay"] = args.weight_decay
    if args.batch_size is not None:
        config["optim"]["batch_size"] = args.batch_size
    if args.max_epochs is not None:
        config["optim"]["max_epochs"] = args.max_epochs
    if args.patience is not None:
        config["optim"]["early_stopping_patience"] = args.patience
    if args.num_workers is not None:
        config["optim"]["num_workers"] = args.num_workers
    if args.lr_scheduler is not None:
        config["optim"]["lr_scheduler"] = args.lr_scheduler
    if args.seed is not None:
        config["seed"] = args.seed

    config["task"] = args.task
    config["backbone"] = args.backbone
    config["fold"] = args.fold
    config["pretrained_dir"] = args.pretrained_dir
    config["results_dir"] = args.results_dir

    if args.smoke_test:
        config["optim"]["max_epochs"] = min(2, config["optim"]["max_epochs"])
        config["optim"]["early_stopping_patience"] = 100
        config["smoke_test"] = True
    return config


def _worker_init_fn(worker_id: int) -> None:
    # DataLoader workers are forked and otherwise inherit the parent's numpy
    # RNG state verbatim, which would make augmentations correlated across
    # workers. torch already reseeds its own generator per worker; mirror
    # that for numpy (used by our augmentations in src/dataset.py).
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)


def build_loader(df: pd.DataFrame, image_size: int, backbone: str, train: bool, batch_size: int, num_workers: int) -> DataLoader:
    ds = LeopardThumbnailDataset(df, image_size=image_size, backbone=backbone, train=train)
    return DataLoader(
        ds,
        batch_size=max(1, min(batch_size, len(ds))),
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )


def forward_and_loss(model, loss_fn, task, batch, device):
    images = batch["image"].to(device, non_blocking=True)
    event = batch["event"].to(device, non_blocking=True)
    follow_up = batch["follow_up_years"].to(device, non_blocking=True)
    output = model(images)

    if task == "classification":
        loss = loss_fn(output, event)
    elif task == "regression":
        loss = loss_fn(output, follow_up)
    else:
        loss = loss_fn(output, follow_up, event)
    return output, event, follow_up, loss


def run_epoch(model, loader, loss_fn, task, device, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, n_seen = 0.0, 0
    case_ids, outputs, events, follow_ups = [], [], [], []

    torch.set_grad_enabled(train_mode)
    for batch in loader:
        output, event, follow_up, loss = forward_and_loss(model, loss_fn, task, batch, device)
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = output.shape[0]
        total_loss += loss.item() * bs
        n_seen += bs
        case_ids.extend(batch["case_id"])
        outputs.append(output.detach().cpu().numpy())
        events.append(event.detach().cpu().numpy())
        follow_ups.append(follow_up.detach().cpu().numpy())
    torch.set_grad_enabled(True)

    avg_loss = total_loss / max(n_seen, 1)
    result = {
        "case_id": case_ids,
        "output": np.concatenate(outputs) if outputs else np.array([]),
        "event": np.concatenate(events) if events else np.array([]),
        "follow_up_years": np.concatenate(follow_ups) if follow_ups else np.array([]),
    }
    return avg_loss, result


def compute_task_metrics(task: str, result: dict) -> dict:
    if task == "classification":
        prob = expit(result["output"])
        return metrics_mod.classification_metrics(result["event"], prob)
    if task == "regression":
        return metrics_mod.regression_metrics(result["follow_up_years"], result["output"])
    return metrics_mod.survival_metrics(result["follow_up_years"], result["output"], result["event"])


def save_predictions(task: str, fold: int, result: dict, out_path: Path) -> None:
    case_id, event, follow_up, raw = result["case_id"], result["event"], result["follow_up_years"], result["output"]
    if task == "classification":
        prob = expit(raw)
        df = pd.DataFrame(
            {
                "case_id": case_id,
                "fold": fold,
                "target": event,
                "prediction": prob,
                "event": event,
                "probability": prob,
                "predicted_class": (prob >= 0.5).astype(int),
            }
        )
    elif task == "regression":
        df = pd.DataFrame(
            {
                "case_id": case_id,
                "fold": fold,
                "target": follow_up,
                "prediction": raw,
                "follow_up_years": follow_up,
                "predicted_follow_up_years": raw,
                "event": event,
            }
        )
    else:
        df = pd.DataFrame(
            {
                "case_id": case_id,
                "fold": fold,
                "target": follow_up,
                "prediction": raw,
                "follow_up_years": follow_up,
                "event": event,
                "risk_score": raw,
            }
        )
    df.to_csv(out_path, index=False)


def save_checkpoint(model, path: Path) -> None:
    """A frozen backbone never changes, so only the head is saved (the
    backbone is rebuilt from pretrained/ on load). Avoids rewriting multi-GB
    ViT-g/ViT-H weights at every best epoch.
    """
    state = model.state_dict()
    if model.freeze_backbone:
        state = {k: v for k, v in state.items() if k.startswith("head.")}
    torch.save(state, path)


def load_checkpoint(model, path: Path, device) -> None:
    state = torch.load(path, map_location=device, weights_only=True)
    result = model.load_state_dict(state, strict=not model.freeze_backbone)
    if model.freeze_backbone and (result.unexpected_keys or any(k.startswith("head.") for k in result.missing_keys)):
        raise RuntimeError(f"head checkpoint mismatch: {result}")


def build_scheduler(optimizer, config):
    kind = config["optim"]["lr_scheduler"]
    if kind == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config["optim"]["lr_scheduler_factor"],
            patience=config["optim"]["lr_scheduler_patience"],
        )
    if kind == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config["optim"]["max_epochs"])
    return None


def main() -> None:
    args = parse_args()
    config = build_effective_config(args)

    out_dir = Path(config["results_dir"]) / config["task"] / config["backbone"] / f"fold_{config['fold']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / "log.txt", name=f"{config['task']}_{config['backbone']}_fold{config['fold']}")

    logger.info(f"Effective config: {config}")
    set_global_seed(config["seed"], deterministic=config["deterministic"])
    device = get_device()
    logger.info(f"device={device}")

    df = load_folds(config["data"]["splits_csv"])
    fold = config["fold"]
    if fold not in df["fold"].unique():
        raise ValueError(f"fold {fold} not present in {config['data']['splits_csv']} (available: {sorted(df['fold'].unique())})")

    test_df = df[df["fold"] == fold].reset_index(drop=True)
    trainval_df = df[df["fold"] != fold].reset_index(drop=True)
    train_df, val_df = make_train_val_split(trainval_df, config["data"]["val_fraction"], config["seed"])

    if config.get("smoke_test"):
        train_df = train_df.iloc[: min(12, len(train_df))].reset_index(drop=True)
        val_df = val_df.iloc[: min(8, len(val_df))].reset_index(drop=True)
        test_df = test_df.iloc[: min(8, len(test_df))].reset_index(drop=True)

    logger.info(
        f"train={len(train_df)} (event rate={train_df['event'].mean():.3f}) | "
        f"val={len(val_df)} (event rate={val_df['event'].mean():.3f}) | "
        f"test={len(test_df)} (event rate={test_df['event'].mean():.3f})"
    )

    image_size = config["data"]["image_size"]
    num_workers = config["optim"]["num_workers"]
    batch_size = config["optim"]["batch_size"]

    if config["task"] == "survival":
        # Cox partial-likelihood needs a risk set big enough to be meaningful;
        # with only ~450 training subjects we use full-batch gradient descent
        # for this task instead of the configured minibatch size.
        batch_size = len(train_df)
        logger.info(f"survival task: overriding batch_size to full training set size ({batch_size})")

    train_loader = build_loader(train_df, image_size, config["backbone"], True, batch_size, num_workers)
    val_loader = build_loader(val_df, image_size, config["backbone"], False, max(len(val_df), 1), 0)
    test_loader = build_loader(test_df, image_size, config["backbone"], False, max(len(test_df), 1), 0)

    backbone_input_size = input_size_for_backbone(config["backbone"])
    if image_size != backbone_input_size:
        logger.warning(
            f"image_size={image_size} differs from {config['backbone']}'s native input size "
            f"({backbone_input_size}); some fixed-size ViTs will fail on this."
        )

    model = build_model(
        config["backbone"],
        config["task"],
        config["pretrained_dir"],
        config["model"]["freeze_backbone"],
        logger,
        amp=config["model"].get("amp", False),
    ).to(device)
    logger.info(f"param counts: {count_parameters(model)}")

    loss_fn = get_loss(config["task"])
    if config["task"] == "classification":
        n_pos = train_df["event"].sum()
        n_neg = len(train_df) - n_pos
        pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32, device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info(f"classification pos_weight={pos_weight.item():.3f} (n_pos={n_pos}, n_neg={n_neg})")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if model.freeze_backbone and args.lr is None:
        effective_lr = config["optim"].get("lr_frozen", config["optim"]["lr"])
        logger.info(f"frozen backbone: using lr_frozen={effective_lr:g} instead of lr={config['optim']['lr']:g}")
    else:
        effective_lr = config["optim"]["lr"]
    config["optim"]["effective_lr"] = effective_lr

    optimizer = AdamW(trainable_params, lr=effective_lr, weight_decay=config["optim"]["weight_decay"])
    scheduler = build_scheduler(optimizer, config)
    early_stopper = EarlyStopping(patience=config["optim"]["early_stopping_patience"], mode="min")

    best_path = out_dir / "best_model.pt"
    history = []
    t0 = time.time()

    for epoch in range(config["optim"]["max_epochs"]):
        train_loss, _ = run_epoch(model, train_loader, loss_fn, config["task"], device, optimizer)
        val_loss, val_result = run_epoch(model, val_loader, loss_fn, config["task"], device, optimizer=None)
        val_metrics = compute_task_metrics(config["task"], val_result)

        is_best = early_stopper.step(val_loss)
        if is_best:
            save_checkpoint(model, best_path)

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr_now, "is_best": is_best}
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        logger.info(
            f"epoch {epoch:03d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"lr={lr_now:.2e} best={is_best} metrics={val_metrics}"
        )

        if early_stopper.should_stop:
            logger.info(f"Early stopping at epoch {epoch} (no val_loss improvement for {early_stopper.patience} epochs)")
            break

    if not best_path.exists():
        # Degenerate case (e.g. 0 epochs run): save current weights so downstream code doesn't break.
        save_checkpoint(model, best_path)

    load_checkpoint(model, best_path, device)
    test_loss, test_result = run_epoch(model, test_loader, loss_fn, config["task"], device, optimizer=None)
    test_metrics = compute_task_metrics(config["task"], test_result)
    test_metrics["test_loss"] = test_loss
    logger.info(f"TEST fold={fold} metrics={test_metrics}")
    logger.info(f"Total wall time: {time.time() - t0:.1f}s")

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    save_predictions(config["task"], fold, test_result, out_dir / "test_predictions.csv")
    save_json(test_metrics, out_dir / "metrics.json")
    save_json(model.load_info, out_dir / "checkpoint_load_info.json")
    save_config(config, out_dir / "config.yaml")

    logger.info(f"Saved all artifacts to {out_dir}")


if __name__ == "__main__":
    main()

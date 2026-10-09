#!/usr/bin/env python3
"""Stage 2 of the linear-probe protocol: ridge-penalized linear Cox model on
frozen embeddings (features/<backbone>.npz from scripts/extract_features.py).

For each outer fold k (same splits/folds.csv as every other experiment):
  * test = fold k, never touched until the very end;
  * the other 9 folds ("trainval", ~457 patients, ~80 events) are split into
    5 stratified INNER folds. For every L2 strength lambda on a grid, a ridge
    Cox model is fit on 4 inner folds and scored (C-index) on the 5th;
  * lambda* = argmax of the MEAN inner C-index (ties -> larger lambda);
  * the model is refit on all 9 folds at lambda* and scores the test fold.

Why nested CV + C-index rather than a single val split + partial likelihood
(the first version of this script, fixed on 2026-09-21 before any backbone
comparison was run):
  - the single inner val split of train.py holds only ~12 events: far too
    few to choose lambda reliably;
  - the Cox partial likelihood also penalises the SCALE of the risk scores,
    so on weak signal it always prefers the null model (lambda -> max) even
    when the ranking is informative. The C-index is the evaluation metric
    and is scale-invariant, so it is the right selection criterion.
Both choices are fixed for every backbone and never tuned on test results.

Each fit minimizes
    mean-per-event negative Cox partial log-likelihood + lambda/2 * ||w||^2
with L-BFGS in float64 (strictly convex -> unique optimum), using
src.losses.cox_ph_loss -- the same function and sign convention (higher
risk_score = higher risk) as the rest of the project. Features are
standardized with statistics of the data the model is fit on only.

Outputs mirror train.py so scripts/summarize_results.py works unchanged:
    <results-dir>/survival/<backbone>/fold_<k>/{test_predictions.csv,
        metrics.json, history.csv, config.yaml, head.pt}

Usage:
    python scripts/linear_probe_survival.py --backbone uni
    python scripts/summarize_results.py --results-dir results_probe
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import load_folds  # noqa: E402
from src.losses import cox_ph_loss  # noqa: E402
from src.metrics import survival_metrics  # noqa: E402
from src.utils import save_config, save_json, setup_logger  # noqa: E402

DEFAULT_LAMBDAS = [float(10.0 ** e) for e in np.arange(-4.0, 3.01, 0.5)]  # 1e-4 ... 1e3, 15 values
N_INNER_FOLDS = 5


def cox_loss_np(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    with torch.no_grad():
        return float(cox_ph_loss(torch.from_numpy(risk), torch.from_numpy(time), torch.from_numpy(event)))


def fit_ridge_cox(
    X: torch.Tensor, time: torch.Tensor, event: torch.Tensor, lam: float, w_init: torch.Tensor
) -> tuple[torch.Tensor, dict]:
    """Minimize cox_ph_loss(X @ w) + lam/2 ||w||^2 with L-BFGS. Strictly convex
    for lam > 0, so the optimum is unique and independent of initialization;
    w_init (warm start from the neighbouring lambda) only speeds it up.
    """
    w = w_init.clone().requires_grad_(True)
    opt = torch.optim.LBFGS(
        [w], lr=1.0, max_iter=1000, tolerance_grad=1e-9, tolerance_change=1e-12,
        history_size=20, line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss = cox_ph_loss(X @ w, time, event) + 0.5 * lam * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    w_det = w.detach()
    # Recompute objective and gradient at the returned point to report
    # convergence honestly (L-BFGS can stop on max_iter).
    w_chk = w_det.clone().requires_grad_(True)
    obj = cox_ph_loss(X @ w_chk, time, event) + 0.5 * lam * (w_chk * w_chk).sum()
    obj.backward()
    grad_norm = float(w_chk.grad.norm())
    return w_det, {"objective": float(obj), "grad_norm": grad_norm, "converged": grad_norm < 1e-5}


def standardize(X_fit: np.ndarray, *others: np.ndarray):
    mu = X_fit.mean(axis=0)
    sd = X_fit.std(axis=0)
    sd[sd < 1e-8] = 1.0  # constant dims -> centred to 0, contribute nothing
    return mu, sd, [(x - mu) / sd for x in (X_fit, *others)]


def fit_path(X: np.ndarray, t: np.ndarray, e: np.ndarray, lambdas: list[float]) -> dict[float, tuple]:
    """Fit the whole lambda path (strong -> weak penalty, warm-started)."""
    Xt, tt, et = torch.from_numpy(X), torch.from_numpy(t), torch.from_numpy(e)
    w = torch.zeros(X.shape[1], dtype=torch.float64)
    out = {}
    for lam in sorted(lambdas, reverse=True):
        w, conv = fit_ridge_cox(Xt, tt, et, lam, w)
        out[lam] = (w.numpy().copy(), conv)
    return out


def run_fold(fold: int, df: pd.DataFrame, feats: np.ndarray, lambdas: list[float], seed: int,
             out_dir: Path, backbone: str, logger) -> dict:
    fold = int(fold)
    test_df = df[df["fold"] == fold].reset_index(drop=True)
    trainval_df = df[df["fold"] != fold].reset_index(drop=True)

    X_tv = feats[trainval_df["row"].to_numpy()]
    X_te = feats[test_df["row"].to_numpy()]
    t_tv = trainval_df["follow_up_years"].to_numpy(dtype=np.float64)
    e_tv = trainval_df["event"].to_numpy(dtype=np.float64)
    t_te = test_df["follow_up_years"].to_numpy(dtype=np.float64)
    e_te = test_df["event"].to_numpy(dtype=np.float64)

    logger.info(f"fold {fold}: trainval={len(trainval_df)} ({int(e_tv.sum())} events) | "
                f"test={len(test_df)} ({int(e_te.sum())} events) | D={X_tv.shape[1]}")

    # --- inner CV: choose lambda without ever touching the test fold ---
    skf = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=seed)
    inner_c = {lam: [] for lam in lambdas}
    inner_train_c = {lam: [] for lam in lambdas}
    n_not_conv = 0
    for i, (tr_idx, va_idx) in enumerate(skf.split(X_tv, e_tv.astype(int))):
        _, _, (X_tr, X_va) = standardize(X_tv[tr_idx], X_tv[va_idx])
        path = fit_path(X_tr, t_tv[tr_idx], e_tv[tr_idx], lambdas)
        for lam, (w, conv) in path.items():
            inner_c[lam].append(survival_metrics(t_tv[va_idx], X_va @ w, e_tv[va_idx])["c_index"])
            inner_train_c[lam].append(survival_metrics(t_tv[tr_idx], X_tr @ w, e_tv[tr_idx])["c_index"])
            n_not_conv += int(not conv["converged"])

    rows = []
    for lam in sorted(lambdas):
        cs = np.array(inner_c[lam], dtype=float)
        rows.append({
            "lambda": lam,
            "inner_val_c_index_mean": float(np.nanmean(cs)),
            "inner_val_c_index_std": float(np.nanstd(cs)),
            "inner_train_c_index_mean": float(np.nanmean(inner_train_c[lam])),
            **{f"inner_val_c_fold{j}": float(c) for j, c in enumerate(cs)},
        })
    hist = pd.DataFrame(rows)
    for _, r in hist.iterrows():
        logger.info(f"  lambda={r['lambda']:9.2e} inner_val_C={r['inner_val_c_index_mean']:.3f}"
                    f"±{r['inner_val_c_index_std']:.3f}  inner_train_C={r['inner_train_c_index_mean']:.3f}")

    best = hist["inner_val_c_index_mean"].max()
    lam_star = float(hist.loc[hist["inner_val_c_index_mean"] >= best - 1e-12, "lambda"].max())
    sel = hist[hist["lambda"] == lam_star].iloc[0]

    # --- refit on all 9 training folds at lambda*, score the test fold ---
    mu, sd, (X_tv_s, X_te_s) = standardize(X_tv, X_te)
    refit = fit_path(X_tv_s, t_tv, e_tv, [lam for lam in lambdas if lam >= lam_star])
    w_star, conv_star = refit[lam_star]
    if not conv_star["converged"]:
        logger.warning(f"fold {fold}: refit at lambda={lam_star:.2e} did not fully converge "
                       f"(grad_norm={conv_star['grad_norm']:.2e})")

    risk_te = X_te_s @ w_star
    metrics = survival_metrics(t_te, risk_te, e_te)
    metrics.update({
        "test_loss": cox_loss_np(risk_te, t_te, e_te),
        "selected_lambda": lam_star,
        "inner_val_c_index_at_selected": float(sel["inner_val_c_index_mean"]),
        "inner_val_c_index_std_at_selected": float(sel["inner_val_c_index_std"]),
        "lambda_at_grid_edge": lam_star in (min(lambdas), max(lambdas)),
        "inner_fits_not_converged": n_not_conv,
        "refit_converged": bool(conv_star["converged"]),
    })
    logger.info(f"fold {fold}: lambda*={lam_star:.2e} (inner C={sel['inner_val_c_index_mean']:.3f}) "
                f"-> TEST c_index={metrics['c_index']:.4f}  [inner fits not converged: {n_not_conv}]")

    fold_dir = out_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "case_id": test_df["case_id"],
        "fold": fold,
        "target": t_te,
        "prediction": risk_te,
        "follow_up_years": t_te,
        "event": e_te,
        "risk_score": risk_te,
    }).to_csv(fold_dir / "test_predictions.csv", index=False)
    hist.to_csv(fold_dir / "history.csv", index=False)
    save_json(metrics, fold_dir / "metrics.json")
    torch.save({"w": torch.from_numpy(w_star), "feature_mean": torch.from_numpy(mu),
                "feature_std": torch.from_numpy(sd), "lambda": lam_star}, fold_dir / "head.pt")
    save_config({
        "task": "survival",
        "protocol": "linear_probe: frozen features + ridge Cox (L-BFGS), nested CV",
        "backbone": backbone,
        "fold": fold,
        "seed": int(seed),
        "inner_folds": N_INNER_FOLDS,
        "lambda_grid": [float(l) for l in lambdas],
        "selection": "argmax mean inner-fold C-index, ties -> largest lambda",
        "standardization": "mean/std of the data each model is fit on",
        "final_fit_on": "all 9 non-test folds at lambda*",
    }, fold_dir / "config.yaml")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", required=True)
    p.add_argument("--features-dir", default="features")
    p.add_argument("--splits-csv", default="splits/folds.csv")
    p.add_argument("--results-dir", default="results_probe")
    p.add_argument("--folds", default=None, help="comma-separated, default all")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.set_num_threads(4)
    npz = np.load(Path(args.features_dir) / f"{args.backbone}.npz", allow_pickle=False)
    feat_ids = npz["case_id"].astype(str)
    feats = npz["features"].astype(np.float64)

    df = load_folds(args.splits_csv)
    index = {cid: i for i, cid in enumerate(feat_ids)}
    missing = sorted(set(df["case_id"]) - set(index))
    if missing:
        raise ValueError(f"{len(missing)} cases in folds.csv have no features, e.g. {missing[:5]}")
    df["row"] = df["case_id"].map(index)

    out_dir = Path(args.results_dir) / "survival" / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / "probe_log.txt", name=f"probe_{args.backbone}")
    logger.info(f"backbone={args.backbone} features={feats.shape} lambdas={len(DEFAULT_LAMBDAS)} "
                f"inner_folds={N_INNER_FOLDS}")

    folds = [int(f) for f in args.folds.split(",")] if args.folds else [int(f) for f in sorted(df["fold"].unique())]
    results = {}
    for fold in folds:
        results[fold] = run_fold(fold, df, feats, DEFAULT_LAMBDAS, args.seed, out_dir, args.backbone, logger)

    cis = [m["c_index"] for m in results.values() if np.isfinite(m["c_index"])]
    lams = ", ".join(f"{m['selected_lambda']:.0e}" for m in results.values())
    logger.info(f"SUMMARY {args.backbone}: C-index mean={np.mean(cis):.4f} std={np.std(cis):.4f} "
                f"over {len(cis)} folds; selected lambdas=[{lams}]")
    edge = [f for f, m in results.items() if m["lambda_at_grid_edge"]]
    if edge:
        logger.info(f"note: lambda* sits at the edge of the grid in folds {edge}")


if __name__ == "__main__":
    main()

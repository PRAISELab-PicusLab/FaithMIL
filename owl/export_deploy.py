"""Collect a trained FaithMIL run into owl/model/ (the model tarball, extracted to /opt/ml/model on Grand Challenge).

    python owl/export_deploy.py --data-root /path/to/wacv_challenge

Reads (all from the data root)
  exp_ctx_l1l2/results/<run>/fold_<k>/model_seed<s>.pt      the 10 x 3 ABMIL members (mu, sd of their training risks)
  exp_ctx_l1l2/results/<run>/fold_<k>/test_predictions.csv  out-of-fold ensemble z of every patient
  exp_dino_prost40m_tree/results/dino_l1/<ckpt>             the DINO L1 encoder the features were extracted with
Writes
  model/dino_L1.pth                                          encoder state dict
  model/ensemble/fold_<k>/model_seed<s>.pt                   state dict + mu, sd + sizes (loadable with weights_only)
  model/calibration.json                                     z -> years (Weibull AFT median) and the risk-class cuts
  model/cohort_ref.csv                                       OOF z, years, class, follow-up, event of the RUMC patients
  model/deploy.json                                          provenance and OOF C-index

Years and classes are calibrated on the out-of-fold predictions of the OWL (RUMC) patients only, the population of the
challenge test set. The ensemble risk z is higher for earlier recurrence; years decrease with z, so the C-index of the
years equals the C-index of -z.
"""
import argparse
import glob
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lifelines import WeibullAFTFitter
from lifelines.utils import concordance_index

OWL = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--run", default="l1only_abmil_dino")
    ap.add_argument("--encoder", default="exp_dino_prost40m_tree/results/dino_l1/teacher_backbone_ep005.pth")
    ap.add_argument("--out", type=Path, default=OWL / "model")
    a = ap.parse_args()
    run_dir = a.data_root / "exp_ctx_l1l2" / "results" / a.run
    out = a.out
    for sub in ("ensemble",):
        shutil.rmtree(out / sub, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    # encoder
    sd = torch.load(a.data_root / a.encoder, map_location="cpu", weights_only=True)
    torch.save(sd, out / "dino_L1.pth")

    # ensemble members
    n = 0
    for fp in sorted(run_dir.glob("fold_*/model_seed*.pt")):
        ck = torch.load(fp, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        assert cfg["arch"] == "abmil" and not cfg["slide_norm"], cfg
        dst = out / "ensemble" / fp.parent.name / fp.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": ck["state_dict"], "mu": float(ck["mu"]), "sd": float(ck["sd"]), "D": int(ck["D1"]),
                    "hidden_dim": int(cfg["hidden_dim"]), "attention_dim": int(cfg["attention_dim"])}, dst)
        n += 1
    print(f"{n} ensemble members")

    # calibration on the RUMC out-of-fold predictions
    oof = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(run_dir / "fold_*" / "test_predictions.csv"))))
    oof = oof[oof.case_id.str.startswith("case_radboud")].rename(columns={"risk_score": "z"}).reset_index(drop=True)
    aft = WeibullAFTFitter().fit(oof[["follow_up_years", "event", "z"]], "follow_up_years", "event")
    lam_b, lam_0 = float(aft.params_["lambda_"]["z"]), float(aft.params_["lambda_"]["Intercept"])
    rho = float(np.exp(aft.params_["rho_"]["Intercept"]))
    # Weibull AFT median: exp(lam_0 + lam_b z) * ln(2)^(1/rho)
    years = np.exp(lam_0 + lam_b * oof.z) * np.log(2) ** (1 / rho)
    assert lam_b < 0, "years must decrease with the risk"
    cut_hi, cut_lo = np.quantile(years, [2 / 3, 1 / 3])  # years >= cut_hi: low risk; < cut_lo: high risk
    cls = np.where(years >= cut_hi, "low", np.where(years >= cut_lo, "medium", "high"))
    cal = {"years": {"model": "weibull_aft_median", "log_scale_intercept": lam_0, "log_scale_slope": lam_b, "rho": rho,
                     "formula": "exp(log_scale_intercept + log_scale_slope * z) * ln(2) ** (1 / rho)"},
           "risk_class": {"low_if_years_at_least": float(cut_hi), "high_if_years_below": float(cut_lo),
                          "source": "tertiles of the predicted years of the RUMC out-of-fold patients"},
           "max_follow_up_years": float(oof.follow_up_years.max())}
    (out / "calibration.json").write_text(json.dumps(cal, indent=2))
    ref = oof.assign(years=years, risk_class=cls)[["case_id", "z", "years", "risk_class", "follow_up_years", "event"]]
    ref.to_csv(out / "cohort_ref.csv", index=False)

    c_oof = concordance_index(oof.follow_up_years, years, oof.event)
    for g in ("low", "medium", "high"):
        r = ref[ref.risk_class == g]
        print(f"  {g:6s} n={len(r)} events={int(r.event.sum())} ({r.event.mean():.0%})")
    deploy = {"run": a.run, "n_members": n,
              "encoder": "dino_L1.pth: ViT-S/14 trained by us with DINO from random initialisation on unlabeled L1 tiles "
                         "of the OWL training slides; no pre-trained weights",
              "instances": "L1 tiles (224 px at ~0.97 um/px, 0.22 mm)",
              "training_data": "OWL Radboudumc training set (508) + TCGA-PRAD (298), 10-fold CV x 3 seeds",
              "oof_c_index_rumc_years": float(c_oof), "n_rumc": len(oof), "events_rumc": int(oof.event.sum())}
    (out / "deploy.json").write_text(json.dumps(deploy, indent=2))
    print(json.dumps(deploy, indent=2))
    print(json.dumps(cal, indent=2))


if __name__ == "__main__":
    main()

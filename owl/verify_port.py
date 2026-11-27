"""Check that the container reproduces the trained model (run on a GPU node, from the data root).

  1. port:     the exported ABMIL members on the STORED training features give the stored per-seed test risks
  2. features: the container's tile reading + encoder give the stored features (cosine) and the same ensemble z

    python FaithMIL/owl/verify_port.py --data-root . --cases case_radboud_0000 case_radboud_0466
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OWL = Path(__file__).resolve().parent
sys.path.insert(0, str(OWL.parent))
sys.path.insert(0, str(OWL))
import owl_model as om  # noqa: E402
from trace_mil.anchors import tree_anchors  # noqa: E402
from trace_mil.wsi import TissueMask, read_wsi_header  # noqa: E402

RUN = "exp_ctx_l1l2/results/l1only_abmil_dino"
FEAT = "exp_dino_prost40m_tree/roots/dino_tree/features/tree_L1children/dino_tree/L1"
ANC = "exp_dino_prost40m_tree/roots/dino_tree/anchors/tree_L1children"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=OWL / "model")
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--port-only", action="store_true")
    a = ap.parse_args()
    root = a.data_root
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ens = om.Ensemble(a.model, dev)
    fold_of = {}
    for f in sorted(p for p in (root / RUN).glob("fold_*") if p.is_dir()):
        for c in pd.read_csv(f / "test_predictions.csv").case_id:
            fold_of[c] = f

    worst = 0.0
    print("1. port: stored features -> exported members of the case's test fold vs stored per-seed risks")
    for case in a.cases:
        fdir = fold_of[case]
        X = torch.from_numpy(np.load(root / FEAT / f"{case}.npy"))
        members = sorted((a.model / "ensemble" / fdir.name).glob("model_seed*.pt"))
        for s, fp in enumerate(members):
            sub = om.Ensemble.__new__(om.Ensemble)
            sub.device, sub.members = dev, []
            ck = torch.load(fp, map_location="cpu", weights_only=True)
            m = om.ABMIL(ck["D"], ck["hidden_dim"], ck["attention_dim"])
            m.load_state_dict(ck["state_dict"])
            sub.members = [m.to(dev).eval()]
            g, sc = sub.tile_terms(X)
            a_ = np.exp(g - g.max()); a_ /= a_.sum()
            risk = float((a_ * sc).sum())
            ref = pd.read_csv(fdir / f"test_predictions_seed{s}.csv").set_index("case_id").loc[case, "risk_score"]
            worst = max(worst, abs(risk - ref))
            print(f"  {case} {fdir.name} seed{s}: {risk:+.6f} vs stored {ref:+.6f}")
    print(f"  max |diff| {worst:.2e}  ->  {'OK' if worst < 1e-4 else 'MISMATCH'}")
    if a.port_only:
        return

    print("2. features: container reading + encoder vs stored features")
    vit = om.load_encoder(a.model / "dino_L1.pth", dev)
    for case in a.cases:
        wsi, mask_p = root / "images" / f"{case}.tif", root / "tissue-masks" / f"{case}_tissue.tif"
        h = read_wsi_header(wsi)
        parents, children = tree_anchors(TissueMask(mask_p, 1, h["res_l0"]), h["width_l0"], h["height_l0"])
        stored_anc = pd.read_csv(root / ANC / f"{case}.csv")
        same = (children[["x", "y"]].to_numpy() == stored_anc[["x", "y"]].to_numpy()).all()
        Xn = om.extract_features(vit, wsi, parents, children, dev).float()
        Xs = torch.from_numpy(np.load(root / FEAT / f"{case}.npy")).float()
        cos = torch.nn.functional.cosine_similarity(Xn, Xs, dim=1)
        zn = om.ensemble_z(*ens.tile_terms(Xn), ens.mu, ens.sd)
        zs = om.ensemble_z(*ens.tile_terms(Xs), ens.mu, ens.sd)
        print(f"  {case}: {len(children)} tiles, same anchors {same}, cosine min {cos.min():.4f} mean {cos.mean():.5f}, "
              f"ensemble z {zn:+.4f} (container) vs {zs:+.4f} (stored features), |dz| {abs(zn - zs):.4f}")


if __name__ == "__main__":
    main()

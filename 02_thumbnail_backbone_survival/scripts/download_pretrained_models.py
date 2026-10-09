#!/usr/bin/env python3
"""Download every pretrained checkpoint the backbones need, into pretrained/.

RUN THIS ONLY FROM THE LOGIN NODE. Everything else in this project (train.py,
run_cv.py, src/models.py) loads exclusively from local files and must work
with zero network access, as required on Leonardo's compute nodes.

Each HuggingFace-hosted backbone is actually *built* here, not just fetched:
that guarantees the cache contains exactly the files the training-time
loading recipe asks for, and verifies the recipe works before you queue any
jobs.

Usage (login node):
    source .venv-train/bin/activate

    python scripts/download_pretrained_models.py                    # everything available
    python scripts/download_pretrained_models.py --models open      # only non-gated
    python scripts/download_pretrained_models.py --models gated     # only gated (needs token)
    python scripts/download_pretrained_models.py --models uni,virchow2

GATED MODELS (UNI, Virchow2, H-optimus-0, PathOrchestra) additionally need a
HuggingFace account that has accepted each model's licence, plus a token:

    huggingface-cli login          # paste a read token from hf.co/settings/tokens
    # or: export HF_TOKEN=hf_xxx

PathOrchestra is manual-approval gated, so access can take days and this
script will report it as unavailable until the owners approve the request.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backbones import BACKBONES, build_backbone, set_hf_env  # noqa: E402

TORCHVISION_IMAGENET_FILE = "imagenet_resnet50.pth"

LUNIT_URL_PREFIX = (
    "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
)
LUNIT_FILES = {
    "barlow_twins": "bt_rn50_ep200.torch",
    "mocov2": "mocov2_rn50_ep200.torch",
    "swav": "swav_rn50_ep200.torch",
}

# HistoEncoder ships its prostate models as plain torch checkpoints; URLs
# taken from histoencoder/functional/_model.py in jopo666/HistoEncoder.
HISTOENCODER_URLS = {
    "histoencoder_prostate_small": "https://dl.dropboxusercontent.com/s/tbff9wslc8p7ie3/prostate_small.pth?dl=0",
    "histoencoder_prostate_medium": "https://dl.dropboxusercontent.com/s/k1fr09x5auki8sp/prostate_medium.pth?dl=0",
}


def _download(url: str, dest: Path) -> None:
    print(f"Downloading {url}")
    print(f"        -> {dest}")
    last_shown = [-1]

    def _progress(block_num, block_size, total_size):
        if total_size <= 0:
            return
        pct = min(100.0, 100.0 * block_num * block_size / total_size)
        if int(pct) != last_shown[0]:
            last_shown[0] = int(pct)
            sys.stdout.write(f"\r  {pct:5.1f}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    sys.stdout.write("\n")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_imagenet_resnet50(pretrained_dir: Path, force: bool) -> None:
    dest = pretrained_dir / TORCHVISION_IMAGENET_FILE
    if dest.exists() and not force:
        print(f"[skip] {dest.name} already present ({dest.stat().st_size:,} bytes)")
        return

    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    print("Downloading torchvision ResNet50 ImageNet weights (IMAGENET1K_V2)...")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    torch.save(model.state_dict(), dest)
    print(f"  saved -> {dest}  size={dest.stat().st_size:,} bytes  sha256={_sha256(dest)}")


def download_url_checkpoint(name: str, url: str, dest: Path, force: bool) -> None:
    if dest.exists() and not force:
        print(f"[skip] {dest.name} already present ({dest.stat().st_size:,} bytes)")
        return
    _download(url, dest)
    print(f"  sha256={_sha256(dest)}  size={dest.stat().st_size:,} bytes")


def ensure_safetensors_copy(repo_id: str) -> None:
    """transformers >= 5 refuses `pytorch_model.bin` with torch < 2.6
    (CVE-2025-32434). Some repos (vinid/plip) only ship the .bin, so convert
    it once, here on the login node, into `model.safetensors` inside the
    same HF cache snapshot. from_pretrained then prefers the safetensors
    file and never touches torch.load again -- and torch stays pinned at
    the version all earlier results were produced with.
    """
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file

    snapshot = Path(snapshot_download(repo_id))
    st_path = snapshot / "model.safetensors"
    bin_path = snapshot / "pytorch_model.bin"
    if st_path.exists() or not bin_path.exists():
        return

    print(f"  converting {bin_path.name} -> model.safetensors (one-off, trusted public repo)")
    state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    # clone() breaks any shared storage, which safetensors refuses to serialize
    state_dict = {k: v.contiguous().clone() for k, v in state_dict.items()}
    save_file(state_dict, str(st_path), metadata={"format": "pt"})
    print(f"  wrote {st_path} ({st_path.stat().st_size:,} bytes, {len(state_dict)} tensors)")


def download_timm_hub_files(repo_id: str) -> tuple[bool, str]:
    """Fetch only the files timm's hf-hub loader will actually read, without
    building the model.

    Leonardo login nodes kill any process after 600 s of CPU time
    (`ulimit -t 600`); building + forward-passing a ViT-H/ViT-g on CPU blows
    that budget, so for these models the login node only downloads and the
    load + forward-pass check must run on a GPU node.
    Prefers model.safetensors and skips a duplicate pytorch_model.bin
    (Virchow2 ships both at 2.5 GB each).
    """
    from huggingface_hub import list_repo_files, snapshot_download

    files = set(list_repo_files(repo_id))
    weights = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
    if weights not in files:
        return False, f"FAILED: no model.safetensors / pytorch_model.bin in {repo_id}"

    path = snapshot_download(repo_id, allow_patterns=["config.json", weights])
    size = (Path(path) / weights).stat().st_size
    return True, f"OK downloaded {weights} ({size / 1e9:.2f} GB) -- verify it with a load + forward pass on a GPU node"


def fetch_hub_backbone(name: str, pretrained_dir: Path) -> tuple[bool, str]:
    """Build the backbone once with downloads enabled, populating the local
    HF cache and proving the loading recipe works. Returns (ok, message).
    """
    logger = logging.getLogger("download")
    try:
        _, feature_dim, _ = build_backbone(name, pretrained_dir, logger, allow_download=True)
        return True, f"OK (feature_dim={feature_dim})"
    except Exception as exc:  # noqa: BLE001 - we want to report, not crash the batch
        msg = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)
        return False, f"FAILED: {type(exc).__name__}: {msg[:300]}"


def resolve_selection(selection: str) -> list[str]:
    if selection == "all":
        return list(BACKBONES)
    if selection == "open":
        return [n for n, s in BACKBONES.items() if not s.gated]
    if selection == "gated":
        return [n for n, s in BACKBONES.items() if s.gated]
    if selection == "resnet50":
        return [n for n, s in BACKBONES.items() if s.family == "resnet50_local"]
    names = [n.strip() for n in selection.split(",") if n.strip()]
    unknown = [n for n in names if n not in BACKBONES]
    if unknown:
        raise SystemExit(f"Unknown backbone(s): {unknown}. Available: {list(BACKBONES)}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pretrained-dir", default="pretrained")
    parser.add_argument(
        "--models",
        default="all",
        help="'all' | 'open' | 'gated' | 'resnet50' | comma-separated backbone names",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if already present")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    pretrained_dir = Path(args.pretrained_dir)
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = set_hf_env(pretrained_dir, offline=False)
    cache_dir.mkdir(parents=True, exist_ok=True)

    selected = resolve_selection(args.models)
    results: dict[str, str] = {}

    for name in selected:
        spec = BACKBONES[name]
        print(f"\n=== {name} === ({spec.description})")

        try:
            if name == "imagenet":
                download_imagenet_resnet50(pretrained_dir, args.force)
                results[name] = "OK"
            elif name in LUNIT_FILES:
                download_url_checkpoint(
                    name, f"{LUNIT_URL_PREFIX}/{LUNIT_FILES[name]}",
                    pretrained_dir / LUNIT_FILES[name], args.force,
                )
                results[name] = "OK"
            elif name in HISTOENCODER_URLS:
                download_url_checkpoint(
                    name, HISTOENCODER_URLS[name],
                    pretrained_dir / spec.local_file, args.force,
                )
                ok, msg = fetch_hub_backbone(name, pretrained_dir)
                results[name] = msg
            else:
                if spec.gated:
                    print("  (gated repo: needs an HF token with the licence accepted)")
                if spec.family == "hf_clip":
                    ensure_safetensors_copy(spec.hf_repo)
                if spec.family == "timm_hub":
                    ok, msg = download_timm_hub_files(spec.hf_repo)
                else:
                    ok, msg = fetch_hub_backbone(name, pretrained_dir)
                results[name] = msg
                print(f"  {msg}")
        except Exception as exc:  # noqa: BLE001
            results[name] = f"FAILED: {type(exc).__name__}: {str(exc)[:200]}"
            print(f"  {results[name]}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    width = max(len(n) for n in results) if results else 10
    failed = []
    for name, status in results.items():
        gated_tag = " [gated]" if BACKBONES[name].gated else ""
        print(f"  {name:<{width}}  {status}{gated_tag}")
        if not status.startswith("OK"):
            failed.append(name)

    print(f"\nLocal files under: {pretrained_dir.resolve()}")
    print(f"HF cache under:    {cache_dir.resolve()}")
    if failed:
        gated_failed = [n for n in failed if BACKBONES[n].gated]
        if gated_failed:
            print(
                f"\nGated models still unavailable: {gated_failed}\n"
                "  1. Accept each licence on its huggingface.co model page\n"
                "  2. huggingface-cli login   (read token from hf.co/settings/tokens)\n"
                "  3. re-run:  python scripts/download_pretrained_models.py --models gated"
            )
        sys.exit(1)
    print("\nAll selected backbones are local. Training can now run fully offline.")


if __name__ == "__main__":
    main()

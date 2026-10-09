"""Registry of feature-extractor backbones.

Every backbone exposes the same contract: a torch module mapping a batch of
normalized images (B, 3, H, W) to a flat embedding (B, feature_dim), plus the
normalization statistics and input size its authors specified. train.py puts
a single Linear head on top, so swapping a backbone is the only thing that
changes across experiments.

All loading is OFFLINE: timm / open_clip / transformers all resolve their
weights from the local HuggingFace cache that
scripts/download_pretrained_models.py pre-populates under
`pretrained/hf_cache` (see set_offline_hf_env below). Nothing here touches
the network at train time, which is required on Leonardo's compute nodes.

feature_dim is DISCOVERED with a dummy forward pass at build time rather than
trusted from the table below -- the declared value is only a cross-check, so
a silently-wrong dimension can't propagate into the head.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Lunit pathology-SSL normalization (Barlow Twins / MoCoV2 / SwAV ResNet50).
LUNIT_MEAN = (0.70322989, 0.53606487, 0.66096631)
LUNIT_STD = (0.21716536, 0.26081574, 0.20723464)

# H-optimus-0's own published statistics.
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)

# OpenAI CLIP statistics, used by the CLIP-family pathology models.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    family: str  # resnet50_local | timm_hub | timm_xcit_local | open_clip_hub | hf_clip
    expected_dim: int | None
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    input_size: int = 224
    hf_repo: str | None = None
    local_file: str | None = None
    gated: bool = False
    default_freeze: bool = False
    timm_kwargs: dict = field(default_factory=dict)
    description: str = ""


BACKBONES: dict[str, BackboneSpec] = {
    # --- ResNet50 baseline family (unchanged; earlier results stay comparable) ---
    "imagenet": BackboneSpec(
        "imagenet", "resnet50_local", 2048, IMAGENET_MEAN, IMAGENET_STD,
        local_file="imagenet_resnet50.pth",
        description="torchvision ResNet50, ImageNet-1k (IMAGENET1K_V2)",
    ),
    "barlow_twins": BackboneSpec(
        "barlow_twins", "resnet50_local", 2048, LUNIT_MEAN, LUNIT_STD,
        local_file="bt_rn50_ep200.torch",
        description="Lunit ResNet50, Barlow Twins on TCGA",
    ),
    "mocov2": BackboneSpec(
        "mocov2", "resnet50_local", 2048, LUNIT_MEAN, LUNIT_STD,
        local_file="mocov2_rn50_ep200.torch",
        description="Lunit ResNet50, MoCoV2 on TCGA",
    ),
    "swav": BackboneSpec(
        "swav", "resnet50_local", 2048, LUNIT_MEAN, LUNIT_STD,
        local_file="swav_rn50_ep200.torch",
        description="Lunit ResNet50, SwAV on TCGA",
    ),

    # --- Pathology foundation models (frozen by default: 0.3B-1.1B params
    #     cannot be meaningfully fine-tuned on ~450 training thumbnails) ---
    "uni": BackboneSpec(
        "uni", "timm_hub", 1024, IMAGENET_MEAN, IMAGENET_STD,
        hf_repo="MahmoodLab/UNI", gated=True, default_freeze=True,
        timm_kwargs={"init_values": 1e-5, "dynamic_img_size": True},
        description="UNI, ViT-L/16 DINOv2 on 100k slides (Mahmood Lab)",
    ),
    "virchow2": BackboneSpec(
        "virchow2", "timm_hub", 2560, IMAGENET_MEAN, IMAGENET_STD,
        hf_repo="paige-ai/Virchow2", gated=True, default_freeze=True,
        description="Virchow2, ViT-H/14 DINOv2 on 3.1M slides (Paige); embedding = cls || mean(patch)",
    ),
    "h_optimus_0": BackboneSpec(
        "h_optimus_0", "timm_hub", 1536, HOPTIMUS_MEAN, HOPTIMUS_STD,
        hf_repo="bioptimus/H-optimus-0", gated=True, default_freeze=True,
        timm_kwargs={"init_values": 1e-5, "dynamic_img_size": False},
        description="H-optimus-0, ViT-g/14 on 500k slides (Bioptimus)",
    ),
    "prost40m": BackboneSpec(
        "prost40m", "timm_hub", 384, IMAGENET_MEAN, IMAGENET_STD,
        hf_repo="waticlems/Prost40M", default_freeze=True,
        description="Prost40M, ViT-S/14 trained on 40M prostate tiles (BCR-risk paper)",
    ),
    "pathorchestra": BackboneSpec(
        "pathorchestra", "timm_hub", None, IMAGENET_MEAN, IMAGENET_STD,
        hf_repo="AI4Pathology/PathOrchestra", gated=True, default_freeze=True,
        description="PathOrchestra (manual-approval gated; dim auto-detected on first load)",
    ),

    # --- Prostate-specific (HistoEncoder, DINO-pretrained XCiT on prostate) ---
    "histoencoder_prostate_small": BackboneSpec(
        "histoencoder_prostate_small", "timm_xcit_local", 384, IMAGENET_MEAN, IMAGENET_STD,
        local_file="histoencoder_prostate_small.pth", default_freeze=True,
        timm_kwargs={"arch": "xcit_small_12_p16_224"},
        description="HistoEncoder prostate_small, XCiT-S12/p16 DINO on prostate tissue",
    ),
    "histoencoder_prostate_medium": BackboneSpec(
        "histoencoder_prostate_medium", "timm_xcit_local", 512, IMAGENET_MEAN, IMAGENET_STD,
        local_file="histoencoder_prostate_medium.pth", default_freeze=True,
        timm_kwargs={"arch": "xcit_medium_24_p16_224"},
        description="HistoEncoder prostate_medium, XCiT-M24/p16 DINO on prostate tissue",
    ),

    # --- Vision-language pathology models (image tower only) ---
    "quiltnet_b16": BackboneSpec(
        "quiltnet_b16", "open_clip_hub", 512, CLIP_MEAN, CLIP_STD,
        hf_repo="wisdomik/QuiltNet-B-16", default_freeze=True,
        description="QuiltNet-B-16, CLIP ViT-B/16 on Quilt-1M histopathology pairs",
    ),
    "plip": BackboneSpec(
        "plip", "hf_clip", 512, CLIP_MEAN, CLIP_STD,
        hf_repo="vinid/plip", default_freeze=True,
        description="PLIP, CLIP ViT-B/32 on OpenPath Twitter pathology pairs",
    ),
    "biomedclip": BackboneSpec(
        "biomedclip", "open_clip_hub", 512, CLIP_MEAN, CLIP_STD,
        hf_repo="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        default_freeze=True,
        description="BiomedCLIP, ViT-B/16 on PMC-15M biomedical figure-caption pairs",
    ),
}

BACKBONE_NAMES = list(BACKBONES.keys())
RESNET50_BACKBONES = [n for n, s in BACKBONES.items() if s.family == "resnet50_local"]
FOUNDATION_BACKBONES = [n for n, s in BACKBONES.items() if s.family != "resnet50_local"]


def get_spec(name: str) -> BackboneSpec:
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Available: {BACKBONE_NAMES}")
    return BACKBONES[name]


def set_hf_env(pretrained_dir: Path | str, offline: bool = True) -> Path:
    """Point HuggingFace at the project-local cache, and (by default) forbid
    network calls.

    Must run BEFORE timm/open_clip/transformers resolve any hub model. The
    download script calls this with offline=False from the login node to
    populate exactly the same cache the compute nodes will read.
    """
    cache_dir = Path(pretrained_dir) / "hf_cache"
    # huggingface_hub looks for the login token at $HF_HOME/token, so moving
    # HF_HOME would hide the token `huggingface-cli login` wrote to the
    # default location -> 401 on gated repos. Pin the token path first.
    if "HF_TOKEN" not in os.environ and "HF_TOKEN_PATH" not in os.environ:
        default_home = os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "huggingface"
        )
        os.environ["HF_TOKEN_PATH"] = os.path.join(default_home, "token")
    os.environ["HF_HOME"] = str(cache_dir)
    flag = "1" if offline else "0"
    os.environ["HF_HUB_OFFLINE"] = flag
    os.environ["TRANSFORMERS_OFFLINE"] = flag
    return cache_dir


# --------------------------------------------------------------------------
# Backbone wrappers -- each maps (B, 3, H, W) -> (B, feature_dim)
# --------------------------------------------------------------------------


class TimmPooledBackbone(nn.Module):
    """Any timm model created with num_classes=0, i.e. already pooled."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Virchow2Backbone(nn.Module):
    """Virchow2's published embedding: class token concatenated with the mean
    of the patch tokens (tokens 0 and 5: -- indices 1..4 are register tokens).
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        class_token = out[:, 0]
        patch_tokens = out[:, 5:]
        return torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)


class OpenClipImageBackbone(nn.Module):
    """open_clip image tower; returns the projected image embedding."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(x)


class HfClipImageBackbone(nn.Module):
    """transformers CLIPModel image tower; returns the projected embedding."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model.get_image_features(pixel_values=x)
        if isinstance(out, torch.Tensor):  # transformers < 5 returned the tensor directly
            return out
        # transformers >= 5 returns the vision ModelOutput with .pooler_output
        # overwritten by the projected image embedding.
        return out.pooler_output


def _build_resnet50(spec: BackboneSpec, pretrained_dir: Path, logger: logging.Logger):
    """Kept byte-for-byte compatible with the original ResNet50 loading path
    so previously-produced results remain reproducible.
    """
    from torchvision.models import resnet50

    model = resnet50(weights=None)
    ckpt_path = pretrained_dir / spec.local_file
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {ckpt_path}. "
            "Run scripts/download_pretrained_models.py from the LOGIN NODE first."
        )

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict and not any(
        k.startswith("conv1") or k.startswith("layer1") for k in state_dict
    ):
        state_dict = state_dict["state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    result = model.load_state_dict(state_dict, strict=False)
    missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
    logger.info(f"[{spec.name}] loaded checkpoint: {ckpt_path.name}")
    logger.info(f"[{spec.name}] missing_keys={missing}")
    logger.info(f"[{spec.name}] unexpected_keys={unexpected}")

    expected_missing = set() if spec.name == "imagenet" else {"fc.weight", "fc.bias"}
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(
            f"[{spec.name}] checkpoint mismatch: expected missing={sorted(expected_missing)}, "
            f"unexpected=[], but got missing={sorted(missing)}, unexpected={sorted(unexpected)}. "
            "Refusing to silently continue with a possibly-corrupt trunk."
        )

    model.fc = nn.Identity()  # forward() now returns the pooled 2048-d feature
    info = {"checkpoint": str(ckpt_path), "missing_keys": missing, "unexpected_keys": unexpected}
    return model, info


def _build_timm_hub(spec: BackboneSpec, logger: logging.Logger):
    import timm

    kwargs = dict(spec.timm_kwargs)
    if spec.name == "virchow2":
        from timm.layers import SwiGLUPacked

        kwargs.update({"mlp_layer": SwiGLUPacked, "act_layer": torch.nn.SiLU})

    model = timm.create_model(f"hf-hub:{spec.hf_repo}", pretrained=True, **kwargs)
    logger.info(f"[{spec.name}] loaded timm hub model {spec.hf_repo} (kwargs={list(kwargs)})")

    wrapper = Virchow2Backbone(model) if spec.name == "virchow2" else TimmPooledBackbone(model)
    return wrapper, {"checkpoint": f"hf-hub:{spec.hf_repo}"}


def _build_timm_xcit_local(spec: BackboneSpec, pretrained_dir: Path, logger: logging.Logger):
    """HistoEncoder prostate models: stock timm XCiT + the author's weights."""
    import timm

    ckpt_path = pretrained_dir / spec.local_file
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"HistoEncoder checkpoint not found: {ckpt_path}. "
            "Run scripts/download_pretrained_models.py from the LOGIN NODE first."
        )

    model = timm.create_model(spec.timm_kwargs["arch"], pretrained=False, num_classes=0)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    result = model.load_state_dict(state_dict, strict=False)
    missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
    logger.info(f"[{spec.name}] loaded checkpoint: {ckpt_path.name}")
    logger.info(f"[{spec.name}] missing_keys={missing}")
    logger.info(f"[{spec.name}] unexpected_keys={unexpected}")

    # num_classes=0 removes the head, so head.* keys in the checkpoint are the
    # only acceptable leftovers; anything else means the arch doesn't match.
    bad_unexpected = [k for k in unexpected if not k.startswith("head.")]
    if missing or bad_unexpected:
        raise RuntimeError(
            f"[{spec.name}] checkpoint mismatch: missing={missing}, unexpected={bad_unexpected}. "
            "Refusing to silently continue with a possibly-corrupt trunk."
        )

    return TimmPooledBackbone(model), {
        "checkpoint": str(ckpt_path),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def _build_open_clip(spec: BackboneSpec, logger: logging.Logger):
    import open_clip

    model, _ = open_clip.create_model_from_pretrained(f"hf-hub:{spec.hf_repo}")
    logger.info(f"[{spec.name}] loaded open_clip model {spec.hf_repo}")
    return OpenClipImageBackbone(model), {"checkpoint": f"hf-hub:{spec.hf_repo}"}


def _build_hf_clip(spec: BackboneSpec, logger: logging.Logger):
    from huggingface_hub import snapshot_download
    from transformers import CLIPModel

    # Load from the cached snapshot DIRECTORY, not the repo id: transformers
    # otherwise consults the Hub's file listing and picks pytorch_model.bin,
    # which it then refuses to torch.load with torch < 2.6. The download
    # script writes a model.safetensors next to it precisely so that a
    # directory load finds (and prefers) the safe file.
    snapshot = snapshot_download(spec.hf_repo)
    model = CLIPModel.from_pretrained(snapshot, use_safetensors=True)
    logger.info(f"[{spec.name}] loaded transformers CLIPModel from {snapshot}")
    return HfClipImageBackbone(model), {"checkpoint": f"hf-hub:{spec.hf_repo}", "snapshot": snapshot}


def build_backbone(
    name: str,
    pretrained_dir: Path | str,
    logger: logging.Logger | None = None,
    allow_download: bool = False,
) -> tuple[nn.Module, int, dict]:
    """Returns (backbone module, feature_dim, info dict).

    feature_dim comes from an actual dummy forward pass, not from the table.
    `allow_download` is only ever True in the login-node download script.
    """
    log = logger or logging.getLogger(__name__)
    spec = get_spec(name)
    pretrained_dir = Path(pretrained_dir)

    if spec.family != "resnet50_local":
        set_hf_env(pretrained_dir, offline=not allow_download)

    if spec.family == "resnet50_local":
        backbone, info = _build_resnet50(spec, pretrained_dir, log)
    elif spec.family == "timm_hub":
        backbone, info = _build_timm_hub(spec, log)
    elif spec.family == "timm_xcit_local":
        backbone, info = _build_timm_xcit_local(spec, pretrained_dir, log)
    elif spec.family == "open_clip_hub":
        backbone, info = _build_open_clip(spec, log)
    elif spec.family == "hf_clip":
        backbone, info = _build_hf_clip(spec, log)
    else:
        raise ValueError(f"Unknown backbone family '{spec.family}' for '{name}'")

    backbone.eval()
    with torch.no_grad():
        dummy = torch.zeros(2, 3, spec.input_size, spec.input_size)
        out = backbone(dummy)
    if out.ndim != 2 or out.shape[0] != 2:
        raise RuntimeError(f"[{name}] backbone produced shape {tuple(out.shape)}, expected (2, D)")
    feature_dim = int(out.shape[1])

    if spec.expected_dim is not None and feature_dim != spec.expected_dim:
        raise RuntimeError(
            f"[{name}] feature dim {feature_dim} != documented {spec.expected_dim}. "
            "The loading recipe or the published weights changed -- not continuing blindly."
        )
    log.info(f"[{name}] feature_dim={feature_dim} (verified by forward pass)")

    info.update({
        "backbone": name,
        "family": spec.family,
        "feature_dim": feature_dim,
        "input_size": spec.input_size,
        "normalization_mean": list(spec.mean),
        "normalization_std": list(spec.std),
        "description": spec.description,
    })
    return backbone, feature_dim, info

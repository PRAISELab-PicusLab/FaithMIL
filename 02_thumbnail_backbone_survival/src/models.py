"""Backbone + 1-D task head.

Architecture is fully determined by `src/backbones.py`; this module only
attaches a freshly-initialized linear head and handles freezing. That keeps
"which pretraining" the single varying factor across experiments.

Loading is 100% offline: see src/backbones.py. No torch.hub / HF / network
access happens at train time, as required on Leonardo's compute nodes.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from src.backbones import (
    BACKBONE_NAMES,
    FOUNDATION_BACKBONES,
    RESNET50_BACKBONES,
    build_backbone,
    get_spec,
)

TASK_OUTPUT_DIM = {"classification": 1, "regression": 1, "survival": 1}

__all__ = [
    "BACKBONE_NAMES",
    "FOUNDATION_BACKBONES",
    "RESNET50_BACKBONES",
    "TASK_OUTPUT_DIM",
    "BackboneTaskModel",
    "build_model",
]


class BackboneTaskModel(nn.Module):
    """Pretrained backbone (frozen or trainable) + a fresh Linear(dim -> 1)."""

    def __init__(
        self,
        backbone_name: str,
        task: str,
        pretrained_dir: Path | str,
        freeze_backbone: bool | None = None,
        logger: logging.Logger | None = None,
        amp: bool = False,
    ):
        super().__init__()
        log = logger or logging.getLogger(__name__)
        if task not in TASK_OUTPUT_DIM:
            raise ValueError(f"Unknown task '{task}'. Expected one of {list(TASK_OUTPUT_DIM)}")

        spec = get_spec(backbone_name)
        backbone, feature_dim, self.load_info = build_backbone(backbone_name, pretrained_dir, log)
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, TASK_OUTPUT_DIM[task])
        self.amp = amp

        # Foundation models default to frozen: 0.3B-1.1B parameters cannot be
        # meaningfully fine-tuned on ~450 training thumbnails. An explicit CLI
        # flag always wins over that default.
        if freeze_backbone is None:
            self.freeze_backbone = spec.default_freeze
            log.info(
                f"[{backbone_name}] freeze_backbone not specified, using backbone default "
                f"({self.freeze_backbone})"
            )
        else:
            self.freeze_backbone = freeze_backbone

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            log.info(f"[{backbone_name}] backbone FROZEN (kept in eval mode), only the head is trainable")
        else:
            log.info(f"[{backbone_name}] full fine-tuning: backbone + head are trainable")

        n_total = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"[{backbone_name}] total_params={n_total:,} trainable_params={n_trainable:,}")

        self.load_info.update({
            "task": task,
            "head_randomly_initialized": True,
            "freeze_backbone": self.freeze_backbone,
            "total_params": n_total,
            "trainable_params": n_trainable,
            "amp": amp,
        })

    def train(self, mode: bool = True):
        """A frozen backbone stays in eval mode even while the head trains.

        requires_grad=False alone does NOT stop BatchNorm from updating its
        running statistics, nor dropout/droppath from perturbing features, so
        without this override a "frozen" backbone would still drift between
        epochs and produce non-deterministic features.
        """
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            # no_grad also avoids storing activations, which is what makes
            # ViT-g/ViT-H backbones fit comfortably in memory.
            with torch.no_grad():
                feats = self._extract(x)
            feats = feats.detach()
        else:
            feats = self._extract(x)
        return self.head(feats.float()).squeeze(-1)

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        if self.amp and x.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return self.backbone(x)
        return self.backbone(x)


def build_model(
    backbone_name: str,
    task: str,
    pretrained_dir: Path | str,
    freeze_backbone: bool | None = None,
    logger: logging.Logger | None = None,
    amp: bool = False,
) -> BackboneTaskModel:
    return BackboneTaskModel(backbone_name, task, pretrained_dir, freeze_backbone, logger, amp)

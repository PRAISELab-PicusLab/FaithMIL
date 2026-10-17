"""DINO self-supervised pre-training of one ViT-S/14 from random initialisation on the tiles of one level.

Recipe = original DINO (Caron et al., 2021), single GPU:
  * student / teacher ViT-S/14 (384-d, the architecture of Prost40M), the teacher is the EMA of the student and is the
    encoder that is kept;
  * 2 global crops (224 px) + 8 local crops (98 px = 7 patches; 96 is not a multiple of 14);
  * DINO head 384-2048-2048-256 + weight-normed 65536-way layer, centering + sharpening;
  * AdamW, lr 5e-4 * batch / 256 (linear warm-up, cosine), weight decay 0.04 -> 0.4 (cosine), EMA momentum
    0.996 -> 1 (cosine), gradient clipping 3.0, last layer frozen in epoch 0, bf16.

Augmentations run on the GPU in batch (the CPUs only read uint8 tiles): per-tile dihedral (rot90 + flip), random-resized
crop (roi_align), brightness / contrast / saturation jitter and per-channel gain (stain-like, replaces hue), grayscale
p=0.1, gaussian blur (p = 1.0 / 0.1 for the two global crops, 0.5 for the local ones). No solarization.

    encoders/dino_L<k>/teacher_backbone.pth   the encoder (state dict of the teacher's ViT)
    encoders/dino_L<k>/ckpt_last.pt           full training state, the run resumes from it
    encoders/dino_L<k>/history.csv            loss, teacher entropy and feature std per epoch
"""
from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from .layout import Layout

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
GLOBAL, LOCAL = 224, 98
EPOCHS = {2: 18, 1: 6, 0: 6, 3: 15}  # epochs used for the paper (about 14 M tiles seen at every level)


# ----------------------------------------------------------------------------- data
class TileBank:
    """All tiles of all cases. Batches of uint8 (B, 224, 224, 3) tiles are read from memmaps by
    a thread pool (no worker processes: forking after the CUDA / OpenMP init deadlocked)."""

    def __init__(self, tiles_dir: Path, cases, threads=64, prefetch=24):
        self.paths = [Path(tiles_dir) / f"{c}.npy" for c in cases]
        maps = [np.load(p, mmap_mode="r") for p in self.paths]
        self.offsets = np.concatenate([[0], np.cumsum([m.shape[0] for m in maps])])
        self.header = [m.offset for m in maps]  # byte offset of the data in each .npy
        del maps
        self._fds: dict = {}
        self.pool = ThreadPoolExecutor(threads)
        self.prefetch = prefetch

    def __len__(self):
        return int(self.offsets[-1])

    def _fd(self, c):
        fd = self._fds.get(c)
        if fd is None:
            fd = self._fds.setdefault(c, os.open(self.paths[c], os.O_RDONLY))
        return fd

    def _read(self, idx):
        """pread (releases the GIL; random 150 kB reads scale with the number of threads: ~50 ms each)."""
        out = np.empty((len(idx), 224, 224, 3), np.uint8)
        cs = np.searchsorted(self.offsets, idx, side="right") - 1
        size = 224 * 224 * 3
        for k, (i, c) in enumerate(zip(idx, cs)):
            c = int(c)
            n = os.preadv(self._fd(c), [memoryview(out[k]).cast("B")], self.header[c] + int(i - self.offsets[c]) * size)
            if n != size:
                raise IOError(f"short read {self.paths[c]} tile {i - self.offsets[c]}: {n}")
        return torch.from_numpy(out).pin_memory()

    def epoch(self, batch_size, seed):
        """Random permutation of all tiles, drop_last batches, `prefetch` batches in flight."""
        perm = np.random.default_rng(seed).permutation(len(self))
        n = len(perm) // batch_size
        futs = []
        for b in range(n):
            futs.append(self.pool.submit(self._read, np.sort(perm[b * batch_size:(b + 1) * batch_size])))
            if len(futs) > self.prefetch:
                yield futs.pop(0).result()
        while futs:
            yield futs.pop(0).result()


# ----------------------------------------------------------------------------- GPU augmentation
def dihedral(x):
    B = x.shape[0]
    k = torch.randint(0, 4, (B,), device=x.device)
    fl = torch.rand(B, device=x.device) < 0.5
    out = x.clone()
    for r in (1, 2, 3):
        m = k == r
        if m.any():
            out[m] = torch.rot90(x[m], r, (2, 3))
    out[fl] = out[fl].flip(3)
    return out


def crop(x, n, scale, size, ratio=(3 / 4, 4 / 3)):
    """n random-resized crops per tile -> list of n tensors [B, 3, size, size] (roi_align)."""
    B, _, H, W = x.shape
    dev = x.device
    outs = []
    for _ in range(n):
        s = torch.empty(B, device=dev).uniform_(*scale)
        r = torch.exp(torch.empty(B, device=dev).uniform_(math.log(ratio[0]), math.log(ratio[1])))
        w = torch.sqrt(s * H * W * r).clamp(max=W)
        h = torch.sqrt(s * H * W / r).clamp(max=H)
        x0, y0 = torch.rand(B, device=dev) * (W - w), torch.rand(B, device=dev) * (H - h)
        boxes = torch.stack([torch.arange(B, device=dev, dtype=x.dtype), x0, y0, x0 + w, y0 + h], 1)
        outs.append(roi_align(x, boxes, output_size=size, spatial_scale=1.0, sampling_ratio=2, aligned=True))
    return outs


def jitter(x, p=0.8, b=0.4, c=0.4, s=0.2, gain=0.08, p_gray=0.1):
    N, dev = x.shape[0], x.device

    def f(m):
        return 1 + (torch.rand(N, 1, 1, 1, device=dev) * 2 - 1) * m

    y = x * f(b)
    mean = y.mean((1, 2, 3), keepdim=True)
    y = (y - mean) * f(c) + mean
    gray = (0.299 * y[:, 0:1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:3])
    y = (y - gray) * f(s) + gray
    y = (y * (1 + (torch.rand(N, 3, 1, 1, device=dev) * 2 - 1) * gain)).clamp_(0, 1)
    x = torch.where(torch.rand(N, 1, 1, 1, device=dev) < p, y, x)
    g = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).expand(-1, 3, -1, -1)
    return torch.where(torch.rand(N, 1, 1, 1, device=dev) < p_gray, g, x)


def blur(x, p, sigma=(0.1, 2.0), k=13):
    N, C, H, W = x.shape
    sel = torch.rand(N, device=x.device) < p
    n = int(sel.sum())
    if n == 0:
        return x
    sig = torch.empty(n, device=x.device).uniform_(*sigma)
    ax = (torch.arange(k, device=x.device) - k // 2).float()
    ker = torch.exp(-(ax[None] ** 2) / (2 * sig[:, None] ** 2))
    ker = (ker / ker.sum(1, keepdim=True)).repeat_interleave(C, 0)  # [n*C, k]
    z = x[sel].reshape(1, n * C, H, W)
    z = F.conv2d(F.pad(z, (k // 2, k // 2, 0, 0), mode="reflect"), ker.view(n * C, 1, 1, k), groups=n * C)
    z = F.conv2d(F.pad(z, (0, 0, k // 2, k // 2), mode="reflect"), ker.view(n * C, 1, k, 1), groups=n * C)
    x = x.clone()
    x[sel] = z.view(n, C, H, W)
    return x


def make_views(u8, n_local, g_scale, l_scale, mean, std):
    """uint8 [B,224,224,3] on GPU -> (list of 2 global [B,3,224,224], list of n_local local [B,3,98,98])."""
    x = dihedral(u8.permute(0, 3, 1, 2).float().div_(255.0))
    g = [jitter(v) for v in crop(x, 2, g_scale, GLOBAL)]
    g = [blur(g[0], 1.0), blur(g[1], 0.1)]
    loc = [blur(jitter(v), 0.5) for v in crop(x, n_local, l_scale, LOCAL)]
    norm = lambda t: (t - mean) / std
    return [norm(v) for v in g], [norm(v) for v in loc]


# ----------------------------------------------------------------------------- model
def make_vit():
    """ViT-S/14 for 224-px tiles (384-d), the architecture of Prost40M; no download needed."""
    import timm
    return timm.create_model("vit_small_patch16_224", pretrained=False, patch_size=14, img_size=224, num_classes=0,
                             dynamic_img_size=True)


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=2048, bottleneck=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, bottleneck))
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        self.last = nn.utils.weight_norm(nn.Linear(bottleneck, out_dim, bias=False))
        self.last.weight_g.data.fill_(1)
        self.last.weight_g.requires_grad = False

    def forward(self, x):
        return self.last(F.normalize(self.mlp(x), dim=-1, p=2))


class Net(nn.Module):
    """backbone + head with the multi-crop trick: one backbone call per crop resolution."""

    def __init__(self, out_dim):
        super().__init__()
        self.backbone = make_vit()
        self.head = DINOHead(self.backbone.num_features, out_dim)

    def forward(self, crops):
        feats = torch.cat([self.backbone(torch.cat(g)) for g in crops])  # crops = [globals, locals] (locals may be empty)
        return self.head(feats), feats


class DINOLoss(nn.Module):
    def __init__(self, out_dim, n_crops, student_temp=0.1, center_m=0.9):
        super().__init__()
        self.n_crops, self.st, self.cm = n_crops, student_temp, center_m
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, s_out, t_out, t_temp):
        s = (s_out.float() / self.st).chunk(self.n_crops)
        t_out = t_out.float()
        t = F.softmax((t_out - self.center) / t_temp, dim=-1).detach()
        tq = t.chunk(2)
        loss, n = 0.0, 0
        for iq, q in enumerate(tq):
            for v in range(self.n_crops):
                if v == iq:
                    continue
                loss = loss + torch.sum(-q * F.log_softmax(s[v], dim=-1), dim=-1).mean()
                n += 1
        with torch.no_grad():
            self.center.mul_(self.cm).add_(t_out.mean(0, keepdim=True), alpha=1 - self.cm)
            ent = -(t * torch.log(t + 1e-12)).sum(-1).mean()
        return loss / n, ent


def cosine(base, final, total, warm=0, warm_start=0.0):
    w = np.linspace(warm_start, base, warm) if warm else np.array([])
    it = np.arange(total - warm)
    c = final + 0.5 * (base - final) * (1 + np.cos(np.pi * it / max(len(it), 1)))
    return np.concatenate([w, c])


# ----------------------------------------------------------------------------- main
def train(layout: Layout, level: int, epochs: int | None = None, batch_size: int = 256, warmup_epochs: float = 1.0,
          lr: float = 5e-4, min_lr: float = 1e-6, wd: float = 0.04, wd_end: float = 0.4, momentum: float = 0.996,
          teacher_temp: float = 0.04, out_dim: int = 65536, local_crops: int = 8, global_scale=(0.4, 1.0),
          local_scale=(0.05, 0.4), clip_grad: float = 3.0, freeze_last_epochs: int = 1, num_workers: int = 64,
          time_budget_h: float = 1e9, seed: int = 0, log_every: int = 50, max_steps: int = 0, debug: bool = False,
          tiles_dir: Path | None = None, out: Path | None = None):
    epochs = EPOCHS[level] if epochs is None else epochs
    tiles_dir = Path(tiles_dir or layout.tiles(level))
    out = Path(out or layout.encoder(level))
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")

    cases = sorted(p.stem for p in tiles_dir.glob("*.npy") if not p.name.endswith((".tmp.npy", "_idx.npy")))
    if not cases:
        raise FileNotFoundError(f"no tiles in {tiles_dir} (run scripts/02_extract_tiles.py --level {level})")
    if debug:
        cases = cases[::max(1, len(cases) // 40)]
        epochs, max_steps, log_every = 1, 100, 10
    bank = TileBank(tiles_dir, cases, threads=num_workers)
    print(f"{len(cases)} cases, {len(bank)} tiles", flush=True)
    spe = len(bank) // batch_size
    if spe < 1:
        raise ValueError(f"{len(bank)} tiles are fewer than one batch of {batch_size}; lower --batch-size")
    total = epochs * spe

    student, teacher = Net(out_dim).to(dev), Net(out_dim).to(dev)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    n_crops = 2 + local_crops
    criterion = DINOLoss(out_dim, n_crops).to(dev)
    reg = [p for n, p in student.named_parameters() if p.requires_grad and not (n.endswith(".bias") or p.ndim == 1)]
    noreg = [p for n, p in student.named_parameters() if p.requires_grad and (n.endswith(".bias") or p.ndim == 1)]
    opt = torch.optim.AdamW([{"params": reg}, {"params": noreg, "weight_decay": 0.0}], lr=0.0)
    lr_s = cosine(lr * batch_size / 256, min_lr, total, int(warmup_epochs * spe))
    wd_s = cosine(wd, wd_end, total)
    mom_s = cosine(momentum, 1.0, total)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)

    start_epoch, ckpt = 0, out / "ckpt_last.pt"
    hist_path = out / "history.csv"
    if ckpt.exists() and not debug:
        st = torch.load(ckpt, map_location=dev)
        student.load_state_dict(st["student"])
        teacher.load_state_dict(st["teacher"])
        opt.load_state_dict(st["opt"])
        criterion.center.copy_(st["center"])
        start_epoch = st["epoch"]
        print(f"resumed from epoch {start_epoch}", flush=True)
    else:
        hist_path.write_text("epoch,loss,teacher_entropy,feat_std,steps,seconds\n")
    (out / "config.json").write_text(json.dumps({"level": level, "epochs": epochs, "batch_size": batch_size, "lr": lr,
                                                 "n_tiles": len(bank), "n_cases": len(cases), "steps_per_epoch": spe,
                                                 "seed": seed}, indent=1))

    t_start = time.time()
    step_total = start_epoch * spe
    stop = False
    for epoch in range(start_epoch, epochs):
        student.train()
        teacher.train()  # no dropout / drop-path: train == eval for the ViT
        t_ep = time.time()
        sums = np.zeros(3)
        n_log = 0
        for it, u8 in enumerate(bank.epoch(batch_size, seed + 1000 * epoch)):
            u8 = u8.to(dev, non_blocking=True)
            for i, g in enumerate(opt.param_groups):
                g["lr"] = lr_s[step_total]
                if i == 0:
                    g["weight_decay"] = wd_s[step_total]
            with torch.no_grad():
                gl, lc = make_views(u8, local_crops, global_scale, local_scale, mean, std)
                if debug and it == 0:
                    save_grid(gl, lc, out / "aug_grid.png", mean, std)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    t_out, t_feat = teacher([gl])
                s_out, _ = student([gl, lc])
            loss, ent = criterion(s_out, t_out, teacher_temp)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step_total}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for n, p in student.named_parameters():
                if p.grad is not None:
                    torch.nn.utils.clip_grad_norm_(p, clip_grad)
                    if epoch < freeze_last_epochs and "head.last" in n:
                        p.grad = None
            opt.step()
            with torch.no_grad():
                m = mom_s[step_total]
                torch._foreach_mul_(list(teacher.parameters()), m)
                torch._foreach_add_(list(teacher.parameters()), list(student.parameters()), alpha=1 - m)
                fstd = t_feat.float().std(0).mean().item() if it % log_every == 0 else None
            if it % log_every == 0:
                sums += [loss.item(), ent.item(), fstd]
                n_log += 1
                el = time.time() - t_ep
                print(f"ep {epoch:03d} it {it:05d}/{spe} loss {loss.item():.4f} teacher-ent {ent.item():.3f} "
                      f"feat-std {fstd:.3f} lr {lr_s[step_total]:.2e} | {(it + 1) * batch_size / max(el, 1e-9):.0f} tiles/s",
                      flush=True)
            step_total += 1
            if max_steps and step_total >= max_steps:
                stop = True
                break
            if (time.time() - t_start) / 3600 > time_budget_h:
                stop = True
                break
        s = sums / max(n_log, 1)
        with open(hist_path, "a") as fh:
            fh.write(f"{epoch},{s[0]:.5f},{s[1]:.4f},{s[2]:.4f},{it + 1},{time.time() - t_ep:.0f}\n")
        done = epoch + 1 if not stop else epoch + (1 if it + 1 == spe else 0)
        if not debug:
            torch.save({"student": student.state_dict(), "teacher": teacher.state_dict(), "opt": opt.state_dict(),
                        "center": criterion.center, "epoch": done}, ckpt.with_suffix(".tmp"))
            ckpt.with_suffix(".tmp").replace(ckpt)
        torch.save(teacher.backbone.state_dict(), out / "teacher_backbone.pth")  # overwritten every epoch: the last one is kept
        print(f"epoch {epoch} done in {(time.time() - t_ep) / 60:.1f} min", flush=True)
        if stop:
            print("stopping (budget / max-steps)", flush=True)
            break
    print("finished", flush=True)


def save_grid(gl, lc, path, mean, std):
    from PIL import Image
    den = lambda t: (t * std + mean).clamp(0, 1)
    rows = []
    for b in range(min(4, gl[0].shape[0])):
        cells = [den(g[b:b + 1])[0] for g in gl] + [F.interpolate(den(l[b:b + 1]), size=GLOBAL // 2)[0] for l in lc[:4]]
        cells = [F.interpolate(c[None], size=GLOBAL // 2)[0] if c.shape[-1] != GLOBAL // 2 else c for c in cells]
        rows.append(torch.cat(cells, 2))
    img = (torch.cat(rows, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save(path)

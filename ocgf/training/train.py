"""Student training loop: seeded, mixed precision on GPU, a checkpoint every epoch, automatic resume, a time budget.

Checkpoints (<out>/ckpt/last.pt, with ckpt/progress.json naming the last finished epoch) hold everything but the
frozen backbone (reloaded from torch.hub), so a Kaggle session that stops (12 h limit) resumes where it left off. The validation loss, if a validation set is given, is
measured against the teachers' pseudo-labels: it shows whether training converges and is never a reported metric or
used for model selection (CLAUDE.md: never evaluate against pseudo-labels).
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .data import collate, to_device
from .losses import LossWeights, student_loss

FROZEN_PREFIX = "camera.backbone."


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    num_workers: int = 4
    amp: bool = True
    log_every: int = 50
    time_budget_hours: float = 11.0

    @classmethod
    def from_dict(cls, d: dict | None) -> TrainConfig:
        return cls(**dict(d or {}))

    def to_dict(self) -> dict:
        return asdict(self)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _worker_init(worker_id: int) -> None:
    np.random.seed(torch.initial_seed() % 2 ** 32)          # each worker its own numpy stream


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                       collate_fn=collate, worker_init_fn=_worker_init, generator=g,
                                       drop_last=shuffle and len(dataset) >= batch_size,
                                       pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)


def warmup_cosine(total_steps: int, warmup_steps: int):
    def factor(step: int) -> float:
        warm = min(1.0, (step + 1) / max(warmup_steps, 1))
        return warm * 0.5 * (1.0 + math.cos(math.pi * min(step, total_steps) / max(total_steps, 1)))
    return factor


def trainable_state(model) -> dict:
    return {k: v for k, v in model.state_dict().items() if not k.startswith(FROZEN_PREFIX)}


def load_trainable_state(model, state: dict) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith(FROZEN_PREFIX)]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint does not match the model: missing {missing[:5]}, unexpected {unexpected[:5]}")


def save_checkpoint(path: Path, **state) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)                                    # a killed session never leaves a half-written file


def _mean(sums: dict, n: int) -> dict:
    return {k: round(v / max(n, 1), 5) for k, v in sums.items()}


@torch.no_grad()
def evaluate_loss(model, loader, device, weights: LossWeights, amp: bool) -> dict:
    model.eval()
    sums, n = {}, 0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(batch)
        total, parts = student_loss(out, batch, weights)
        for k, v in {"loss": total, **parts}.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        n += 1
    return _mean(sums, n)


def train(model, train_set, val_set, cfg: TrainConfig, weights: LossWeights, out_dir, device=None, seed: int = 0,
          smoke: int = 0, run_info: dict | None = None, log=print) -> dict:
    """Train with resume from <out_dir>/ckpt/last.pt. smoke=N: N steps of one epoch, then a reload check."""
    out_dir, t_start = Path(out_dir), time.time()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_everything(seed)
    model.to(device)
    amp = cfg.amp and device.type == "cuda"
    loader = make_loader(train_set, cfg.batch_size, True, cfg.num_workers, seed)
    val_loader = make_loader(val_set, cfg.batch_size, False, cfg.num_workers, seed) if val_set is not None else None
    epochs = 1 if smoke else cfg.epochs
    steps_per_epoch = min(len(loader), smoke) if smoke else len(loader)
    total = epochs * steps_per_epoch
    optim = torch.optim.AdamW(model.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, warmup_cosine(total, cfg.warmup_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    n_params = sum(p.numel() for p in model.trainable_parameters())
    log(f"student {model.cfg.mode}: {n_params:,} trainable parameters on {device}; {len(train_set)} train frames, "
        f"{len(val_set) if val_set is not None else 0} val frames; {steps_per_epoch} steps per epoch, {epochs} epochs; "
        f"{'mixed precision' if amp else 'float32'}")

    last = out_dir / "ckpt" / "last.pt"
    start_epoch = 0
    if last.exists() and not smoke:
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        load_trainable_state(model, ckpt["model"])
        optim.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        if amp and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        if int(ckpt.get("epochs", epochs)) != epochs:
            log(f"epochs changed from {ckpt.get('epochs')} to {epochs}: the cosine now ends at epoch {epochs}")
        log(f"resumed from {last}: {start_epoch} of {epochs} epochs done")

    def write_metrics(entry):
        with open(out_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

    epoch_seconds, result = [], {"epochs_done": start_epoch, "epochs": epochs}
    for epoch in range(start_epoch, epochs):
        t_epoch = time.time()
        model.train()
        sums, n, since, wait, compute, tick = {}, 0, 0, 0.0, 0.0, time.time()
        for step, batch in enumerate(loader):
            if step >= steps_per_epoch:
                break
            batch = to_device(batch, device)
            got = time.time()
            wait += got - tick
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(batch)
            loss, parts = student_loss(out, batch, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {step}: "
                                         f"{ {k: float(v) for k, v in parts.items()} }")
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
            for k, v in {"loss": loss, **parts}.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            n += 1
            since += 1
            tick = time.time()
            compute += tick - got
            if step % cfg.log_every == 0 or step == steps_per_epoch - 1:
                log(f"  epoch {epoch} step {step}/{steps_per_epoch}: " +
                    ", ".join(f"{name} {v:.4f}" for name, v in _mean(sums, n).items()) +
                    f" | lr {optim.param_groups[0]['lr']:.2e} | per step: data wait {wait / since:.2f} s, "
                    f"compute {compute / since:.2f} s")
                since, wait, compute = 0, 0.0, 0.0
        entry = {"epoch": epoch, "train": _mean(sums, n), "lr": optim.param_groups[0]["lr"],
                 "seconds": round(time.time() - t_epoch, 1)}
        if val_loader is not None:
            entry["val_loss_vs_teacher"] = evaluate_loss(model, val_loader, device, weights, amp)
        save_checkpoint(last, model=trainable_state(model), optimizer=optim.state_dict(), scheduler=sched.state_dict(),
                        scaler=scaler.state_dict() if amp else None, epoch=epoch, epochs=epochs,
                        config=(run_info or {}).get("config"))
        (last.parent / "progress.json").write_text(json.dumps({"epoch": epoch, "epochs": epochs}))
        write_metrics(entry)
        epoch_seconds.append(time.time() - t_epoch)
        result["epochs_done"] = epoch + 1
        log(f"epoch {epoch} done in {entry['seconds']:.0f} s: train {entry['train']}" +
            (f"; val loss vs teacher (debug only) {entry['val_loss_vs_teacher']}" if val_loader is not None else ""))
        elapsed = (time.time() - t_start) / 3600
        if elapsed + np.mean(epoch_seconds) / 3600 > cfg.time_budget_hours and epoch + 1 < epochs:
            log(f"time budget: {elapsed:.1f} h used of {cfg.time_budget_hours} h; stopping after epoch {epoch}, "
                f"run again to resume")
            result["stopped_for_time"] = True
            break

    if smoke:                                                # the checkpoint must load into a fresh model
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        fresh = type(model)(model.cfg, backbone=getattr(getattr(model, "camera", None), "backbone", None))
        load_trainable_state(fresh, ckpt["model"])
        result["reload_check"] = "ok"
        log("smoke run: checkpoint reloads into a fresh model")
    result["finished"] = result["epochs_done"] >= epochs
    return result

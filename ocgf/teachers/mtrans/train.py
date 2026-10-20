# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# From MTrans (Liu et al., ECCV 2022; github.com/Cliu2/MTrans) via the team fork github.com/GNaveenkumar614/Transmodel,
# commit ebf8a90 (2026-10-06): train.py and the Kaggle driver train_kaggle.py written in its mtrans_kaggle.ipynb.
# Modified for OCGF:
#   - object pools from the KITTI train/val split: labelled objects from the frozen 500-frame list only, unlabelled
#     objects from zero-shot detections on the other train frames, KITTI val (human labels) for evaluation only;
#   - no best-checkpoint selection: KITTI val labels must not steer training (they would add labels beyond the
#     500-frame budget), so the last epoch of the fixed schedule is the result; val is evaluated for reporting;
#   - per-class evaluation (mean 3D IoU, recall at 0.5 / 0.7, confidence error) and per-object val results;
#   - a checkpoint after every epoch with resume, a wall-clock budget and a smoke mode (as in train_kaggle.py);
#   - stops on a non-finite loss; JSON-lines logs instead of TensorBoard; YAML config without easydict.
"""Training and evaluation of the multi-class MTrans teacher on KITTI 3D Object."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ...datasets.kitti_object import KittiObjectFrames, kitti_fid
from ..detector2d.mapping import read_detections
from .data import (CLASSES, ObjectDataset, ProjectedFrames, build_records, class_anchors, class_balanced_sampler,
                   detection_objects, label_objects, make_loader, merge_batches, to_device)
from .model import MTrans
from .ops import AttrDict, to_attrdict
from .schedule import WarmupCosineAnnealing

REPO = Path(__file__).resolve().parents[3]
KITTI_SPLITS = REPO / "configs" / "splits" / "kitti_object"


def load_config(path) -> AttrDict:
    return to_attrdict(yaml.safe_load(Path(path).read_text()))


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- object pools

def build_pools(cfg, kitti_folders: dict, detections_file, cache_dir, out_dir, log=print,
                frame_lists: dict | None = None) -> AttrDict:
    """Labelled, unlabelled and validation records plus class anchors, saved as out_dir/records.json and reused when
    the frame lists, the detections and the data settings are unchanged.

    frame_lists: {"train", "val", "labelled": [fid, ...]}; default: the frozen KITTI split and cfg.labelled_frames."""
    log = log or (lambda msg: None)
    out_dir = Path(out_dir)
    if frame_lists is None:
        read = lambda path: [kitti_fid(f) for f in Path(path).read_text().split()]
        frame_lists = {"train": read(KITTI_SPLITS / "train.txt"), "val": read(KITTI_SPLITS / "val.txt"),
                       "labelled": read(REPO / cfg.labelled_frames)}
    train, val, labelled = (list(frame_lists[k]) for k in ("train", "val", "labelled"))
    digest = lambda fids: hashlib.sha256("\n".join(fids).encode()).hexdigest()
    key = {"labelled_frames": cfg.labelled_frames, "labelled_sha256": digest(labelled), "train_sha256": digest(train),
           "val_sha256": digest(val), "detections_sha256": sha256_file(detections_file),
           "data": json.loads(json.dumps(cfg.data)), "classes": list(cfg.classes)}
    cached = out_dir / "records.json"
    if cached.exists():
        data = json.loads(cached.read_text())
        if data.get("key") == key:
            log(f"object records reused from {cached}")
            return to_attrdict(data)
        log(f"{cached} was built from other inputs or settings; building again")

    if not set(labelled) <= set(train):
        raise ValueError("labelled frames must come from the KITTI train split")
    labelled_set = set(labelled)
    unlabelled = [f for f in train if f not in labelled_set]

    d, classes = cfg.data, tuple(cfg.classes)
    source = KittiObjectFrames(kitti_folders)
    frames = ProjectedFrames(source, cache_dir)
    log(f"labelled objects (human 3D labels) from {len(labelled)} frames")
    labelled_records = build_records(frames, label_objects(source, labelled, d.max_truncation), classes,
                                     d.min_points, d.min_foreground, log)
    log(f"unlabelled objects (detector boxes, score >= {d.unlabelled_min_score}) from {len(unlabelled)} frames")
    unlabelled_records = build_records(frames, detection_objects(read_detections(detections_file), unlabelled,
                                                                 d.unlabelled_min_score, classes),
                                       classes, d.min_points, d.min_foreground, log)
    log(f"evaluation objects (KITTI val, human labels) from {len(val)} frames")
    val_records = build_records(frames, label_objects(source, val, d.max_truncation), classes,
                                d.min_points, d.min_foreground, log)

    count = lambda recs: {c: sum(r["cls"] == c for r in recs) for c in classes}
    data = {"key": key, "anchors": class_anchors(labelled_records, classes),
            "counts": {"labelled": count(labelled_records), "unlabelled": count(unlabelled_records),
                       "val": count(val_records)},
            "labelled": labelled_records, "unlabelled": unlabelled_records, "val": val_records}
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, cached)
    log(f"records: {json.dumps(data['counts'])}; anchors (l, w, h): {data['anchors']}")
    return to_attrdict(data)


# ---------------------------------------------------------------- evaluation

def summarize(classes, cls_idx, iou, conf) -> dict:
    """Per-class and overall mean 3D IoU, recall at IoU 0.5 / 0.7 and mean |confidence - IoU|."""
    def stats(mask):
        if not mask.any():
            return {"n": 0}
        i, c = iou[mask], conf[mask]
        return {"n": int(mask.sum()), "mean_iou3d": round(float(i.mean()), 4), "recall_0.5": round(float((i > 0.5).mean()), 4),
                "recall_0.7": round(float((i > 0.7).mean()), 4), "conf_abs_err": round(float(np.abs(c - i).mean()), 4)}
    out = {c: stats(cls_idx == k) for k, c in enumerate(classes)}
    out["all"] = stats(np.ones(len(iou), dtype=bool))
    return out


@torch.no_grad()
def evaluate(model, loader, device, classes, per_object_file=None) -> dict:
    """Objects with human 3D labels, given their ground-truth 2D boxes (as MTrans reports it)."""
    model.eval()
    cls_idx, ious, confs, rows = [], [], [], []
    for data in loader:
        data = to_device(data, device)
        pred = model(data)
        _, _, iou, conf = model.get_loss(pred, data)
        labelled = data.use_3d_label.cpu().numpy()
        c = data.classes.cpu().numpy()[labelled]
        cls_idx.append(c)
        ious.append(iou.cpu().numpy())
        confs.append(conf.cpu().numpy())
        if per_object_file:
            fids = [f for f, keep in zip(data.fids, labelled) if keep]
            rows += [{"fid": f, "cls": classes[k], "iou3d": round(float(i), 4), "conf": round(float(p), 4)}
                     for f, k, i, p in zip(fids, c, ious[-1], confs[-1])]
    result = summarize(classes, np.concatenate(cls_idx), np.concatenate(ious), np.concatenate(confs))
    if per_object_file:
        Path(per_object_file).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return result


# ---------------------------------------------------------------- training

def save_checkpoint(path: Path, **state) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)                       # a killed session never leaves a half-written checkpoint


def train(cfg, pools, kitti_folders: dict, cache_dir, out_dir, device=None, time_budget_hours: float = 11.0,
          smoke: int = 0, log=print) -> dict:
    """Train with resume from out_dir/ckpt/last.pt. smoke=N: one epoch on N objects per pool, then a reload check."""
    log = log or (lambda msg: None)
    out_dir, t_start = Path(out_dir), time.time()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    t, d, classes = cfg.train, cfg.data, tuple(cfg.classes)
    epochs = 1 if smoke else int(t.epochs)
    seed_everything(int(cfg.seed))
    take = (lambda recs: list(recs)[:smoke]) if smoke else list

    frames = ProjectedFrames(KittiObjectFrames(kitti_folders), cache_dir)
    anchors = pools.anchors
    labelled = ObjectDataset(take(pools.labelled), frames, anchors, d.out_img_size, d.out_cloud_size,
                             d.train_mask_ratio, d.box2d_augmentation)
    unlabelled = ObjectDataset(take(pools.unlabelled), frames, anchors, d.out_img_size, d.out_cloud_size,
                               d.train_mask_ratio, d.box2d_augmentation)
    val = ObjectDataset(take(pools.val), frames, anchors, d.out_img_size, d.out_cloud_size)
    sampler = class_balanced_sampler(labelled.records, float(d.class_balance_power), int(cfg.seed)) \
        if d.class_balance_power else None
    workers = int(t.num_workers)
    lab_loader = make_loader(labelled, int(t.batch_size), shuffle=True, sampler=sampler, num_workers=workers,
                             seed=int(cfg.seed))
    unl_loader = make_loader(unlabelled, int(t.unlabelled_batch_size), shuffle=True, num_workers=workers,
                             seed=int(cfg.seed) + 1) if len(unlabelled) and int(t.unlabelled_batch_size) else None
    val_loader = make_loader(val, int(t.eval_batch_size), num_workers=workers)

    model = MTrans(cfg.model, anchors).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=float(t.lr))
    scheduler = WarmupCosineAnnealing(optim, float(t.lr), float(t.warmup_rate), epochs * len(lab_loader), eta_min=0)
    scheduler.step()
    log(f"MTrans: {sum(p.numel() for p in model.parameters()):,} parameters on {device}; objects: "
        f"{len(labelled)} labelled, {len(unlabelled)} unlabelled, {len(val)} val; {len(lab_loader)} steps per epoch")

    ckpt_dir = out_dir / "ckpt"
    last = ckpt_dir / "last.pt"
    start_epoch = 0
    if last.exists() and not smoke:
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        log(f"resumed from {last}: {start_epoch} of {epochs} epochs done")
    metrics_log = out_dir / "metrics.jsonl"

    def write_metrics(entry):
        with open(metrics_log, "a") as f:
            f.write(json.dumps(entry) + "\n")

    epoch_seconds, result = [], {"epochs_done": start_epoch, "epochs": epochs}
    for epoch in range(start_epoch, epochs):
        t_epoch = time.time()
        model.train()
        totals, counts = {}, {}
        unl_iter = iter(unl_loader) if unl_loader else None
        for step, data in enumerate(lab_loader):
            if unl_iter is not None:
                try:
                    extra = next(unl_iter)
                except StopIteration:
                    unl_iter = iter(unl_loader)
                    extra = next(unl_iter)
                data = merge_batches(data, extra)
            data = to_device(data, device)
            optim.zero_grad()
            pred = model(data)
            loss, stats, _, _ = model.get_loss(pred, data)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {step}: {stats}")
            loss.backward()
            optim.step()
            scheduler.step()
            for k, (v, n) in stats.items():
                totals[k] = totals.get(k, 0.0) + v * n
                counts[k] = counts.get(k, 0) + n
            if step % int(t.log_every) == 0:
                avg = {k: totals[k] / counts[k] for k in totals}
                log(f"  epoch {epoch} step {step}/{len(lab_loader)}: loss {avg['loss']:.3f}, "
                    f"IoU {avg.get('iou3d', float('nan')):.3f}, xyz err {avg['err_dist']:.3f} m")
        train_stats = {k: round(totals[k] / counts[k], 5) for k in totals}
        entry = {"epoch": epoch, "train": train_stats, "lr": optim.param_groups[0]["lr"]}

        is_last = epoch + 1 == epochs
        if is_last or (int(t.eval_every) and (epoch + 1) % int(t.eval_every) == 0):
            per_object = out_dir / "val_objects.jsonl" if is_last else None
            entry["val"] = evaluate(model, val_loader, device, classes, per_object)
            log(f"  epoch {epoch} val (GT 2D boxes): " + ", ".join(
                f"{c} IoU {v['mean_iou3d']:.3f} R0.7 {v['recall_0.7']:.3f}" for c, v in entry["val"].items() if v["n"]))
        save_checkpoint(last, epoch=epoch, epochs=epochs, model=model.state_dict(), optimizer=optim.state_dict(),
                        scheduler=scheduler.state_dict(), classes=list(classes), anchors=[list(a) for a in anchors],
                        model_config=json.loads(json.dumps(cfg.model)), data_config=json.loads(json.dumps(d)))
        epoch_seconds.append(time.time() - t_epoch)
        entry["seconds"] = round(epoch_seconds[-1], 1)
        write_metrics(entry)
        (ckpt_dir / "progress.json").write_text(json.dumps({"epoch": epoch, "epochs": epochs}))
        result = {"epochs_done": epoch + 1, "epochs": epochs, "last": entry}
        left = epochs - epoch - 1
        log(f"epoch {epoch} took {epoch_seconds[-1] / 60:.1f} min; {left} left, about "
            f"{left * np.mean(epoch_seconds) / 3600:.1f} h at this pace")
        if left and (time.time() - t_start) + 1.5 * max(epoch_seconds) > time_budget_hours * 3600:
            log(f"time budget reached after epoch {epoch}; the checkpoint is saved, run again to continue")
            break

    if smoke:
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        reloaded = MTrans(cfg.model, ckpt["anchors"])
        reloaded.load_state_dict(ckpt["model"])
        result["reload_ok"] = all(torch.equal(a.cpu(), b) for a, b in zip(model.state_dict().values(),
                                                                           reloaded.state_dict().values()))
        log(f"smoke test: checkpoint reload {'ok' if result['reload_ok'] else 'FAILED'}")
    return result

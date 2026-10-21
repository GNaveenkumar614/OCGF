# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# From MTrans (Liu et al., ECCV 2022; github.com/Cliu2/MTrans) via the team fork github.com/GNaveenkumar614/Transmodel,
# commit ebf8a90 (2026-10-06), files datasets/kitti_detection.py and datasets/kitti_loader.py.
# Modified for OCGF:
#   - one record per 2D box from any frame source (KITTI Object now, KITTI-360 later), built from either human labels
#     or detector boxes. Unlabelled objects come from the zero-shot detector, never from label_2, and are kept by a
#     frustum point count (the original filtered them by points inside their ground-truth 3D box);
#   - an explicit labelled-frame list replaces partial_frames (which took the first N frame ids);
#   - several classes: per-class anchors give the unlabelled objects' placeholder size, and an optional
#     class-balanced sampler;
#   - in-image points with pixel coordinates are cached per frame (the original cache was written into the data
#     root); every crop is cut from that cache, with or without box augmentation;
#   - each DataLoader worker gets its own numpy seed (the original workers repeated the same augmentations);
#   - guards for empty crops and all-black image crops; easydict replaced by AttrDict.
"""Object-level data for MTrans: records (one per 2D box), items (image crop, frustum points, masks) and batches."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ...geometry.projection import project
from .ops import AttrDict, build_image_location_map_single, check_points_in_box

CLASSES = ("car", "large_vehicle", "pedestrian", "cyclist")


# ---------------------------------------------------------------- frames

class ProjectedFrames:
    """A frame source plus each frame's in-image LiDAR points with pixel coordinates, cached as .npy files.

    The source needs image(fid) -> uint8 [H, W, 3], points(fid) -> [N, >=4] (x, y, z, intensity, ...) and
    camera(fid) -> CameraCalib, e.g. ocgf.datasets.kitti_object.KittiObjectFrames.
    """

    def __init__(self, source, cache_dir=None):
        self.source = source
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def projected(self, fid: str) -> tuple[np.ndarray, np.ndarray]:
        """(points [M, 4] x y z intensity, uv [M, 2]) of the points that land inside the image."""
        path = self.cache_dir / (fid.replace("/", "__") + ".npy") if self.cache_dir else None
        if path is not None and path.exists():
            data = np.load(path)
        else:
            points = np.asarray(self.source.points(fid), dtype=np.float32)[:, :4]
            uv, _, valid = project(points[:, :3], self.source.camera(fid))
            data = np.concatenate([points[valid], uv[valid]], axis=1).astype(np.float32)
            if path is not None:
                tmp = path.with_suffix(f".{os.getpid()}.tmp.npy")
                np.save(tmp, data)
                os.replace(tmp, path)                    # parallel workers never see a half-written file
        return data[:, :4], data[:, 4:6]

    def image(self, fid: str) -> np.ndarray:
        return self.source.image(fid)

    def warm(self, fids, threads: int = 32, log=print) -> None:
        """Fill the cache for these frames with parallel readers (slow network mounts serve parallel reads well)."""
        import time
        from concurrent.futures import ThreadPoolExecutor

        todo = [f for f in fids if not (self.cache_dir / (f.replace("/", "__") + ".npy")).exists()] if self.cache_dir else []
        t0 = time.time()
        with ThreadPoolExecutor(threads) as pool:
            for k, _ in enumerate(pool.map(self.projected, todo), start=1):
                if log and (k % 1000 == 0 or k == len(todo)):
                    log(f"  point cache: {k}/{len(todo)} frames, {k / max(time.time() - t0, 1e-6):.1f} frames/s")


# ---------------------------------------------------------------- objects and records

def label_objects(source, fids, max_truncation: float = 0.95) -> dict[str, list]:
    """{fid: objects} from human labels (source.labels(fid), e.g. KITTI label_2). Only for labelled or evaluation frames."""
    out = {}
    for fid in fids:
        out[fid] = [dict(cls=o["class"], box_2d=np.asarray(o["box_2d"], float), box=np.asarray(o["box"], float),
                         score=1.0, has_label=True, truncated=float(o["truncated"]), occluded=int(o["occluded"]))
                    for o in source.labels(fid) if o["truncated"] <= max_truncation]
    return out


def detection_objects(detections: dict, fids, min_score: float = 0.0, classes=CLASSES) -> dict[str, list]:
    """{fid: objects} from 2D detections ({fid: Detections} in our classes). These objects carry no 3D label."""
    out = {}
    for fid in fids:
        d = detections.get(fid)
        out[fid] = [] if d is None else [
            dict(cls=label, box_2d=np.asarray(box, float), box=None, score=float(score), has_label=False,
                 truncated=-1.0, occluded=-1)
            for label, score, box in zip(d.labels, d.scores, d.boxes) if label in classes and score >= min_score]
    return out


def _overlapping(boxes: np.ndarray) -> list[list[int]]:
    """For each 2D box, the other boxes it overlaps (strictly, in both directions), as in the original."""
    out = [[] for _ in range(len(boxes))]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            b1, b2 = boxes[i], boxes[j]
            if max(b1[1], b2[1]) < min(b1[3], b2[3]) and max(b1[0], b2[0]) < min(b1[2], b2[2]):
                out[i].append(j)
                out[j].append(i)
    return out


def _in_box2d(uv, box, inclusive: bool) -> np.ndarray:
    l, t, r, b = box
    if inclusive:
        return (uv[:, 0] >= l) & (uv[:, 1] >= t) & (uv[:, 0] <= r) & (uv[:, 1] <= b)
    return (uv[:, 0] > l) & (uv[:, 1] > t) & (uv[:, 0] < r) & (uv[:, 1] < b)


def build_records(frames: ProjectedFrames, objects: dict, classes=CLASSES, min_points: int = 5,
                  min_foreground: int = 5, log=print, keep_all: bool = False) -> list[dict]:
    """One JSON-friendly record per object with at least min_points frustum points and, if labelled, at least
    min_foreground points inside its 3D box. keep_all keeps every object (inference: the caller decides)."""
    records, t_frames = [], len(objects)
    for k, (fid, objs) in enumerate(objects.items()):
        if not objs:
            continue
        points, uv = frames.projected(fid)
        boxes2d = np.array([o["box_2d"] for o in objs]).reshape(-1, 4)
        overlaps = _overlapping(boxes2d)
        for i, o in enumerate(objs):
            inside = _in_box2d(uv, o["box_2d"], inclusive=False)
            n_points = int(inside.sum())
            n_fg = None
            if o["has_label"]:
                b = o["box"]
                n_fg = int(check_points_in_box(points[inside, :3], location=b[:3], dimension=b[3:6], yaw=b[6]).sum())
            if keep_all or (n_points >= min_points and (n_fg is None or n_fg >= min_foreground)):
                records.append(dict(fid=fid, cls=o["cls"], class_idx=classes.index(o["cls"]),
                                    box_2d=[float(v) for v in o["box_2d"]],
                                    box=None if o["box"] is None else [float(v) for v in o["box"]],
                                    score=o["score"], has_label=o["has_label"], truncated=o["truncated"],
                                    occluded=o["occluded"], n_points=n_points, n_foreground=n_fg,
                                    overlap_boxes=[[float(v) for v in boxes2d[j]] for j in overlaps[i]]))
        if log and ((k + 1) % 500 == 0 or k + 1 == t_frames):
            log(f"  records: {k + 1}/{t_frames} frames, {len(records)} objects")
    return records


def class_anchors(records, classes=CLASSES) -> list[list[float]]:
    """Mean (l, w, h) per class over the labelled records: the size anchors of MTrans's box head."""
    anchors = []
    for c in classes:
        dims = np.array([r["box"][3:6] for r in records if r["has_label"] and r["cls"] == c])
        if not len(dims):
            raise ValueError(f"no labelled object of class {c!r}: its anchor cannot be computed")
        anchors.append([round(float(v), 4) for v in dims.mean(axis=0)])
    return anchors


# ---------------------------------------------------------------- items

class ObjectDataset(Dataset):
    """MTrans inputs for each record: the 2D box crop of the image (square, out_img_size), an overlap mask of the
    other boxes, and out_cloud_size frustum point slots (real points, then 2D-only query pixels)."""

    def __init__(self, records, frames: ProjectedFrames, anchors, out_img_size: int = 112, out_cloud_size: int = 512,
                 mask_ratio=(0.0, 0.0), box2d_augmentation: bool = False):
        self.records, self.frames = records, frames
        self.anchors = np.asarray(anchors, dtype=np.float64)
        self.out_img_size, self.out_cloud_size = int(out_img_size), int(out_cloud_size)
        self.mask_ratio, self.box2d_augmentation = tuple(mask_ratio), bool(box2d_augmentation)
        self.img_coords = build_image_location_map_single(self.out_img_size, self.out_img_size, "cpu")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        fid, has_label = rec["fid"], bool(rec["has_label"])
        S, cloud_size = self.out_img_size, self.out_cloud_size
        full_img = torch.from_numpy(np.array(self.frames.image(fid), dtype=np.uint8)).permute(2, 0, 1).float() / 255
        H, W = full_img.shape[1:]
        points, uv = self.frames.projected(fid)
        if has_label:
            box3d = np.asarray(rec["box"], dtype=np.float64)
            location, dimensions, yaw = box3d[:3], box3d[3:6], float(box3d[6])
        else:
            location, dimensions, yaw = np.zeros(3), self.anchors[rec["class_idx"]].copy(), 0.0

        # build overlap mask
        overlap_mask = torch.ones_like(full_img[0:1])
        for ol, ot, orr, ob in rec["overlap_boxes"]:
            overlap_mask[:, int(np.floor(ot)):int(np.ceil(ob)), int(np.floor(ol)):int(np.ceil(orr))] = 0
        full_img = torch.cat([full_img, overlap_mask], dim=0)

        l, t, r, b = rec["box_2d"]
        inside = _in_box2d(uv, (l, t, r, b), inclusive=False)
        # box2d augmentation, random scale + shift
        if self.box2d_augmentation:
            random_scale = np.random.rand(2) * 0.2 + 0.95        # [95% ~ 115%]
            random_shift = np.random.rand(2) * 0.1 - 0.05        # [-5% ~ 5%]
            tw, th, tx, ty = r - l, b - t, (r + l) / 2, (t + b) / 2
            tx, ty = tx + tw * random_shift[0], ty + th * random_shift[1]   # random shift
            tw, th = tw * random_scale[0], th * random_scale[1]           # random scale
            al, at, ar, ab = max(0, tx - tw / 2), max(0, ty - th / 2), min(tx + tw / 2, W - 1), min(ty + th / 2, H - 1)
            aug_inside = _in_box2d(uv, (al, at, ar, ab), inclusive=True)
            if aug_inside.any():                                  # keep the original box if the shift lost every point
                l, t, r, b, inside = al, at, ar, ab, aug_inside
        sub_cloud, sub_cloud2d = points[inside].astype(np.float64), uv[inside].astype(np.float64)
        if has_label:
            foreground_label = check_points_in_box(sub_cloud[:, :3], location=location, dimension=dimensions, yaw=yaw)
        else:
            foreground_label = np.full(len(sub_cloud), 2)       # unknown: no 3D label

        l, t = int(np.floor(max(l, 0))), int(np.floor(max(t, 0)))
        r, b = max(int(np.ceil(min(r, W))), l + 1), max(int(np.ceil(min(b, H))), t + 1)
        img = full_img[:, t:b, l:r].unsqueeze(0)  # (1, 4, box_h, box_w)

        # crop original image by the obj's 2D box
        box_size = max(b - t, r - l)
        img = torch.nn.functional.interpolate(img, scale_factor=S / box_size, mode='bilinear', align_corners=True,
                                              recompute_scale_factor=False)[:, :, :S, :S]
        h, w = img.shape[-2:]
        num_padding = (int(np.floor((S - w) / 2)), int(np.ceil((S - w) / 2)), int(np.floor((S - h) / 2)), int(np.ceil((S - h) / 2)))
        img = torch.nn.functional.pad(img, num_padding)     # zero-padding to make it square
        crop_sub_cloud2d = (sub_cloud2d - np.array([l, t])) * (S / box_size) + np.array([num_padding[0], num_padding[2]])
        crop_sub_cloud2d = np.clip(crop_sub_cloud2d, 0, S - 1)
        img, overlap = img[:, 0:3], img[:, 3:4]

        # sampling the point cloud to fixed size
        out_sub_cloud = np.ones((cloud_size, 4)) * (-9999)       # -9999 for paddings
        out_sub_cloud2d = np.ones((cloud_size, 2)) * (-9999)
        out_ori_cloud2d = np.ones((cloud_size, 2)) * (-9999)
        out_real_point_mask = np.zeros(cloud_size)    # 0 for padding, 1 for real points, 2 for masked, 3 for jittered
        out_foreground_label = np.ones(cloud_size) * 2   # 0 for background, 1 for foreground, 2 for unknown

        if len(sub_cloud) > cloud_size:
            sample_idx = np.random.choice(np.arange(len(sub_cloud)), cloud_size, replace=False)            # random sampling
            out_sub_cloud[...] = sub_cloud[sample_idx]
            out_sub_cloud2d[...] = crop_sub_cloud2d[sample_idx]
            out_ori_cloud2d[...] = sub_cloud2d[sample_idx]
            out_real_point_mask[...] = 1
            out_foreground_label[...] = foreground_label[sample_idx]
        else:
            pc_size = len(sub_cloud)
            out_sub_cloud[:pc_size] = sub_cloud
            out_sub_cloud2d[:pc_size] = crop_sub_cloud2d
            out_ori_cloud2d[:pc_size] = sub_cloud2d
            out_real_point_mask[:pc_size] = 1
            out_foreground_label[:pc_size] = foreground_label

            # sample 2D points, leave blank for 3D coords
            p = ((img[0] != 0).all(dim=0) * 1).numpy().astype(np.float64)    # only sample pixels from not-padding-area
            if p.sum() == 0:
                p[:] = 1                                       # an all-black crop: sample anywhere
            p = p / p.sum()
            resample = (p > 0).sum() < (cloud_size - pc_size)
            sample_idx = np.random.choice(np.arange(S * S), cloud_size - pc_size, replace=resample, p=p.reshape(-1))
            sampled_c2d = self.img_coords.view(-1, 2)[sample_idx, :].numpy()
            out_sub_cloud2d[pc_size:, :] = sampled_c2d
            out_ori_cloud2d[pc_size:, :] = (sampled_c2d - np.array([num_padding[0], num_padding[2]])) / (S / box_size) + np.array([l, t])

        # random mask/jitter points
        num_real_points = int((out_real_point_mask == 1).sum())
        mask_ratio = np.random.rand() * (self.mask_ratio[1] - self.mask_ratio[0]) + self.mask_ratio[0]
        num_mask = min(int(mask_ratio * num_real_points), max(0, num_real_points - 5))        # leave at least 5 points
        out_real_point_mask[np.random.choice(np.arange(num_real_points), num_mask, replace=False)] = 2   # 2 for masked

        return dict(use_3d_label=has_label, frame_img=img, overlap_mask=overlap, sub_cloud=out_sub_cloud,
                    sub_cloud2d=out_sub_cloud2d, ori_cloud2d=out_ori_cloud2d, real_point_mask=out_real_point_mask,
                    foreground_label=out_foreground_label, class_idx=int(rec["class_idx"]),
                    box_2d=np.asarray(rec["box_2d"], dtype=np.float64), location=np.asarray(location, np.float64),
                    dimensions=np.asarray(dimensions, np.float64), yaw=float(yaw), fid=fid, score=float(rec["score"]),
                    record_index=int(index))


# ---------------------------------------------------------------- batches

def collate(batch: list) -> AttrDict:
    """Merge items into one batch (as kitti_collate_wrapper, without calibrations)."""
    stack = lambda key: np.stack([o[key] for o in batch], axis=0)
    return AttrDict(
        use_3d_label=torch.tensor([o["use_3d_label"] for o in batch], dtype=torch.bool),          # (B, )
        images=torch.cat([o["frame_img"] for o in batch], dim=0),                                 # (B, 3, S, S)
        overlap_masks=torch.cat([o["overlap_mask"] for o in batch], dim=0),                       # (B, 1, S, S)
        sub_clouds=torch.from_numpy(stack("sub_cloud")).float(),                                  # (B, N, 4)
        sub_clouds2d=torch.from_numpy(stack("sub_cloud2d")).float(),                              # (B, N, 2)
        ori_clouds2d=torch.from_numpy(stack("ori_cloud2d")).float(),                              # (B, N, 2)
        real_point_mask=torch.from_numpy(stack("real_point_mask")).long(),                        # (B, N)
        foreground_label=torch.from_numpy(stack("foreground_label")).long(),                      # (B, N)
        classes=torch.tensor([o["class_idx"] for o in batch], dtype=torch.long),                  # (B, )
        boxes_2d=torch.from_numpy(stack("box_2d")).float(),                                       # (B, 4)
        locations=torch.from_numpy(stack("location")).float(),                                   # (B, 3)
        dimensions=torch.from_numpy(stack("dimensions")).float(),                                 # (B, 3)
        yaws=torch.tensor([o["yaw"] for o in batch], dtype=torch.float32).unsqueeze(-1),          # (B, 1)
        scores=torch.tensor([o["score"] for o in batch], dtype=torch.float32),                    # (B, )
        fids=[o["fid"] for o in batch], record_index=[o["record_index"] for o in batch])


def merge_batches(a: AttrDict, b: AttrDict) -> AttrDict:
    out = AttrDict()
    for k in a:
        out[k] = torch.cat([a[k], b[k]], dim=0) if isinstance(a[k], torch.Tensor) else a[k] + b[k]
    return out


def to_device(data: AttrDict, device) -> AttrDict:
    return AttrDict({k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in data.items()})


def _seed_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)        # torch gives every worker its own initial seed


def class_balanced_sampler(records, power: float = 0.5, seed: int = 0) -> WeightedRandomSampler:
    """Draws each object with weight n_class^-power (0.5: square-root balancing), same epoch length."""
    counts = {}
    for r in records:
        counts[r["cls"]] = counts.get(r["cls"], 0) + 1
    weights = torch.tensor([counts[r["cls"]] ** -power for r in records], dtype=torch.double)
    g = torch.Generator()
    g.manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(records), replacement=True, generator=g)


def make_loader(dataset, batch_size: int, shuffle: bool = False, sampler=None, num_workers: int = 0, seed: int = 0,
                drop_last: bool = False) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle and sampler is None, sampler=sampler,
                      num_workers=num_workers, collate_fn=collate, pin_memory=torch.cuda.is_available(),
                      drop_last=drop_last, generator=g, worker_init_fn=_seed_worker,
                      persistent_workers=num_workers > 0)

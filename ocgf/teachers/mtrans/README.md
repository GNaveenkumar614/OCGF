# MTrans teacher (vendored)

Source: MTrans, *Multimodal Transformer for Automatic 3D Annotation and Object Detection* (Liu et al., ECCV 2022,
[github.com/Cliu2/MTrans](https://github.com/Cliu2/MTrans)), taken from the team fork
[GNaveenkumar614/Transmodel](https://github.com/GNaveenkumar614/Transmodel) at commit `ebf8a90` (2026-10-06).

## Licences

- MTrans is under the **Mozilla Public License 2.0** (`LICENSE` in this folder). The licence is per file: every file
  here that contains MTrans code carries the MPL-2.0 notice, and our changes to those files are MPL-2.0 as well.
- The rotated-IoU loss in `iou/` comes from [Rotated_IoU](https://github.com/lilanxiao/Rotated_IoU)
  (MIT, Copyright (c) 2020 Lanxiao Li, `iou/LICENSE-Rotated_IoU`), which MTrans distributes as `loss/`.
- `infer.py` is new OCGF code under the repository licence.
- Weights trained on KITTI are for non-commercial use only (KITTI is CC BY-NC-SA 3.0).

## Files

| File | From | Change |
|---|---|---|
| `model.py` | `models/MTrans.py` | per-class anchors, class embedding in the box/confidence heads, `predict()` with raw confidence (no batch-max rescaling), loss tolerant of unlabelled-only batches |
| `data.py` | `datasets/kitti_detection.py`, `kitti_loader.py` | records from any frame source; unlabelled objects from detector boxes (never `label_2`); explicit labelled-frame list; class-balanced sampler; per-frame point cache; per-worker seeds |
| `train.py` | `train.py`, the fork's `train_kaggle.py` | object pools, per-epoch checkpoint and resume, time budget, smoke mode, per-class evaluation; **no best-checkpoint selection on KITTI val** |
| `attention.py` | `models/modules/point_sa.py` | unused import removed |
| `fpn.py` | `models/CNN_backbone/FPN.py` | none |
| `ops.py` | `modality_mapper.py`, `utils/point_ops.py` | only the three helpers used; `AttrDict` replaces easydict |
| `schedule.py` | `utils/lr_scheduler.py` | only `WarmupCosineAnnealing` |
| `iou/` | `loss/utils/*` (Rotated_IoU) | pure-PyTorch vertex sorting instead of the CUDA extension; `np.int` fix |
| `infer.py` | new | detector boxes -> neutral object files |

## Label budget

Human 3D labels enter training only through the frozen 500-frame list
(`configs/splits/kitti_object/mtrans_labelled_500.txt`, 125-frame ablation nested in it), chosen from detections
without looking at labels. Unlabelled objects (the other KITTI train frames) use detector boxes and feed only the
point-reconstruction loss. KITTI val labels are used to evaluate, never to select a checkpoint.

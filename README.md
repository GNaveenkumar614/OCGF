# OCGF

Research code for confidence-guided, object-aware drivable-area estimation from LiDAR–camera fusion, in bird's-eye view (BEV). Work in progress.

From one front-camera image and one LiDAR scan, the model predicts free road, object occupancy and unknown cells on a 50 m × 40 m BEV grid at 0.25 m. The object branch has two parts: a zero-shot 2D detector and an MTrans-based 3D box teacher. The scene branch is inspired by FusionViT; it keeps camera and LiDAR BEV features separate until the OCGF gate fuses them per cell. Pseudo-labels support training only.

## Layout

| Path | Contents |
|---|---|
| `ocgf/datasets/` | dataset maps, KITTI-360 and KITTI Object frames, 3D boxes, frame index, splits (numpy only) |
| `ocgf/geometry/` | calibration, transforms, projection, BEV grid, ground surface, viewshed (numpy only) |
| `ocgf/teachers/` | offline teachers: 2D detector, MTrans, road teacher |
| `ocgf/models/` | student model (torch) |
| `ocgf/training/` | targets, losses, trainer |
| `ocgf/evaluation/` | ground-truth builder, metrics, calibration, perturbations |
| `ocgf/utils/` | config, seeding, provenance |
| `tests/` | unit tests on small synthetic data |

The student code never imports `ocgf.teachers`; teachers hand their outputs over as files.

## Setup (CPU, for development and tests)

```
python -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
```

Training and teacher inference run on Kaggle GPUs.

## Data

This repository contains no dataset files and never redistributes them. KITTI and KITTI-360 are licensed CC BY-NC-SA 3.0; get them from the official sources after registering. Model weights trained on them are for non-commercial use only.

## License

The repository license has not been chosen yet. Code under `ocgf/teachers/mtrans/` will remain under MPL-2.0, its original license.

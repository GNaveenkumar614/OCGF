"""Experiment configs: a YAML file merged over its `base:` chain, resolved into the typed configs every part uses.

The `objects:` switches decide how the MTrans objects enter an experiment:
  inputs     the object channels O_B go into the model
  occluders  box tops occlude in the student's viewshed (M_C, shadows); ablation A3 turns this off
  targets    strict boxes are occupancy pseudo-labels, and no road negative lies inside a detected box
E0 baselines use none; the E1 post-hoc arm uses targets only (boxes are subtracted after prediction); the E1/E2 object
arm uses all three. `objects.source` picks the object file: "detector" (MTrans on the detector's boxes, the real
pipeline), or E3's oracle arms "oracle2d" (MTrans on ground-truth 2D boxes) and "oracle3d" (ground-truth 3D boxes).
Training and evaluation resolve a run's config the same way, so they never disagree.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..datasets.bev_inputs import BEVInputConfig
from ..datasets.locate import OBJECT_SOURCES
from ..datasets.objects import ObjectConfig
from ..models.student import StudentConfig
from .losses import LossWeights
from .targets import TargetConfig
from .train import TrainConfig

OBJECT_SWITCHES = ("inputs", "occluders", "targets")


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _merged(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    base = raw.pop("base", None)
    return _merge(_merged(path.parent / base), raw) if base else raw


def load_config(path) -> dict:
    """An experiment config merged over the config it names under `base:` (recursively), with its targets file
    (relative to this config's folder) embedded under `targets_config`."""
    path = Path(path)
    cfg = _merged(path)
    if "targets" in cfg:
        cfg["targets_config"] = yaml.safe_load((path.parent / cfg["targets"]).read_text())
    return cfg


@dataclass(frozen=True)
class Resolved:
    model: StudentConfig
    train: TrainConfig
    loss: LossWeights
    inputs: BEVInputConfig
    targets: TargetConfig
    objects: ObjectConfig
    switches: dict
    object_source: str = "detector"

    @property
    def uses_objects(self) -> bool:
        return any(self.switches.values())


def resolve(cfg: dict) -> Resolved:
    """The typed configs of a (merged) experiment config; also accepts the config stored in a checkpoint."""
    switches = {k: bool((cfg.get("objects") or {}).get(k, False)) for k in OBJECT_SWITCHES}
    source = (cfg.get("objects") or {}).get("source", "detector")
    if source not in OBJECT_SOURCES:
        raise ValueError(f"objects.source {source!r} is not one of {OBJECT_SOURCES}")
    tc = cfg.get("targets_config") or {}
    inputs = dataclasses.replace(BEVInputConfig.from_dict(tc.get("bev_inputs")), box_occluders=switches["occluders"])
    targets = dataclasses.replace(TargetConfig.from_dict(tc.get("targets")), box_targets=switches["targets"])
    model = dataclasses.replace(StudentConfig.from_dict(cfg.get("model")), objects=switches["inputs"])
    return Resolved(model=model, train=TrainConfig.from_dict(cfg.get("train")), loss=LossWeights.from_dict(cfg.get("loss")),
                    inputs=inputs, targets=targets, objects=ObjectConfig.from_dict(tc.get("objects")),
                    switches=switches, object_source=source)

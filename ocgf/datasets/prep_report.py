"""Quality report of the data-prep run: figures (if matplotlib is available) and report.md."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .frame_index import compress_ranges


def _figure_class():
    """matplotlib's Figure, used without pyplot: no backend switch, nothing global changes in a notebook."""
    try:
        from matplotlib.figure import Figure
        return Figure
    except ImportError:
        return None


def ring_checks(profiles: dict) -> dict:
    """From sampled ring profiles: share of scans whose median elevation rises with the ring index."""
    elevation = profiles.get("elevation", np.zeros((0, 64)))
    rising = []
    for e in elevation:
        e = e[~np.isnan(e)]
        rising.append(len(e) > 1 and bool(np.all(np.diff(e) > 0)))
    return {"sampled_scans": len(elevation), "rising_share": float(np.mean(rising)) if rising else float("nan")}


def make_figures(out_dir, index: dict, scan_stats: dict, profiles: dict, overlays: list) -> list[str]:
    """Write the figures and return their file names. overlays: [(frame, image [H, W, 3], uv [N, 2], depth [N])]."""
    Figure = _figure_class()
    if Figure is None:
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []

    def subplots(figsize):
        fig = Figure(figsize=figsize)
        return fig, fig.subplots()

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=110)
        names.append(name)

    dt = index["dt_ms"][np.isfinite(index["dt_ms"])]
    fig, ax = subplots(figsize=(6, 3.5))
    ax.hist(dt, bins=80, color="#3b6fb6")
    ax.set_xlabel("scan time - image time (ms)")
    ax.set_ylabel("frames")
    ax.set_title("Image / scan alignment")
    save(fig, "dt_hist.png")

    fig, ax = subplots(figsize=(7, 3.5))
    ax.plot(scan_stats["frame"], scan_stats["n_raw"], lw=0.6, label="raw")
    ax.plot(scan_stats["frame"], scan_stats["n_crop"], lw=0.6, label="after crop")
    ax.set_xlabel("frame")
    ax.set_ylabel("points")
    ax.legend()
    ax.set_title("Points per scan")
    save(fig, "points_per_frame.png")

    if len(profiles.get("elevation", [])):
        fig, ax = subplots(figsize=(6, 3.5))
        for e in profiles["elevation"]:
            ax.plot(np.arange(len(e)), e, lw=0.5, alpha=0.6, color="#3b6fb6")
        ax.set_xlabel("ring index (0 = last ring stored)")
        ax.set_ylabel("median elevation (deg)")
        ax.set_title(f"Ring elevation, {len(profiles['elevation'])} sampled scans")
        save(fig, "ring_profile.png")

    fig, ax = subplots(figsize=(6, 6))
    posed = np.isfinite(index["x_world"])
    ax.plot(index["x_world"][posed], index["y_world"][posed], lw=0.8, color="#9aa5b5", label="posed frames")
    u = index["usable"]
    ax.scatter(index["x_world"][u], index["y_world"][u], s=1, color="#3b6fb6", label="usable")
    s = index["sample_5m"]
    ax.scatter(index["x_world"][s], index["y_world"][s], s=6, color="#c0392b", label="sampled every 5 m")
    ax.set_aspect("equal")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.legend(markerscale=4)
    ax.set_title("Trajectory")
    save(fig, "trajectory.png")

    for frame, image, uv, depth in overlays:
        h, w = image.shape[:2]
        fig, ax = subplots(figsize=(w / 110, h / 110))
        ax.imshow(image)
        ax.scatter(uv[:, 0], uv[:, 1], c=np.clip(depth, 0, 50), s=0.4, cmap="turbo", vmin=0, vmax=50)
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.axis("off")
        ax.set_title(f"frame {frame}: LiDAR points coloured by depth (0-50 m)", fontsize=8)
        save(fig, f"overlay_{frame:010d}.png")
    return names


def write_report(path, sequence: str, index: dict, scan_stats: dict, profiles: dict, figures: list[str],
                 settings: dict, storage_bytes: int) -> str:
    """Write report.md and return its text."""
    n = len(index["frame"])
    span = (index["has_image"] | index["has_scan"] | index["has_pose"])
    lines = [f"# Data-prep report: {sequence}", ""]

    lines += ["## Frames", "", "| | frames |", "|---|---|"]
    for key, label in [("has_image", "image"), ("has_scan", "scan"), ("has_pose", "pose"), ("has_sem2d", "2D semantics"),
                       ("aligned", f"aligned (image/scan gap at most {settings['max_dt_ms']} ms)"), ("usable", "usable"),
                       ("kept", f"kept (stored; every {settings.get('keep_every_m')} m)" if settings.get("keep_every_m")
                        else "kept (stored; every frame)"),
                       ("sample_5m", f"sampled every {settings['step_m']} m")]:
        if key in index:
            lines.append(f"| {label} | {int(index[key].sum())} |")
    in_window = index["window_start"] >= 0
    lines.append(f"| inside a semantic window | {int(in_window.sum())} |")
    lines += ["", f"Frames 0-{n - 1}. Missing, among frames that have at least one source:", ""]
    for key, label in [("has_image", "image"), ("has_scan", "scan"), ("has_pose", "pose")]:
        lines.append(f"- no {label}: {compress_ranges(index['frame'][span & ~index[key]])}")
    lines.append(f"- outside every semantic window: {compress_ranges(index['frame'][span & ~in_window])}")

    dt = index["dt_ms"][np.isfinite(index["dt_ms"])]
    lines += ["", "## Image / scan alignment", ""]
    if len(dt):
        lines.append(f"scan time - image time over {len(dt)} frames: median {np.median(dt):.2f} ms, "
                     f"5th-95th percentile {np.percentile(dt, 5):.2f} to {np.percentile(dt, 95):.2f} ms, "
                     f"largest |dt| {np.abs(dt).max():.2f} ms.")
    else:
        lines.append("No frame has both timestamps.")

    checks = ring_checks(profiles)
    lines += ["", "## Scans", "",
              f"- scans read: {len(scan_stats['frame'])}",
              f"- points per scan, raw: median {np.median(scan_stats['n_raw']):.0f} "
              f"(min {scan_stats['n_raw'].min()}, max {scan_stats['n_raw'].max()})" if len(scan_stats["frame"]) else "- no scans",
              f"- points per scan after the crop: median {np.median(scan_stats['n_crop']):.0f}" if len(scan_stats["frame"]) else "",
              f"- rings found per scan: min {scan_stats['n_rings'].min()}, median {np.median(scan_stats['n_rings']):.0f}, "
              f"max {scan_stats['n_rings'].max()} (expected 64)" if len(scan_stats["frame"]) else "",
              f"- ring elevation rises with the ring index in {checks['rising_share']:.0%} of {checks['sampled_scans']} sampled scans "
              "(ring 0 should be the lowest beam)",
              f"- crop: x in [{settings['crop']['x_min']}, {settings['crop']['x_max']}) m, "
              f"y in [{settings['crop']['y_min']}, {settings['crop']['y_max']}) m",
              f"- stored: {storage_bytes / 1e9:.2f} GB"
              + (f" ({storage_bytes / len(scan_stats['frame']) / 1e6:.2f} MB per scan)" if len(scan_stats["frame"]) else "")]

    odo = index["odometer_m"][np.isfinite(index["odometer_m"])]
    lines += ["", "## Travel", "",
              f"- distance driven (posed frames): {odo.max() if len(odo) else 0:.0f} m",
              f"- a usable frame needs at least {settings['min_future_m']} m of future path"]

    if figures:
        lines += ["", "## Figures", ""] + [f"![{f}]({f})" for f in figures]
    else:
        lines += ["", "Figures were skipped: matplotlib is not installed."]

    text = "\n".join(line for line in lines if line is not None) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return text

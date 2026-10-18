"""Per-drive statistics for choosing the KITTI-360 paper drives (train / val / test split by drive).

Reads vehicle poses and OXTS for all drives (data_poses.zip, data_poses_oxts.zip), the 3D-box XMLs, and the download
sizes of the scan and image zips (HEAD requests). Prints a table and a drive-to-drive overlap matrix: the share of each
drive's route that passes within --near-m of another drive's route (from GPS, so all drives share one frame).

    python scripts/drive_stats.py --boxes-dir C:/Users/nanin/Downloads/data_3d_bboxes/data_3d_bboxes/train

Used on 2026-10-18 to choose the paper split (configs/splits/kitti360_paper_v1.json):
  - the drives fall into five separate areas: 0000 | 0002+0003 | 0004+0005+0006 | 0007 | 0009+0010; overlap only
    within an area (0005 runs 23 % of its route within 50 m of 0004 and of 0006);
  - 0009 + 0010 have the most moving pedestrians (85) and cyclists (54) and 1218 parked cars, so they are the test set;
  - 0003 and 0007 are fast roads (55-63 km/h median) with few near-obstacle cases, so they are unused.
"""
import argparse
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

from ocgf.datasets.boxes3d import _CLASS_OF, TWO_WHEELERS, parse_bboxes
from ocgf.datasets.frame_index import read_poses, sample_every

REPO = Path(__file__).resolve().parents[1]
DRIVES = ("0000", "0002", "0003", "0004", "0005", "0006", "0007", "0009", "0010")
EARTH_RADIUS_M = 6378137.0


def seq_name(drive: str) -> str:
    return f"2013_05_28_drive_{drive}_sync"


def read_oxts(z: zipfile.ZipFile, seq: str):
    """lat, lon (deg) and horizontal speed (m/s) per frame, in frame order."""
    prefix = f"data_poses/{seq}/oxts/data/"
    names = sorted(n for n in z.namelist() if n.startswith(prefix) and n.endswith(".txt"))
    values = np.array([z.read(n).split()[:11] for n in names], dtype=np.float64)
    return values[:, 0], values[:, 1], np.hypot(values[:, 6], values[:, 7])


def local_xy(lat, lon, lat0, lon0) -> np.ndarray:
    """Equirectangular metres around (lat0, lon0); plenty accurate over a city."""
    return np.column_stack([np.radians(lon - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0)),
                            np.radians(lat - lat0) * EARTH_RADIUS_M])


def object_stats(xml: Path) -> dict:
    static, moving = Counter(), defaultdict(set)
    for o in parse_bboxes(xml):
        cls = "two_wheeler" if o.name in TWO_WHEELERS else _CLASS_OF.get(o.name, "other")
        if o.timestamp < 0:
            static[cls] += 1
        else:
            moving[cls].add(o.instance_id)
    return {"parked_cars": static["car"], "static_large": static["large_vehicle"], "static_ped": static["pedestrian"],
            "static_two_wheelers": static["two_wheeler"], "static_other": static["other"],
            **{f"moving_{c}": len(moving[c]) for c in ("car", "large_vehicle", "pedestrian", "cyclist")}}


def zip_size_gb(url: str) -> float:
    import urllib.request
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
        return round(int(r.headers["Content-Length"]) / 1e9, 1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meta", type=Path, default=REPO / "data" / "kitti360_meta")
    p.add_argument("--boxes-dir", type=Path, required=True, help="folder with data_3d_bboxes/train/*.xml")
    p.add_argument("--sources", type=Path, default=REPO / "configs" / "data_sources" / "kitti360.yaml")
    p.add_argument("--no-sizes", action="store_true", help="skip the HEAD requests for download sizes")
    p.add_argument("--near-m", type=float, default=50.0)
    p.add_argument("--keep-every-m", type=float, default=2.5)
    p.add_argument("--figure", type=Path, default=REPO / "data" / "qa" / "drives_map.png")
    args = p.parse_args(argv)

    sources = yaml.safe_load(args.sources.read_text())
    poses_zip = zipfile.ZipFile(args.meta / "data_poses.zip")
    oxts_zip = zipfile.ZipFile(args.meta / "data_poses_oxts.zip")
    stats, routes, gps = {}, {}, {}
    for d in DRIVES:
        seq = seq_name(d)
        lat, lon, speed = read_oxts(oxts_zip, seq)
        gps[d] = (lat, lon)
        frames, T = read_poses(io.StringIO(poses_zip.read(f"{seq}/poses.txt").decode()))
        steps = np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1)
        km = float(steps.sum()) / 1000
        moving = speed > 0.5
        s = {"frames": len(lat), "posed_frames": len(frames), "km": round(km, 2),
             "minutes": round(len(lat) / 10 / 60, 1), "stopped_share": round(float(1 - moving.mean()), 2),
             "median_moving_kmh": round(float(np.median(speed[moving]) * 3.6), 1) if moving.any() else 0.0,
             "fast_share": round(float((speed * 3.6 > 50).mean()), 2),
             "frames_kept_at_2p5m": int(km * 1000 / args.keep_every_m)}
        s.update(object_stats(args.boxes_dir / f"{seq}.xml"))
        for c in ("car", "pedestrian", "cyclist", "large_vehicle"):
            s[f"moving_{c}_per_km"] = round(s[f"moving_{c}"] / max(km, 1e-6), 1)
        if not args.no_sizes:
            s["velodyne_gb"] = zip_size_gb(sources["velodyne"][seq])
            s["images_00_gb"] = zip_size_gb(sources["images_00"][seq])
        stats[d] = s
        print(f"drive {d}: {s['frames']} frames, {s['km']} km", flush=True)

    lat0 = np.mean(np.concatenate([g[0] for g in gps.values()]))
    lon0 = np.mean(np.concatenate([g[1] for g in gps.values()]))
    for d, (lat, lon) in gps.items():
        xy = local_xy(lat, lon, lat0, lon0)
        dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
        routes[d] = xy[sample_every(dist, np.ones(len(dist), bool), 5.0)]     # one point every 5 m
    overlap = {a: {b: round(float((cKDTree(routes[b]).query(routes[a])[0] < args.near_m).mean()), 2)
                   for b in DRIVES if b != a} for a in DRIVES}

    cols = [("frames", "frames"), ("km", "km"), ("posed_frames", "posed"), ("stopped_share", "stopped"),
            ("median_moving_kmh", "km/h"), ("fast_share", ">50km/h"), ("parked_cars", "parked"),
            ("moving_car", "mov car"), ("moving_large_vehicle", "mov large"), ("moving_pedestrian", "mov ped"),
            ("moving_cyclist", "mov cyc"), ("static_two_wheelers", "parked 2w"), ("frames_kept_at_2p5m", "kept@2.5m")]
    if not args.no_sizes:
        cols += [("velodyne_gb", "scans GB"), ("images_00_gb", "imgs GB")]
    print("\n| drive | " + " | ".join(c[1] for c in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for d in DRIVES:
        print(f"| {d} | " + " | ".join(str(stats[d][c[0]]) for c in cols) + " |")
    print(f"\nshare of the row drive's route within {args.near_m:.0f} m of the column drive's route:")
    print("      " + " ".join(f"{b:>5}" for b in DRIVES))
    for a in DRIVES:
        print(f"{a}  " + " ".join("    -" if a == b else f"{overlap[a][b]:5.2f}" for b in DRIVES))

    out = args.meta / "drive_stats.json"
    out.write_text(json.dumps({"stats": stats, "overlap_within_m": args.near_m, "overlap": overlap}, indent=1))
    print(f"\nwrote {out}")

    from matplotlib.figure import Figure
    fig = Figure(figsize=(9, 9))
    ax = fig.subplots()
    for d, xy in routes.items():
        ax.plot(xy[:, 0], xy[:, 1], lw=2, label=f"{d} ({stats[d]['km']} km)")
        ax.annotate(d, xy[len(xy) // 2], fontsize=9, weight="bold")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_title("KITTI-360 training drives (GPS)")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.figure, dpi=90)
    print(f"wrote {args.figure}")


if __name__ == "__main__":
    main()

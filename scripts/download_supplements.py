"""Back up prepared KITTI-360 drives from Kaggle to this machine, and check each copy.

Every data-prep run (notebook 01) writes /kaggle/working/kitti360_supplement, saved as the private dataset
'<owner>/kitti360-d<drive>-supplement'. This script downloads those datasets with the Kaggle API into
data/kitti360_supplements/<drive>/ (git-ignored: the data is CC BY-NC-SA and never goes into the repo) and checks each
copy against its manifest: kept frames = scan shards = image shards, frame index and report present.

Setup, once (kaggle package 2.x, installed in .venv via the 'tools' extra):
  .venv\\Scripts\\kaggle auth login          browser login; credentials are cached in your user folder
  or: generate a token at https://www.kaggle.com/settings/api and save it as C:\\Users\\<you>\\.kaggle\\access_token
  (a credential: it lives outside the repo and is never committed)

    python scripts/download_supplements.py --drives 0009 0010
    python scripts/download_supplements.py --drives 0009 --deep           # also verify every zip member's CRC
    python scripts/download_supplements.py --drives 0009 --notebook nanimack/kitti360-supplement
        # the latest output of the notebook instead of a dataset (before the dataset exists)

Parts downloaded with notebook 01's download cell need no script: unzip them all into data/kitti360_supplements/<drive>/,
then run this script for that drive to check the copy (a complete copy is checked and skipped, no login needed).

Paper drives (kitti360_paper_v1): train 0000, 0002, 0004, 0005 · val 0006 · test 0009, 0010.
"""
import argparse
import shutil
import zipfile
from pathlib import Path

from ocgf.datasets.supplement_check import check_supplement, find_root

REPO = Path(__file__).resolve().parents[1]
PAPER_DRIVES = ("0000", "0002", "0004", "0005", "0006", "0009", "0010")


def kaggle_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise SystemExit("the kaggle package is missing: .venv\\Scripts\\python -m pip install kaggle")
    api = KaggleApi()
    try:
        api.authenticate()
    except (OSError, ValueError) as err:
        raise SystemExit(f"Kaggle credentials not found ({err}). Run '.venv\\Scripts\\kaggle auth login', or save a "
                         "token from https://www.kaggle.com/settings/api as ~/.kaggle/access_token.")
    return api


def unzip_leftovers(folder: Path) -> None:
    """The API may leave single files zipped ('x.csv.zip'); unpack those, never our own shard_*.zip files."""
    for z in folder.rglob("*.zip"):
        if not z.name.startswith("shard_") and z.name != "dev_sample.zip":
            with zipfile.ZipFile(z) as f:
                f.extractall(z.parent)
            z.unlink()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drives", nargs="+", default=list(PAPER_DRIVES), help="e.g. 0009 0010 (default: all paper drives)")
    p.add_argument("--owner", default="nanimack", help="Kaggle user that owns the datasets")
    p.add_argument("--notebook", help="download the latest output of this notebook (owner/slug) instead of datasets")
    p.add_argument("--out", type=Path, default=REPO / "data" / "kitti360_supplements")
    p.add_argument("--force", action="store_true", help="download again even if a checked copy exists")
    p.add_argument("--deep", action="store_true", help="verify every zip member's CRC (reads all bytes)")
    args = p.parse_args(argv)
    if args.notebook and len(args.drives) != 1:
        raise SystemExit("--notebook holds one drive (its latest run): pass exactly one --drives value")

    api = None
    results = {}
    for drive in args.drives:
        seq = f"2013_05_28_drive_{drive}_sync"
        target = args.out / drive
        root = find_root(target) if target.exists() else None
        if root and not args.force:
            summary, problems = check_supplement(root, seq, deep=args.deep)
            if not problems:
                print(f"{drive}: already here and complete ({summary['kept']} frames, {summary['gb']} GB), skipped")
                results[drive] = "ok (existing)"
                continue
            print(f"{drive}: existing copy has problems ({'; '.join(problems)}), downloading again")
        api = api or kaggle_api()                                # log in before touching the target folder
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        try:
            if args.notebook:
                print(f"{drive}: downloading the latest output of notebook {args.notebook} ...", flush=True)
                api.kernels_output(args.notebook, path=str(target), force=True, quiet=False)
            else:
                ref = f"{args.owner}/kitti360-d{drive}-supplement"
                print(f"{drive}: downloading dataset {ref} ...", flush=True)
                api.dataset_download_files(ref, path=str(target), unzip=True, quiet=False, force=True)
        except Exception as err:            # the Kaggle SDK raises requests' HTTPError (403 / 404 for a missing dataset)
            shutil.rmtree(target, ignore_errors=True)
            what = f"notebook {args.notebook}" if args.notebook else f"dataset {args.owner}/kitti360-d{drive}-supplement"
            print(f"{drive}: FAILED, {what} could not be downloaded ({err}).\n"
                  f"   A 403/404 usually means it does not exist under that name yet: create it from the run's Output tab "
                  f"(New Dataset, title kitti360-d{drive}-supplement), or use --notebook <owner>/<notebook-slug> once the "
                  f"run's saved version is complete.")
            results[drive] = "failed: not downloadable"
            continue
        unzip_leftovers(target)
        root = find_root(target)
        if root is None:
            print(f"{drive}: FAILED, no manifest.json in the download")
            results[drive] = "failed: no manifest"
            continue
        summary, problems = check_supplement(root, seq, deep=args.deep)
        if problems:
            print(f"{drive}: PROBLEMS: " + "; ".join(problems))
            results[drive] = "problems"
        else:
            print(f"{drive}: ok, {summary['role']} drive, {summary['kept']} frames, {summary['gb']} GB in {root}")
            results[drive] = "ok"
    print("\nsummary: " + ", ".join(f"{d} {r}" for d, r in results.items()))


if __name__ == "__main__":
    main()

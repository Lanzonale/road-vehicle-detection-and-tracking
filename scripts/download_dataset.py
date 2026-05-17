#!/usr/bin/env python3
"""Download the Kaggle Road Vehicle Images Dataset and unzip it.

Before running:
1. Create a Kaggle API token from your Kaggle account settings.
2. Put kaggle.json at ~/.kaggle/kaggle.json.
3. Run: chmod 600 ~/.kaggle/kaggle.json

Usage:
    python scripts/download_dataset.py --out datasets/road_vehicle
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATASET_SLUG = "ashfakyeafi/road-vehicle-images-dataset"


def run_command(cmd: list[str]) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def unzip_file(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Unzipping {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def find_dataset_root(out_dir: Path) -> Path:
    """Try to find the folder containing train/images and valid/images."""
    candidates = [out_dir] + [p for p in out_dir.rglob("*") if p.is_dir()]
    for p in candidates:
        if (p / "train" / "images").exists() and (p / "valid" / "images").exists():
            return p
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("datasets/road_vehicle"))
    parser.add_argument("--force", action="store_true", help="Remove existing output directory first.")
    args = parser.parse_args()

    if args.force and args.out.exists():
        print(f"[INFO] Removing existing directory: {args.out}")
        shutil.rmtree(args.out)

    args.out.mkdir(parents=True, exist_ok=True)

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("[ERROR] Kaggle API token not found at ~/.kaggle/kaggle.json")
        print("        Download it from Kaggle account settings, then run:")
        print("        mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)

    download_dir = args.out.parent
    run_command([
        "kaggle", "datasets", "download",
        "-d", DATASET_SLUG,
        "-p", str(download_dir),
    ])

    zip_files = sorted(download_dir.glob("road-vehicle-images-dataset*.zip"))
    if not zip_files:
        print("[ERROR] Could not find downloaded zip file.")
        sys.exit(1)

    unzip_file(zip_files[-1], args.out)
    root = find_dataset_root(args.out)
    print(f"[INFO] Dataset root candidate: {root}")
    print("[INFO] Expected YAML path setting: path:", root.as_posix())
    print("[INFO] Done.")


if __name__ == "__main__":
    main()

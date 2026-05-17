#!/usr/bin/env python3
"""Check YOLO dataset structure and basic label statistics."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_split(root: Path, image_rel: str, label_rel: str | None, nc: int) -> dict:
    image_dir = root / image_rel
    if label_rel is None:
        # YOLO convention: replace /images with /labels
        label_dir = Path(str(image_dir).replace("images", "labels"))
    else:
        label_dir = root / label_rel

    images = [p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    labels = [label_dir / f"{p.stem}.txt" for p in images]

    missing = [p for p in labels if not p.exists()]
    class_counter = Counter()
    bad_lines = []

    for label_path in labels:
        if not label_path.exists():
            continue
        for i, line in enumerate(label_path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                bad_lines.append((label_path, i, line))
                continue
            try:
                cls = int(float(parts[0]))
                vals = list(map(float, parts[1:]))
            except ValueError:
                bad_lines.append((label_path, i, line))
                continue
            if cls < 0 or cls >= nc or any(v < 0 or v > 1 for v in vals):
                bad_lines.append((label_path, i, line))
            class_counter[cls] += 1

    return {
        "image_dir": image_dir,
        "label_dir": label_dir,
        "num_images": len(images),
        "num_missing_labels": len(missing),
        "num_bad_lines": len(bad_lines),
        "class_counter": class_counter,
        "missing_examples": missing[:10],
        "bad_examples": bad_lines[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("configs/road_vehicle.yaml"))
    args = parser.parse_args()

    cfg = load_yaml(args.data)
    root = Path(cfg.get("path", "."))
    names = cfg["names"]
    nc = len(names)

    print(f"[INFO] Dataset root: {root.resolve()}")
    print(f"[INFO] Number of classes: {nc}")

    for split_name, image_rel in [("train", cfg["train"]), ("val", cfg["val"] if "val" in cfg else cfg.get("valid"))]:
        if image_rel is None:
            continue
        stats = count_split(root, image_rel, None, nc)
        print("\n" + "=" * 80)
        print(f"[{split_name.upper()}]")
        print(f"images: {stats['num_images']}")
        print(f"missing label files: {stats['num_missing_labels']}")
        print(f"bad label lines: {stats['num_bad_lines']}")
        print("class counts:")
        for k in range(nc):
            class_name = names[k] if isinstance(names, list) else names.get(k, names.get(str(k), str(k)))
            print(f"  {k:2d} {class_name:22s}: {stats['class_counter'].get(k, 0)}")

        if stats["missing_examples"]:
            print("missing examples:")
            for p in stats["missing_examples"]:
                print("  ", p)
        if stats["bad_examples"]:
            print("bad examples:")
            for p, line_no, line in stats["bad_examples"]:
                print(f"  {p}:{line_no}: {line}")


if __name__ == "__main__":
    main()

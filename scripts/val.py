#!/usr/bin/env python3
"""Validate a trained YOLOv8 detector."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data", type=str, default="configs/road_vehicle.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output-json", type=Path, default=Path("results/val_metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        plots=True,
    )

    summary = {
        "weights": args.weights,
        "data": args.data,
        "box_map50": float(metrics.box.map50),
        "box_map50_95": float(metrics.box.map),
        "box_precision_mean": float(metrics.box.mp),
        "box_recall_mean": float(metrics.box.mr),
    }
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[VALIDATION SUMMARY]")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"[INFO] Saved metrics to {args.output_json}")


if __name__ == "__main__":
    main()

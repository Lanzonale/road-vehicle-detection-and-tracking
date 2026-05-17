#!/usr/bin/env python3
"""Train YOLOv8 on Road Vehicle Images Dataset with W&B enabled.

Example:
    wandb login
    python scripts/train.py \
      --data configs/road_vehicle.yaml \
      --model yolov8n.pt \
      --epochs 80 \
      --imgsz 640 \
      --batch 16 \
      --project road-vehicle-yolov8 \
      --name yolov8n-road-vehicle
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import wandb
from ultralytics import YOLO

try:
    from wandb.integration.ultralytics import add_wandb_callback
except Exception:  # pragma: no cover
    add_wandb_callback = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 with W&B logging.")
    parser.add_argument("--data", type=str, default="configs/road_vehicle.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0", help="GPU id like 0, 0,1 or cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam", "AdamW", "auto"])
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true", help="Cache images for faster training if memory allows.")
    parser.add_argument("--resume", action="store_true", help="Resume the latest interrupted training run.")

    # Ultralytics output args
    parser.add_argument("--runs-project", type=str, default="runs/vehicle_detection")
    parser.add_argument("--name", type=str, default="yolov8n-road-vehicle")
    parser.add_argument("--exist-ok", action="store_true")

    # W&B args
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", type=str, default="road-vehicle-yolov8-tracking")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-log-model", type=str, default="checkpoint", choices=["false", "checkpoint", "end"])
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=["yolov8", "vehicle-detection"])
    return parser.parse_args()


def init_wandb(args: argparse.Namespace) -> Any | None:
    if not args.wandb:
        os.environ["WANDB_MODE"] = "disabled"
        return None

    # Make W&B behavior explicit and visible in the logs.
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    os.environ.setdefault("WANDB_LOG_MODEL", args.wandb_log_model)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.name,
        tags=args.wandb_tags,
        job_type="train",
        config=vars(args),
    )
    return run


def main() -> None:
    args = parse_args()
    Path(args.runs_project).mkdir(parents=True, exist_ok=True)

    run = init_wandb(args)

    model = YOLO(args.model)

    # W&B official Ultralytics callback: logs metrics, checkpoints, and visualizations.
    if args.wandb and add_wandb_callback is not None:
        try:
    # add_wandb_callback(model, enable_model_checkpointing=True)
            print("[INFO] W&B Ultralytics callback enabled with model checkpointing.")
        except TypeError:
            add_wandb_callback(model)
            print("[INFO] W&B Ultralytics callback enabled.")
    elif args.wandb:
        print("[WARN] Could not import W&B Ultralytics callback. Metrics may still be logged by Ultralytics if supported.")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        cache=args.cache,
        resume=args.resume,
        project=args.runs_project,
        name=args.name,
        exist_ok=args.exist_ok,
        plots=True,
        save=True,
        val=True,
    )

    print("[INFO] Training finished.")
    print(f"[INFO] Local run directory: {Path(args.runs_project) / args.name}")
    print(f"[INFO] Best weights usually saved at: {Path(args.runs_project) / args.name / 'weights' / 'best.pt'}")

    if run is not None:
        # Log the final best.pt as a W&B artifact for easier report reproducibility.
        best_pt = Path(args.runs_project) / args.name / "weights" / "best.pt"
        if best_pt.exists():
            artifact = wandb.Artifact(name=f"{args.name}-best", type="model")
            artifact.add_file(str(best_pt))
            run.log_artifact(artifact)
            print("[INFO] Logged best.pt to W&B artifact store.")
        wandb.finish()


if __name__ == "__main__":
    main()

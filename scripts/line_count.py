#!/usr/bin/env python3
"""Video multi-object tracking + line-crossing counting.

Counting rule:
For each stable Tracking ID, keep the sign of the bounding-box center point
relative to a virtual line. If the same ID moves from one side of the line to
the other side and has not been counted before, add one count.

Example:
    python scripts/line_count.py \
      --weights runs/vehicle_detection/yolov8n-road-vehicle/weights/best.pt \
      --source videos/test_video.mp4 \
      --output results/line_count_output.mp4 \
      --line 200 450 1100 450 \
      --save-frames 120 121 122 123
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/line_count_output.mp4"))
    parser.add_argument("--tracker", type=str, default="botsort.yaml", choices=["botsort.yaml", "bytetrack.yaml"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--line", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"), default=[200, 450, 1100, 450])
    parser.add_argument("--dead-zone", type=float, default=5.0, help="Ignore points too close to the line to reduce jitter counting.")
    parser.add_argument("--max-trace", type=int, default=30)
    parser.add_argument("--save-frames", type=int, nargs="*", default=[], help="Frame indices to save as images for occlusion/ID analysis.")
    parser.add_argument("--frame-dir", type=Path, default=Path("results/occlusion_frames"))
    parser.add_argument("--events-csv", type=Path, default=Path("results/crossing_events.csv"))
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def point_side(point: tuple[int, int], line_p1: tuple[int, int], line_p2: tuple[int, int]) -> float:
    """Return signed area: sign indicates which side of a directed line the point is on."""
    x, y = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    return float((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1))


def draw_text(frame: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.7, thickness: int = 2) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), thickness, cv2.LINE_AA)


def write_events(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("frame,track_id,class_id,class_name,cx,cy,total_count\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.frame_dir.mkdir(parents=True, exist_ok=True)

    line_p1 = (args.line[0], args.line[1])
    line_p2 = (args.line[2], args.line[3])
    save_frame_set = set(args.save_frames)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    model = YOLO(args.weights)
    names = model.names

    # Track state.
    last_side: dict[int, int] = {}
    counted_ids: set[int] = set()
    track_history: dict[int, list[tuple[int, int]]] = defaultdict(list)
    crossing_events: list[dict] = []
    total_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        result = model.track(
            source=frame,
            persist=True,
            tracker=args.tracker,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

        annotated = result.plot()

        # Draw counting line and UI panel.
        cv2.line(annotated, line_p1, line_p2, (0, 0, 255), 3)
        draw_text(annotated, "Counting Line", (line_p1[0], max(25, line_p1[1] - 10)), 0.7, 2)

        if result.boxes is not None and result.boxes.id is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()
            cls_ids = result.boxes.cls.int().cpu().tolist()

            for box, track_id, cls_id in zip(boxes_xyxy, track_ids, cls_ids):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Center point.
                cv2.circle(annotated, (cx, cy), 4, (255, 0, 0), -1)

                # Trajectory line.
                track_history[track_id].append((cx, cy))
                if len(track_history[track_id]) > args.max_trace:
                    track_history[track_id].pop(0)
                pts = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], isClosed=False, color=(255, 255, 0), thickness=2)

                # Line-crossing logic.
                side_value = point_side((cx, cy), line_p1, line_p2)
                if abs(side_value) <= args.dead_zone:
                    continue

                current_side = 1 if side_value > 0 else -1
                previous_side = last_side.get(track_id)

                if previous_side is not None and previous_side != current_side and track_id not in counted_ids:
                    counted_ids.add(track_id)
                    total_count += 1
                    class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                    crossing_events.append({
                        "frame": frame_idx,
                        "track_id": track_id,
                        "class_id": cls_id,
                        "class_name": class_name,
                        "cx": cx,
                        "cy": cy,
                        "total_count": total_count,
                    })
                    cv2.circle(annotated, (cx, cy), 12, (0, 255, 0), 3)

                last_side[track_id] = current_side

        # HUD.
        cv2.rectangle(annotated, (15, 15), (460, 95), (0, 0, 0), -1)
        draw_text(annotated, f"Frame: {frame_idx}", (30, 45), 0.8, 2)
        draw_text(annotated, f"Total crossed vehicles: {total_count}", (30, 80), 0.8, 2)

        if frame_idx in save_frame_set:
            out_img = args.frame_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_img), annotated)
            print(f"[INFO] Saved frame {frame_idx}: {out_img}")

        writer.write(annotated)
        if args.show:
            cv2.imshow("line-count", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    write_events(args.events_csv, crossing_events)
    print(f"[INFO] Saved line-counting video to {args.output}")
    print(f"[INFO] Saved crossing events to {args.events_csv}")
    print(f"[INFO] Total crossed vehicles: {total_count}")


if __name__ == "__main__":
    main()

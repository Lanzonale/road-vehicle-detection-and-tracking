#!/usr/bin/env python3
"""Extract consecutive frames from an annotated tracking video for occlusion analysis.

Example:
    python scripts/extract_occlusion_frames.py \
      --source results/line_count_output.mp4 \
      --start 120 \
      --num 4 \
      --out results/occlusion_frames
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--start", type=int, required=True, help="1-based starting frame index.")
    parser.add_argument("--num", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results/occlusion_frames"))
    parser.add_argument("--prefix", type=str, default="occlusion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.source}")

    targets = set(range(args.start, args.start + args.num))
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx in targets:
            cv2.putText(
                frame,
                f"Frame {frame_idx}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            out_path = args.out / f"{args.prefix}_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            print(f"[INFO] Saved {out_path}")
            saved += 1

    cap.release()
    print(f"[INFO] Saved {saved} frame(s).")


if __name__ == "__main__":
    main()

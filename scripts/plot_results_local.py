#!/usr/bin/env python3
"""Plot local Ultralytics results.csv as backup figures for the report.

The assignment requires W&B or SwanLab screenshots. Use W&B screenshots as the
main evidence, and use these local plots only as backup or appendix material.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/local_plots"))
    return parser.parse_args()


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def plot_columns(df: pd.DataFrame, columns: list[str], title: str, out_path: Path) -> None:
    existing = [c for c in columns if c in df.columns]
    if not existing:
        print(f"[WARN] No columns found for {title}: {columns}")
        return
    plt.figure(figsize=(8, 5))
    x = df["epoch"] if "epoch" in df.columns else range(len(df))
    for c in existing:
        plt.plot(x, df[c], label=c)
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = clean_columns(pd.read_csv(args.results_csv))

    plot_columns(
        df,
        ["train/box_loss", "train/cls_loss", "train/dfl_loss"],
        "Training Loss Curves",
        args.out_dir / "train_losses.png",
    )
    plot_columns(
        df,
        ["val/box_loss", "val/cls_loss", "val/dfl_loss"],
        "Validation Loss Curves",
        args.out_dir / "val_losses.png",
    )
    plot_columns(
        df,
        ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"],
        "Validation Metrics Curves",
        args.out_dir / "val_metrics.png",
    )


if __name__ == "__main__":
    main()

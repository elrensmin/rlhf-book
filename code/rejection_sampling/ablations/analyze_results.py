#!/usr/bin/env python3
"""Build the result table and figures for the rejection-sampling choice-axis
campaign from W&B run data.

This is a thin helper. It reads a CSV of runs (one row per run) exported from
W&B and produces:

  - a markdown results table
  - reward-vs-random gap plots vs. N, K, model size, and RM

Expected CSV columns (from a W&B export):
  run_name, strategy, test_accuracy, num_completions_per_prompt,
  top_k, model_name, reward_model_name, url

Usage:
  uv run python rejection_sampling/ablations/analyze_results.py \
      --runs path/to/runs.csv \
      --out-dir rejection_sampling/ablations/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_runs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["strategy"] = df["strategy"].astype(str)
    return df


def gap_plot(df: pd.DataFrame, x: str, xlabel: str, out_dir: Path, title: str) -> Path:
    """Plot reward-vs-random test_accuracy gap against a swept variable."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for strategy, color in [("top", "#1f77b4"), ("random", "#aec7e8")]:
        sub = df[df["strategy"].str.startswith(strategy)]
        sub = sub.sort_values(x)
        ax.plot(sub[x], sub["test_accuracy"], marker="o", label=strategy, color=color, lw=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test_accuracy")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"gap_{x}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=str, required=True, help="Path to W&B runs CSV")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="rejection_sampling/ablations/figures",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_runs(Path(args.runs))

    # Results table
    cols = [
        "run_name",
        "strategy",
        "num_completions_per_prompt",
        "top_k",
        "model_name",
        "reward_model_name",
        "test_accuracy",
        "url",
    ]
    table = df[[c for c in cols if c in df.columns]]
    table.to_csv(out_dir / "results_table.csv", index=False)
    print(f"Saved results table: {out_dir / 'results_table.csv'}")

    # Gap plots
    if "num_completions_per_prompt" in df.columns:
        print(
            gap_plot(
                df, "num_completions_per_prompt", "N (completions/prompt)", out_dir, "E1: gap vs N"
            )
        )
    if "top_k" in df.columns:
        print(gap_plot(df, "top_k", "top_k", out_dir, "E2: gap vs K"))


if __name__ == "__main__":
    main()

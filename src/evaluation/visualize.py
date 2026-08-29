"""
visualize.py
============
Automated benchmark visualizations from evaluation summaries (Module 5).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization. Install with: pip install matplotlib"
        ) from exc


def plot_method_comparison(
    summary_csv: Path,
    output_dir: Path,
    *,
    level: str = "level3",
    metrics: Optional[List[str]] = None,
) -> List[Path]:
    """Bar charts comparing methods on key metrics."""
    plt = _require_matplotlib()

    df = pd.read_csv(summary_csv)
    df = df[df["level"] == level].copy()
    if df.empty:
        logger.warning("No rows for level=%s in %s", level, summary_csv)
        return []

    metrics = metrics or [
        "accuracy_mean", "macro_f1_mean", "ari_mean", "nmi_mean",
        "silhouette_mean", "davies_bouldin_mean", "marker_purity_mean",
        "neighborhood_consistency_mean", "spatial_entropy_mean",
    ]
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        logger.warning("No plottable metrics found in %s", summary_csv)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for metric in metrics:
        plot_df = df[["method", metric]].dropna().sort_values(metric, ascending=False)
        if plot_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 0.5), 5))
        ax.bar(plot_df["method"], plot_df[metric], color="steelblue")
        ax.set_title(f"{metric.replace('_mean', '').replace('_', ' ').title()} ({level})")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()

        out_path = output_dir / f"{metric}_{level}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        written.append(out_path)
        logger.info("Wrote %s", out_path)

    return written


def plot_overall_leaderboard(
    summary_csv: Path,
    output_path: Path,
    *,
    level: str = "level3",
    top_n: int = 15,
) -> Optional[Path]:
    """Horizontal bar chart of overall_score across methods."""
    plt = _require_matplotlib()

    df = pd.read_csv(summary_csv)
    df = df[df["level"] == level].copy()
    if "overall_score" not in df.columns or df.empty:
        return None

    plot_df = df.nlargest(top_n, "overall_score")
    fig, ax = plt.subplots(figsize=(8, max(4, len(plot_df) * 0.35)))
    ax.barh(plot_df["method"], plot_df["overall_score"], color="darkorange")
    ax.set_xlabel("Overall score")
    ax.set_title(f"Method leaderboard ({level})")
    ax.invert_yaxis()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote leaderboard: %s", output_path)
    return output_path


def generate_report_plots(
    dataset: str,
    results_root: Path,
    *,
    level: str = "level3",
) -> List[Path]:
    """Generate all standard plots for a dataset from ``final_results.csv``."""
    summary_csv = results_root / dataset / "summary" / "final_results.csv"
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Summary not found: {summary_csv}")

    out_dir = results_root / dataset / "summary" / "plots"
    paths = plot_method_comparison(summary_csv, out_dir, level=level)
    lb = plot_overall_leaderboard(
        summary_csv,
        out_dir / f"leaderboard_{level}.png",
        level=level,
    )
    if lb:
        paths.append(lb)
    return paths

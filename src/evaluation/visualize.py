"""
visualize.py
============
Automated benchmark visualizations from evaluation summaries (Module 5).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from reporting import (
    build_confusion_dataframe,
    collect_method_predictions,
    export_all_report_tables,
    phenotype_counts,
    top_methods_by_score,
)

logger = logging.getLogger(__name__)

_SUPERVISED_RANGE_METRICS = [
    "accuracy", "macro_f1", "weighted_f1", "mcc", "kappa",
    "ari", "nmi", "r2_composition", "pearson_composition",
]
_MULTI_METRIC_COLS = [
    "accuracy_mean", "macro_f1_mean", "weighted_f1_mean", "ari_mean", "nmi_mean",
]


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization. Install with: pip install matplotlib"
        ) from exc


def _save_fig(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    logger.info("Wrote %s", path)
    return path


def _phenotype_colors(phenotypes: Sequence[str]) -> Dict[str, tuple]:
    plt = _require_matplotlib()
    cmap = plt.get_cmap("tab20")
    return {p: cmap(i % 20) for i, p in enumerate(phenotypes)}


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
        "stability", "notebook_overall_score",
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
        written.append(_save_fig(fig, output_dir / f"{metric}_{level}.png"))

    return written


def plot_overall_leaderboard(
    summary_csv: Path,
    output_path: Path,
    *,
    level: str = "level3",
    top_n: int = 15,
    score_col: str = "overall_score",
) -> Optional[Path]:
    """Horizontal bar chart of overall_score across methods."""
    plt = _require_matplotlib()

    df = pd.read_csv(summary_csv)
    df = df[df["level"] == level].copy()
    if score_col not in df.columns and "notebook_overall_score" in df.columns:
        score_col = "notebook_overall_score"
    if score_col not in df.columns or df.empty:
        return None

    plot_df = df.nlargest(top_n, score_col)
    fig, ax = plt.subplots(figsize=(8, max(4, len(plot_df) * 0.35)))
    ax.barh(plot_df["method"], plot_df[score_col], color="darkorange")
    ax.set_xlabel(score_col.replace("_", " "))
    ax.set_title(f"Method leaderboard ({level})")
    ax.invert_yaxis()
    plt.tight_layout()
    return _save_fig(fig, output_path)


def plot_metric_fold_ranges(
    per_fold_csv: Path,
    output_path: Path,
    *,
    level: str = "level3",
    metrics: Optional[List[str]] = None,
) -> Optional[Path]:
    """Fold min–max ranges per method (``calc_metrics.ipynb`` style)."""
    plt = _require_matplotlib()

    df = pd.read_csv(per_fold_csv)
    df = df[df["level"] == level].copy()
    if df.empty:
        return None

    metrics = metrics or [m for m in _SUPERVISED_RANGE_METRICS if m in df.columns]
    if not metrics:
        return None

    methods = sorted(df["method"].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), max(4, len(methods) * 0.3)))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for method in methods:
            vals = df.loc[df["method"] == method, metric].dropna()
            if vals.empty:
                continue
            vmin, vmax = vals.min(), vals.max()
            ax.barh(method, vmax - vmin, left=vmin, height=0.6)
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("Fold range")

    fig.suptitle(f"Metric fold ranges ({level})")
    plt.tight_layout()
    return _save_fig(fig, output_path)


def plot_multi_metric_comparison(
    summary_csv: Path,
    output_path: Path,
    *,
    level: str = "level3",
    metrics: Optional[List[str]] = None,
) -> Optional[Path]:
    """Grouped horizontal bars across multiple metrics (``calc_metrics.ipynb``)."""
    plt = _require_matplotlib()

    df = pd.read_csv(summary_csv)
    df = df[df["level"] == level].copy()
    metrics = metrics or [m for m in _MULTI_METRIC_COLS if m in df.columns]
    if df.empty or not metrics:
        return None

    plot_df = df.set_index("method")[metrics].dropna(how="all")
    if plot_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, max(4, len(plot_df) * 0.45)))
    plot_df.plot(kind="barh", ax=ax, colormap="viridis", width=0.8)
    ax.set_title(f"Multi-metric comparison ({level})")
    ax.set_xlabel("Score")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    return _save_fig(fig, output_path)


def plot_metrics_by_level(
    summary_csv: Path,
    output_dir: Path,
    *,
    metrics: Optional[List[str]] = None,
) -> List[Path]:
    """Line/bar plots comparing level1/2/3 for each method."""
    plt = _require_matplotlib()

    df = pd.read_csv(summary_csv)
    metrics = metrics or ["accuracy_mean", "macro_f1_mean", "notebook_overall_score"]
    metrics = [m for m in metrics if m in df.columns]
    if df.empty or not metrics:
        return []

    written: List[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    level_order = ["level1", "level2", "level3"]

    for metric in metrics:
        pivot = df.pivot_table(index="method", columns="level", values=metric, aggfunc="mean")
        cols = [c for c in level_order if c in pivot.columns]
        if not cols:
            continue
        pivot = pivot[cols].dropna(how="all")
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.35)))
        pivot.plot(kind="barh", ax=ax)
        ax.set_title(f"{metric.replace('_mean', '')} by hierarchy level")
        ax.set_xlabel(metric)
        ax.legend(title="Level")
        plt.tight_layout()
        written.append(_save_fig(fig, output_dir / f"{metric}_by_level.png"))

    return written


def plot_composition_stacked(
    counts: pd.Series,
    output_path: Path,
    *,
    title: str,
    horizontal: bool = True,
) -> Optional[Path]:
    """Stacked bar of phenotype fractions (``celltype_composition.ipynb``)."""
    plt = _require_matplotlib()
    if counts.empty:
        return None

    total = counts.sum()
    colors = _phenotype_colors(counts.index.tolist())
    fig, ax = plt.subplots(figsize=(10, 2.5 if horizontal else 6))
    left = 0.0
    for phenotype, count in counts.items():
        width = count / total if total else 0
        if horizontal:
            ax.barh(["Composition"], [width], left=left, color=colors[phenotype], label=phenotype)
        else:
            ax.bar([phenotype], [count], color=colors[phenotype])
        left += width

    ax.set_title(title)
    if horizontal:
        ax.set_xlabel("Fraction")
        ax.set_xlim(0, 1)
    else:
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=90)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    return _save_fig(fig, output_path)


def plot_composition_grouped(
    true_counts: pd.Series,
    pred_counts: pd.Series,
    output_path: Path,
    *,
    title: str,
) -> Optional[Path]:
    """Grouped bar comparing ground truth vs predicted composition."""
    plt = _require_matplotlib()
    phenotypes = sorted(set(true_counts.index) | set(pred_counts.index))
    if not phenotypes:
        return None

    x = np.arange(len(phenotypes))
    width = 0.38
    true_vals = [true_counts.get(p, 0) for p in phenotypes]
    pred_vals = [pred_counts.get(p, 0) for p in phenotypes]

    fig, ax = plt.subplots(figsize=(max(8, len(phenotypes) * 0.45), 5))
    ax.bar(x - width / 2, true_vals, width, label="Ground truth", color="steelblue")
    ax.bar(x + width / 2, pred_vals, width, label="Predicted", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=90)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return _save_fig(fig, output_path)


def plot_confusion_matrix_heatmap(
    y_true: pd.Series,
    y_pred: pd.Series,
    output_path: Path,
    *,
    title: str,
) -> Optional[Path]:
    """Confusion matrix heatmap (``eval_mapping.ipynb`` style)."""
    plt = _require_matplotlib()
    cm_df, labels = build_confusion_dataframe(y_true, y_pred)
    if cm_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.45), max(5, len(labels) * 0.4)))
    try:
        import seaborn as sns
        sns.heatmap(cm_df, annot=len(labels) <= 15, fmt="d", cmap="Blues", ax=ax, cbar=True)
    except ImportError:
        im = ax.imshow(cm_df.values, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    return _save_fig(fig, output_path)


def plot_spatial_phenotypes(
    df: pd.DataFrame,
    output_path: Path,
    *,
    label_col: str = "predicted_phenotype",
    title: str,
    max_points: int = 25000,
) -> Optional[Path]:
    """Spatial scatter colored by phenotype."""
    plt = _require_matplotlib()
    if "x" not in df.columns or "y" not in df.columns or label_col not in df.columns:
        return None

    plot_df = df.dropna(subset=["x", "y", label_col]).copy()
    if plot_df.empty:
        return None
    if len(plot_df) > max_points:
        plot_df = plot_df.sample(max_points, random_state=42)

    phenotypes = sorted(plot_df[label_col].astype(str).unique())
    colors = _phenotype_colors(phenotypes)
    fig, ax = plt.subplots(figsize=(7, 7))
    for phenotype in phenotypes:
        mask = plot_df[label_col].astype(str) == phenotype
        ax.scatter(
            plot_df.loc[mask, "x"],
            plot_df.loc[mask, "y"],
            s=3,
            alpha=0.7,
            c=[colors[phenotype]],
            label=phenotype,
            linewidths=0,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(markerscale=3, fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    return _save_fig(fig, output_path)


def generate_report_plots(
    dataset: str,
    results_root: Path,
    *,
    level: str = "level3",
    per_fold_csv: Optional[Path] = None,
    quant_path: Optional[Path] = None,
    marker_cols: Optional[List[str]] = None,
    summary_df: Optional[pd.DataFrame] = None,
    top_n_spatial: int = 3,
    top_n_composition: int = 8,
) -> List[Path]:
    """Generate notebook-style tables and plots for a dataset."""
    summary_csv = results_root / dataset / "summary" / "final_results.csv"
    if summary_df is None:
        if not summary_csv.is_file():
            raise FileNotFoundError(f"Summary not found: {summary_csv}")
        summary_df = pd.read_csv(summary_csv)

    per_fold_csv = per_fold_csv or (results_root / dataset / "summary" / "per_fold_metrics.csv")
    out_dir = results_root / dataset / "summary" / "plots"
    dataset_results = results_root / dataset
    paths: List[Path] = []

    paths.extend(plot_method_comparison(summary_csv, out_dir, level=level))
    lb = plot_overall_leaderboard(summary_csv, out_dir / f"leaderboard_{level}.png", level=level)
    if lb:
        paths.append(lb)
    nb_lb = plot_overall_leaderboard(
        summary_csv,
        out_dir / f"notebook_leaderboard_{level}.png",
        level=level,
        score_col="notebook_overall_score",
    )
    if nb_lb:
        paths.append(nb_lb)

    if per_fold_csv.is_file():
        fold_ranges = plot_metric_fold_ranges(
            per_fold_csv, out_dir / f"metric_fold_ranges_{level}.png", level=level,
        )
        if fold_ranges:
            paths.append(fold_ranges)

    multi = plot_multi_metric_comparison(
        summary_csv, out_dir / f"multi_metric_comparison_{level}.png", level=level,
    )
    if multi:
        paths.append(multi)

    paths.extend(plot_metrics_by_level(summary_csv, out_dir / "by_level"))

    tables = export_all_report_tables(
        dataset,
        results_root,
        summary=summary_df,
        quant_path=quant_path,
        marker_cols=marker_cols,
        level=level,
        top_n_methods=top_n_composition,
    )

    if quant_path and quant_path.is_file():
        quant = pd.read_csv(quant_path)
        if "cell_type" in quant.columns:
            stacked = plot_composition_stacked(
                phenotype_counts(quant["cell_type"]),
                out_dir / "composition_ground_truth_stacked.png",
                title=f"{dataset} ground-truth composition",
            )
            if stacked:
                paths.append(stacked)

    methods = top_methods_by_score(summary_df, level=level, top_n=top_n_composition)
    true_counts = pd.Series(dtype=int)
    if quant_path and quant_path.is_file():
        quant_cols = pd.read_csv(quant_path, nrows=0).columns
        if "cell_type" in quant_cols:
            quant = pd.read_csv(quant_path, usecols=["cell_type"])
            true_counts = phenotype_counts(quant["cell_type"])

    for method in methods:
        preds = collect_method_predictions(
            dataset_results,
            method,
            path_level=level,
            quant_path=quant_path,
            marker_cols=marker_cols,
        )
        if preds.empty:
            continue

        pred_counts = phenotype_counts(preds["predicted_phenotype"])
        stacked = plot_composition_stacked(
            pred_counts,
            out_dir / f"composition_{method}_stacked.png",
            title=f"{method} predicted composition",
        )
        if stacked:
            paths.append(stacked)

        if not true_counts.empty:
            grouped = plot_composition_grouped(
                true_counts,
                pred_counts,
                out_dir / f"composition_{method}_grouped.png",
                title=f"{method}: ground truth vs predicted",
            )
            if grouped:
                paths.append(grouped)

        if "true_phenotype" in preds.columns:
            cm_plot = plot_confusion_matrix_heatmap(
                preds["true_phenotype"],
                preds["predicted_phenotype"],
                out_dir / f"confusion_{method}_{level}.png",
                title=f"{method} confusion matrix ({level})",
            )
            if cm_plot:
                paths.append(cm_plot)

    spatial_methods = top_methods_by_score(summary_df, level=level, top_n=top_n_spatial)
    for method in spatial_methods:
        preds = collect_method_predictions(
            dataset_results,
            method,
            path_level=level,
            quant_path=quant_path,
            marker_cols=marker_cols,
        )
        if preds.empty or "x" not in preds.columns:
            continue

        if "Image_ID" in preds.columns:
            for image_id, grp in list(preds.groupby("Image_ID"))[:4]:
                spatial = plot_spatial_phenotypes(
                    grp,
                    out_dir / f"spatial_{method}_{image_id}.png",
                    title=f"{method} — {image_id}",
                )
                if spatial:
                    paths.append(spatial)
        else:
            spatial = plot_spatial_phenotypes(
                preds,
                out_dir / f"spatial_{method}.png",
                title=f"{method} spatial map",
            )
            if spatial:
                paths.append(spatial)

    logger.info("Generated %d plot(s) for %s", len(paths), dataset)
    return paths

"""
evaluator.py
============
Core evaluation loop: scan results/, compute metrics, aggregate across folds.
"""

from __future__ import annotations

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import pandas as pd

from metrics import compute_supervised_metrics, compute_unsupervised_metrics
from performance import find_performance_file
from prediction_io import (
    iter_method_predictions,
    normalize_columns,
    resolve_labels_for_level,
)
from quant_merge import enrich_from_quant, infer_marker_columns
from reporting import notebook_overall_score, stability_score

_UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(_UTILS) not in sys.path:
    sys.path.insert(0, str(_UTILS))

from parallel import resolve_worker_count  # noqa: E402
from prediction_files import evaluation_levels_for_file  # noqa: E402

_METHODS_UTILS = Path(__file__).resolve().parents[1] / "methods" / "utils"
if str(_METHODS_UTILS) not in sys.path:
    sys.path.insert(0, str(_METHODS_UTILS))

_BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmark"
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))
from method_registry import METHOD_REGISTRY, MethodCategory  # noqa: E402

LEVELS = ["level1", "level2", "level3"]


def _resolve_kfold_strategy(method: str) -> Optional[str]:
    if method.endswith("_progressive"):
        return "ProgressiveKFold"
    if method in METHOD_REGISTRY and METHOD_REGISTRY[method].category == MethodCategory.SUPERVISED_KFOLD:
        return "StratifiedKFold"
    return None


logger = logging.getLogger(__name__)

SUPERVISED_METRIC_COLS = [
    "accuracy", "macro_f1", "weighted_f1", "ari", "nmi", "mcc", "kappa",
    "hierarchical_f1", "g_mean", "r2_composition", "pearson_composition",
    "kl_divergence", "jensen_shannon",
    "rare_macro_f1", "common_macro_f1", "min_class_recall", "rare_min_recall",
    "n_rare_types", "n_common_types",
]
UNSUPERVISED_METRIC_COLS = [
    "silhouette", "davies_bouldin", "spatial_entropy", "marker_consistency",
    "marker_purity", "neighborhood_consistency",
]
METRIC_COLS = SUPERVISED_METRIC_COLS + UNSUPERVISED_METRIC_COLS


def evaluate_predictions_file(
    path: Path,
    level: str = "level3",
    hierarchy_path: Optional[Path] = None,
    *,
    quant_path: Optional[Path] = None,
    marker_cols: Optional[List[str]] = None,
    marker_rules: Optional[dict] = None,
    df: Optional[pd.DataFrame] = None,
    unsupervised: Optional[dict] = None,
    rare_fraction: float = 0.01,
    common_fraction: float = 0.05,
) -> dict:
    """Evaluate a single predictions CSV (supervised + unsupervised metrics)."""
    if df is None:
        df = normalize_columns(pd.read_csv(path))
        df = enrich_from_quant(df, quant_path, marker_cols=marker_cols)
    else:
        df = normalize_columns(df)

    yt, yp = resolve_labels_for_level(df, level)
    if yp is None:
        raise ValueError(f"Could not resolve predicted labels in {path}")

    has_gt = yt is not None and yt.notna().any()
    row = {
        "file": str(path),
        "level": level,
        "n_cells": len(df),
        "has_ground_truth": has_gt,
    }

    if unsupervised is None:
        present_markers = infer_marker_columns(df, marker_cols)
        unsupervised = compute_unsupervised_metrics(
            df, "predicted_phenotype", present_markers[:40], marker_rules=marker_rules,
        ).to_dict()
    row.update(unsupervised)

    if has_gt:
        m = compute_supervised_metrics(
            yt, yp, level=level,
            hierarchy_path=str(hierarchy_path) if hierarchy_path else None,
            rare_fraction=rare_fraction,
            common_fraction=common_fraction,
        )
        row.update(m.to_dict())

    return row


def _load_predictions_for_eval(
    path: Path,
    *,
    quant_path: Optional[Path],
    marker_cols: Optional[List[str]],
    marker_rules: Optional[dict],
) -> tuple[pd.DataFrame, dict]:
    df = normalize_columns(pd.read_csv(path))
    df = enrich_from_quant(df, quant_path, marker_cols=marker_cols)
    present_markers = infer_marker_columns(df, marker_cols)
    unsupervised = compute_unsupervised_metrics(
        df, "predicted_phenotype", present_markers[:40], marker_rules=marker_rules,
    ).to_dict()
    return df, unsupervised


def evaluate_dataset(
    dataset_name: str,
    results_root: Path,
    *,
    methods: Optional[List[str]] = None,
    levels: Optional[List[str]] = None,
    hierarchy_path: Optional[Path] = None,
    quant_path: Optional[Path] = None,
    marker_cols: Optional[List[str]] = None,
    marker_rules: Optional[dict] = None,
    rare_fraction: float = 0.01,
    common_fraction: float = 0.05,
    parallel_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """Evaluate all prediction files for one dataset; return per-fold results."""
    dataset_results = results_root / dataset_name
    if not dataset_results.is_dir():
        raise FileNotFoundError(f"No results directory: {dataset_results}")

    levels = levels or LEVELS
    workers = resolve_worker_count(parallel_jobs)
    work_items = [
        (method, path_level, fold_id, path, list(evaluation_levels_for_file(path, levels)))
        for method, path_level, fold_id, path in iter_method_predictions(dataset_results, methods)
        if evaluation_levels_for_file(path, levels)
    ]

    if not work_items:
        logger.warning("No prediction files found under %s", dataset_results)
        return pd.DataFrame()

    def _evaluate_item(item):
        method, path_level, fold_id, path, file_levels = item
        try:
            df, unsupervised = _load_predictions_for_eval(
                path,
                quant_path=quant_path,
                marker_cols=marker_cols,
                marker_rules=marker_rules,
            )
        except Exception as exc:
            logger.warning("Skipping %s: %s", path, exc)
            return []

        perf_dir = path.parent
        while perf_dir != dataset_results and not (perf_dir / "fold_times.txt").exists():
            if perf_dir.parent == perf_dir:
                break
            perf_dir = perf_dir.parent
        perf = find_performance_file(
            perf_dir if (perf_dir / "fold_times.txt").exists() else path.parent
        )

        item_rows = []
        for level in file_levels:
            try:
                row = evaluate_predictions_file(
                    path,
                    level=level,
                    hierarchy_path=hierarchy_path,
                    quant_path=quant_path,
                    marker_cols=marker_cols,
                    marker_rules=marker_rules,
                    df=df,
                    unsupervised=unsupervised,
                    rare_fraction=rare_fraction,
                    common_fraction=common_fraction,
                )
                row["dataset"] = dataset_name
                row["method"] = method
                row["kfold_strategy"] = _resolve_kfold_strategy(method)
                row["fold"] = fold_id
                row["path_level"] = path_level
                row.update(perf.to_dict())
                item_rows.append(row)
            except Exception as exc:
                logger.warning("Skipping %s [%s]: %s", path, level, exc)
        return item_rows

    rows: List[dict] = []
    if workers <= 1 or len(work_items) <= 1:
        for item in work_items:
            rows.extend(_evaluate_item(item))
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(work_items))) as pool:
            futures = [pool.submit(_evaluate_item, item) for item in work_items]
            for future in as_completed(futures):
                rows.extend(future.result())

    if not rows:
        logger.warning("No prediction files evaluated under %s", dataset_results)
        return pd.DataFrame()

    return pd.DataFrame(rows)


def aggregate_results(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Mean/std aggregation across folds per method and level."""
    if per_fold.empty:
        return per_fold

    group_cols = ["dataset", "method", "level"]
    agg_cols = [
        c for c in METRIC_COLS + [
            "train_time_mean", "inference_time_mean", "peak_memory_mb", "total_time_sec",
        ]
        if c in per_fold.columns
    ]

    summary_rows = []
    for keys, grp in per_fold.groupby(group_cols):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        for col in agg_cols:
            row[f"{col}_mean"] = grp[col].mean()
            row[f"{col}_std"] = grp[col].std()
        row["n_folds"] = len(grp)
        if "weighted_f1" in grp.columns:
            row["stability"] = stability_score(grp["weighted_f1"].std())
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    if "stability" in summary.columns:
        summary["notebook_overall_score"] = summary.apply(notebook_overall_score, axis=1)

    if "accuracy_mean" in summary.columns and "macro_f1_mean" in summary.columns:
        summary["supervised_score"] = (
            summary.get("accuracy_mean", 0).fillna(0)
            + summary.get("macro_f1_mean", 0).fillna(0)
            + summary.get("weighted_f1_mean", 0).fillna(0)
        ) / 3
        summary["overall_score"] = summary["supervised_score"]

    unsup_parts = []
    for col in ["silhouette_mean", "marker_purity_mean", "neighborhood_consistency_mean"]:
        if col in summary.columns:
            unsup_parts.append(summary[col].fillna(0))
    if unsup_parts:
        summary["unsupervised_score"] = sum(unsup_parts) / len(unsup_parts)
        if "supervised_score" in summary.columns:
            has_sup = summary["supervised_score"].notna() & (summary["supervised_score"] > 0)
            summary.loc[has_sup, "overall_score"] = (
                summary.loc[has_sup, "supervised_score"].fillna(0) * 0.6
                + summary.loc[has_sup, "unsupervised_score"].fillna(0) * 0.4
            )
            summary.loc[~has_sup, "overall_score"] = summary.loc[~has_sup, "unsupervised_score"]
        else:
            summary["overall_score"] = summary["unsupervised_score"]

    return summary.sort_values(["dataset", "method", "level"]).reset_index(drop=True)

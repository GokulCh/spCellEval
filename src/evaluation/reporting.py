"""
reporting.py
============
Data aggregation helpers for notebook-style benchmark reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from prediction_io import (
    iter_method_predictions,
    normalize_columns,
    resolve_labels_for_level,
)
from quant_merge import enrich_from_quant


def stability_score(std: float, threshold: float = 0.1) -> float:
    """Notebook stability metric: ``1 - std/threshold``, clipped at 0."""
    if std is None or np.isnan(std):
        return 1.0
    return max(0.0, 1.0 - (std / threshold))


def notebook_overall_score(row: pd.Series) -> float:
    """Composite score used in ``eval_levels.ipynb`` (mean of 10 metrics)."""
    cols = [
        "weighted_f1_mean",
        "accuracy_mean",
        "macro_f1_mean",
        "mcc_mean",
        "kappa_mean",
        "r2_composition_mean",
        "pearson_composition_mean",
        "ari_mean",
        "nmi_mean",
        "stability",
    ]
    values = [row[c] for c in cols if c in row.index and pd.notna(row[c])]
    return float(np.mean(values)) if values else np.nan


def phenotype_counts(series: pd.Series) -> pd.Series:
    """Return sorted phenotype counts (drop NA)."""
    return series.dropna().astype(str).value_counts().sort_values(ascending=False)


def composition_table(series: pd.Series, label_name: str) -> pd.DataFrame:
    """Counts and percentages for one phenotype column."""
    counts = phenotype_counts(series)
    total = counts.sum()
    return pd.DataFrame({
        "label": label_name,
        "phenotype": counts.index,
        "count": counts.values,
        "fraction": (counts.values / total) if total else 0.0,
    })


def export_ground_truth_composition(
    quant_path: Path,
    output_dir: Path,
    *,
    label_col: str = "cell_type",
) -> Optional[Path]:
    """Write ground-truth cell-type composition from the quantification CSV."""
    if not quant_path.is_file():
        return None
    quant = pd.read_csv(quant_path)
    if label_col not in quant.columns:
        return None

    tables = [composition_table(quant[label_col], "ground_truth")]
    if "Image_ID" in quant.columns:
        for image_id, grp in quant.groupby("Image_ID"):
            tables.append(composition_table(grp[label_col], f"ground_truth:{image_id}"))

    out = pd.concat(tables, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "composition_ground_truth.csv"
    out.to_csv(path, index=False)
    return path


def collect_method_predictions(
    dataset_results: Path,
    method: str,
    *,
    path_level: str = "level3",
    quant_path: Optional[Path] = None,
    marker_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Concatenate all fold prediction CSVs for one method."""
    frames: List[pd.DataFrame] = []
    for m, lvl, _fold, path in iter_method_predictions(dataset_results, methods=[method]):
        if m != method or lvl != path_level:
            continue
        df = normalize_columns(pd.read_csv(path))
        df = enrich_from_quant(df, quant_path, marker_cols=marker_cols)
        df["source_file"] = path.name
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def export_method_composition(
    preds: pd.DataFrame,
    method: str,
    output_dir: Path,
) -> Optional[Path]:
    """Export predicted phenotype composition for one method."""
    if preds.empty or "predicted_phenotype" not in preds.columns:
        return None

    tables = [composition_table(preds["predicted_phenotype"], method)]
    if "Image_ID" in preds.columns:
        for image_id, grp in preds.groupby("Image_ID"):
            tables.append(composition_table(grp["predicted_phenotype"], f"{method}:{image_id}"))

    out = pd.concat(tables, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"composition_{method}.csv"
    out.to_csv(path, index=False)
    return path


def build_confusion_dataframe(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    max_classes: int = 25,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build a labeled confusion matrix, optionally truncating rare classes."""
    yt = y_true.dropna().astype(str)
    yp = y_pred.dropna().astype(str)
    mask = yt.index.intersection(yp.index)
    yt = yt.loc[mask]
    yp = yp.loc[mask]

    if yt.empty:
        return pd.DataFrame(), []

    freq = yt.value_counts()
    labels = list(freq.head(max_classes).index)
    pred_freq = yp.value_counts()
    for label in pred_freq.head(max_classes).index:
        if label not in labels:
            labels.append(label)

    cm = confusion_matrix(yt, yp, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "true_phenotype"
    cm_df.columns.name = "predicted_phenotype"
    return cm_df, labels


def export_confusion_matrix_csv(
    preds: pd.DataFrame,
    method: str,
    output_dir: Path,
) -> Optional[Path]:
    """Write confusion matrix CSV for a method when ground truth is present."""
    if preds.empty or "true_phenotype" not in preds.columns:
        return None
    cm_df, _ = build_confusion_dataframe(preds["true_phenotype"], preds["predicted_phenotype"])
    if cm_df.empty:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"confusion_{method}.csv"
    cm_df.to_csv(path)
    return path


def top_methods_by_score(
    summary: pd.DataFrame,
    *,
    level: str = "level3",
    score_col: str = "overall_score",
    top_n: int = 5,
) -> List[str]:
    """Return top-N method IDs for a granularity level."""
    df = summary[summary["level"] == level].copy()
    if df.empty:
        return []
    if score_col not in df.columns:
        score_col = "notebook_overall_score" if "notebook_overall_score" in df.columns else "overall_score"
    if score_col not in df.columns:
        return sorted(df["method"].unique())[:top_n]
    return (
        df.sort_values(score_col, ascending=False, na_position="last")["method"]
        .head(top_n)
        .tolist()
    )


def export_all_report_tables(
    dataset: str,
    results_root: Path,
    *,
    summary: pd.DataFrame,
    quant_path: Optional[Path] = None,
    marker_cols: Optional[List[str]] = None,
    level: str = "level3",
    top_n_methods: int = 10,
) -> Dict[str, Path]:
    """Export composition and confusion tables for ground truth and top methods."""
    dataset_results = results_root / dataset
    tables_dir = results_root / dataset / "summary" / "tables"
    written: Dict[str, Path] = {}

    if quant_path:
        gt = export_ground_truth_composition(quant_path, tables_dir)
        if gt:
            written["composition_ground_truth"] = gt

    methods = top_methods_by_score(summary, level=level, top_n=top_n_methods)
    for method in methods:
        preds = collect_method_predictions(
            dataset_results,
            method,
            path_level=level,
            quant_path=quant_path,
            marker_cols=marker_cols,
        )
        comp = export_method_composition(preds, method, tables_dir)
        if comp:
            written[f"composition_{method}"] = comp
        cm = export_confusion_matrix_csv(preds, method, tables_dir)
        if cm:
            written[f"confusion_{method}"] = cm

    return written


def export_cell_type_representation_table(
    quant_path: Path,
    output_dir: Path,
    *,
    label_col: str = "cell_type",
    rare_fraction: float = 0.01,
    common_fraction: float = 0.05,
) -> Optional[Path]:
    """Export ranked cell-type abundance with rare/common tiers."""
    if not quant_path.is_file():
        return None

    _METHODS_UTILS = Path(__file__).resolve().parents[1] / "methods" / "utils"
    import sys
    if str(_METHODS_UTILS) not in sys.path:
        sys.path.insert(0, str(_METHODS_UTILS))
    from kfold_strategies import build_cell_type_representation  # noqa: WPS433
    from sklearn.preprocessing import LabelEncoder  # noqa: WPS433

    quant = pd.read_csv(quant_path)
    if label_col not in quant.columns:
        return None

    encoder = LabelEncoder()
    y = encoder.fit_transform(quant[label_col].astype(str))
    labels = pd.DataFrame({
        "label": range(len(encoder.classes_)),
        "phenotype": encoder.classes_,
    })
    repr_df = build_cell_type_representation(
        labels,
        y,
        rare_fraction=rare_fraction,
        common_fraction=common_fraction,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "cell_type_representation.csv"
    repr_df.to_csv(path, index=False)
    return path


def export_per_class_metrics_tables(
    dataset_results: Path,
    output_dir: Path,
    *,
    methods: Optional[List[str]] = None,
    level: str = "level3",
    quant_path: Optional[Path] = None,
    marker_cols: Optional[List[str]] = None,
    rare_fraction: float = 0.01,
    common_fraction: float = 0.05,
) -> Dict[str, Path]:
    """Export per-phenotype recall/F1 tables for each evaluated method."""
    from metrics import per_class_metrics_table  # noqa: WPS433

    written: Dict[str, Path] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for method, _lvl, _fold, path in iter_method_predictions(dataset_results, methods):
        preds = normalize_columns(pd.read_csv(path))
        preds = enrich_from_quant(preds, quant_path, marker_cols=marker_cols)
        yt, yp = resolve_labels_for_level(preds, level)
        if yt is None or yp is None:
            continue
        table = per_class_metrics_table(
            yt, yp, rare_fraction=rare_fraction, common_fraction=common_fraction,
        )
        if table.empty:
            continue
        table.insert(0, "method", method)
        table.insert(1, "fold_file", path.name)
        out_path = output_dir / f"per_class_metrics_{method}.csv"
        if out_path.is_file():
            table = pd.concat([pd.read_csv(out_path), table], ignore_index=True)
        table.to_csv(out_path, index=False)
        written[method] = out_path

    return written


def export_rare_type_benchmark_summary(
    per_fold: pd.DataFrame,
    output_dir: Path,
    *,
    level: str = "level3",
) -> Optional[Path]:
    """Aggregate rare-type benchmark metrics across methods."""
    cols = [
        "method", "fold", "rare_macro_f1", "common_macro_f1",
        "min_class_recall", "rare_min_recall", "n_rare_types", "n_common_types",
        "most_common_type", "rarest_type",
    ]
    present = [c for c in cols if c in per_fold.columns]
    if "method" not in present:
        return None

    df = per_fold[per_fold["level"] == level][present].copy()
    if df.empty:
        return None

    numeric = [c for c in present if c not in {"method", "fold", "most_common_type", "rarest_type"}]
    agg = df.groupby("method", as_index=False)[numeric].mean(numeric_only=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "rare_type_benchmark_summary.csv"
    agg.to_csv(path, index=False)
    return path

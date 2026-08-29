"""
quant_merge.py
==============
Merge marker expression and spatial coordinates from the quantification CSV
into prediction files for unsupervised metric computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set

import pandas as pd

_NON_MARKER_COLS: Set[str] = {
    "Cell_ID", "Image_ID", "Patient_ID", "x", "y", "batch_id",
    "cell_type", "level_1_cell_type", "level_2_cell_type", "cell_labels",
    "true_phenotype", "predicted_phenotype", "predictions", "prediction",
    "scyan_pop", "confidence", "fold", "image_id", "cell_id",
    "sample_id", "Pos_X", "Pos_Y", "cell_label_idx",
}


def infer_marker_columns(df: pd.DataFrame, known_markers: Optional[List[str]] = None) -> List[str]:
    """Return marker column names present in *df*."""
    if known_markers:
        return [c for c in known_markers if c in df.columns]
    return [
        c for c in df.columns
        if c not in _NON_MARKER_COLS
        and not c.startswith("prob_")
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def _resolve_id_columns(left: pd.DataFrame, right: pd.DataFrame) -> tuple[str, str]:
    for lid, rid in [
        ("Cell_ID", "Cell_ID"),
        ("cell_id", "cell_id"),
        ("Cell_ID", "cell_id"),
        ("cell_id", "Cell_ID"),
    ]:
        if lid in left.columns and rid in right.columns:
            return lid, rid
    for col in ["Cell_ID", "cell_id"]:
        if col in left.columns:
            return col, col
    raise ValueError("Could not resolve cell ID column for quant merge.")


def enrich_from_quant(
    df: pd.DataFrame,
    quant_path: Optional[Path],
    marker_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Attach spatial coords and marker expression from the quant CSV when missing."""
    if quant_path is None or not quant_path.is_file():
        return df

    marker_cols = marker_cols or []
    spatial_cols = [c for c in ["x", "y", "Image_ID", "batch_id"] if c not in df.columns]
    marker_missing = [c for c in marker_cols if c not in df.columns]
    if not spatial_cols and not marker_missing:
        return df

    lid, _ = _resolve_id_columns(df, df)
    id_col = lid
    id_values = df[id_col].dropna().unique()
    usecols = set(spatial_cols) | set(marker_missing) | {id_col, "Cell_ID", "cell_id"}

    quant = pd.read_csv(quant_path, usecols=lambda c: c in usecols)
    if quant.empty:
        return df

    rid_col = "Cell_ID" if "Cell_ID" in quant.columns else "cell_id"
    if rid_col in quant.columns and len(id_values) < len(quant):
        quant = quant[quant[rid_col].isin(id_values)]

    lid, rid = _resolve_id_columns(df, quant)
    merge_cols = [c for c in quant.columns if c != rid]
    merged = df.merge(
        quant[[rid, *merge_cols]],
        left_on=lid,
        right_on=rid,
        how="left",
        suffixes=("", "_quant"),
    )
    if rid != lid and rid in merged.columns:
        merged = merged.drop(columns=[rid])
    return merged

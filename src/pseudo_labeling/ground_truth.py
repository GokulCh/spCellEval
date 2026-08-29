"""
ground_truth.py
===============
Utilities for separating ground-truth labels from feature matrices so that
annotation methods cannot leak label information during training or inference.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

_CYC_SUFFIX_RE = re.compile(r":Cyc_\d+_ch_\d+$", re.IGNORECASE)


def _clean_marker_name(raw: str) -> str:
    return _CYC_SUFFIX_RE.sub("", raw).strip()

# All columns that encode cell-type identity at any hierarchy level.
LABEL_COLUMNS: List[str] = [
    "cell_type",
    "level_1_cell_type",
    "level_2_cell_type",
    "cell_labels",
]

# Standardised spatial / identity metadata (never used as protein features).
STANDARD_METADATA_COLUMNS: List[str] = [
    "Cell_ID",
    "Image_ID",
    "Patient_ID",
    "x",
    "y",
    "batch_id",
]

# Columns commonly dropped by supervised method runners (MAPS, etc.).
DEFAULT_EVAL_DROP_COLUMNS: List[str] = [
    *STANDARD_METADATA_COLUMNS,
    *LABEL_COLUMNS,
    "csv",
    "orig.ident",
    "sample_name",
    "tissue",
    "donor",
    "unique_region",
    "File Name",
    "width_px",
    "height_px",
    "eccentricity",
    "major_axis_length",
    "image",
    "sample_id",
    "BatchId",
    "Batch",
]


def _unique_preserve_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extract_ground_truth(
    df: pd.DataFrame,
    phenotype_column: str = "cell_type",
    id_column: str = "Cell_ID",
) -> pd.DataFrame:
    """Return a sidecar DataFrame with identity + all available label columns."""
    cols = [c for c in [id_column, *LABEL_COLUMNS, phenotype_column] if c in df.columns]
    return df[_unique_preserve_order(cols)].copy()


def strip_ground_truth(
    df: pd.DataFrame,
    phenotype_column: str = "cell_type",
    extra_drop: Optional[List[str]] = None,
    drop_metadata: bool = False,
    keep_id: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove label columns from *df* and return ``(features, labels_sidecar)``.

    Parameters
    ----------
    df:
        Input quantification table (processed CSV schema).
    phenotype_column:
        Primary target column used for evaluation.
    extra_drop:
        Additional columns to remove from the feature matrix.
    drop_metadata:
        When ``True``, also remove standard metadata columns so only markers
        (and optionally ``Cell_ID``) remain.
    keep_id:
        Keep ``Cell_ID`` in the feature matrix when ``drop_metadata`` is set.
    """
    labels = extract_ground_truth(df, phenotype_column=phenotype_column)

    to_drop = [c for c in LABEL_COLUMNS if c in df.columns]
    if phenotype_column not in LABEL_COLUMNS and phenotype_column in df.columns:
        to_drop.append(phenotype_column)
    if drop_metadata:
        meta = [c for c in DEFAULT_EVAL_DROP_COLUMNS if c in df.columns]
        if keep_id and "Cell_ID" in meta:
            meta.remove("Cell_ID")
        to_drop.extend(meta)
    if extra_drop:
        to_drop.extend(extra_drop)

    to_drop = _unique_preserve_order([c for c in to_drop if c in df.columns])
    features = df.drop(columns=to_drop).copy()
    return features, labels


def get_marker_columns(config: Dict[str, Any]) -> List[str]:
    """Return cleaned marker column names from a dataset YAML config."""
    raw_markers: List[str] = config.get("protein_markers", [])
    return [_clean_marker_name(m) for m in raw_markers]


def infer_separate_col(config: Dict[str, Any]) -> str:
    """First metadata column after markers in the TACIT-compatible table.

    After ``prepare_tacit_input`` reorders columns to ``[cell_id, markers…,
    Image_ID, …]``, this is the column where metadata begins.
    """
    _ = config  # reserved for dataset-specific overrides via pseudo_labeling YAML
    return "Image_ID"


def load_dataset_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a dataset YAML config from disk."""
    with Path(config_path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def prepare_tacit_input(
    df: pd.DataFrame,
    config: Dict[str, Any],
    *,
    pseudo_label_column: Optional[str] = None,
) -> pd.DataFrame:
    """Build a TACIT-compatible CSV (``cell_id`` first, ``cell_type`` present).

    When ground truth is absent, *pseudo_label_column* (e.g.
    ``predicted_phenotype``) is copied into ``cell_type`` so TACIT can still
    write evaluation-ready output with a ``true_phenotype`` column.
    """
    out = df.copy()
    if "Cell_ID" in out.columns:
        out = out.rename(columns={"Cell_ID": "cell_id"})
    elif "cell_id" not in out.columns:
        raise ValueError("Input DataFrame must contain 'Cell_ID' or 'cell_id'.")

    if "cell_type" not in out.columns:
        if pseudo_label_column and pseudo_label_column in out.columns:
            out["cell_type"] = out[pseudo_label_column]
        else:
            out["cell_type"] = "undefined"

    # Move cell_id to the front (matches dplyr select(cell_id, everything())).
    cols = ["cell_id"] + [c for c in out.columns if c != "cell_id"]
    return out[cols]

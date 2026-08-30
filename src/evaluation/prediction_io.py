"""
prediction_io.py
================
Discover and normalise prediction CSV files from Stage 3 benchmark outputs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import pandas as pd

from hierarchy import LEVEL_COLUMN, build_level_mappings

_UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(_UTILS) not in sys.path:
    sys.path.insert(0, str(_UTILS))

from prediction_files import is_spatial_smoothed_file  # noqa: E402

_PRED_FILE_RE = re.compile(r"predictions(?:_fold)?[_]?(\d+)\.csv$", re.IGNORECASE)

TRUE_COL_CANDIDATES = ["true_phenotype", "cell_type", "true_label", "label_name"]
PRED_COL_CANDIDATES = ["predicted_phenotype", "predictions", "prediction", "scyan_pop"]


def discover_prediction_files(
    results_dir: Path,
    *,
    include_spatial: bool = False,
) -> List[Path]:
    """Recursively find prediction CSVs under *results_dir*.

    By default, spatial-smoothed outputs (``*_spatial.csv``) are excluded so
    supervised metrics are not averaged with their raw counterparts.
    """
    if not results_dir.is_dir():
        return []
    files = sorted(results_dir.rglob("predictions*.csv"))
    out = [f for f in files if f.name.startswith("predictions")]
    if not include_spatial:
        out = [f for f in out if not is_spatial_smoothed_file(f)]
    return out


def extract_fold_id(path: Path) -> str:
    m = _PRED_FILE_RE.search(path.name)
    return m.group(1) if m else path.stem


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise true/predicted column names."""
    out = df.copy()
    if "true_phenotype" not in out.columns:
        for col in TRUE_COL_CANDIDATES:
            if col in out.columns:
                out = out.rename(columns={col: "true_phenotype"})
                break
    if "predicted_phenotype" not in out.columns:
        for col in PRED_COL_CANDIDATES:
            if col in out.columns:
                out = out.rename(columns={col: "predicted_phenotype"})
                break
    return out


def resolve_labels_for_level(df: pd.DataFrame, level: str) -> Tuple[pd.Series, pd.Series]:
    """Return (y_true, y_pred) for the requested granularity level."""
    df = normalize_columns(df)
    mappings = build_level_mappings()
    col = LEVEL_COLUMN.get(level, "cell_type")

    if level == "level3":
        yt = df["true_phenotype"] if "true_phenotype" in df.columns else df.get("cell_type")
        yp = df["predicted_phenotype"]
        return yt, yp

    # Coarser levels: map via hierarchy
    def _map_level(label: str) -> str:
        info = mappings.get(str(label), {})
        return info.get(col, str(label))

    if col in df.columns:
        yt = df[col]
    else:
        yt = df["true_phenotype"].map(_map_level)

    yp = df["predicted_phenotype"].map(_map_level)
    return yt, yp


def infer_method_and_level(path: Path, dataset_results: Path) -> Tuple[str, str]:
    """Infer method name and level from path like ``results/IMMUcan/random_forest/level3/...``."""
    rel = path.relative_to(dataset_results)
    parts = rel.parts
    method = parts[0] if parts else "unknown"
    level = "level3"
    for p in parts:
        if p.startswith("level"):
            level = p
    return method, level


def iter_method_predictions(
    dataset_results: Path,
    methods: Optional[List[str]] = None,
    *,
    include_spatial: bool = False,
) -> Iterator[Tuple[str, str, str, Path]]:
    """Yield ``(method, level, fold_id, path)`` for each prediction file."""
    for path in discover_prediction_files(dataset_results, include_spatial=include_spatial):
        method, level = infer_method_and_level(path, dataset_results)
        if methods and method not in methods:
            continue
        yield method, level, extract_fold_id(path), path

"""
prediction_files.py
===================
Shared helpers for discovering prediction CSVs and deciding spatial post-processing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_GRANULARITY_DIRS = frozenset({"level1", "level2", "level3"})
RESULTS_NON_METHOD_DIRS = frozenset({"summary", "logs"})
_CV_FOLD_RE = re.compile(r"predictions_fold_\d+\.csv$", re.IGNORECASE)

FULL_TISSUE_COVERAGE_THRESHOLD = 0.9


def is_spatial_smoothed_file(path: Path) -> bool:
    """Return True for outputs written by spatial post-processing."""
    return path.name.endswith("_spatial.csv")


def is_cross_validation_filename(name: str) -> bool:
    """Return True for explicit k-fold test outputs (``predictions_fold_N.csv``)."""
    return bool(_CV_FOLD_RE.match(name))


def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def prediction_coverage_ratio(pred_path: Path, quant_path: Path) -> Optional[float]:
    """Fraction of quantification rows present in a prediction file."""
    if not pred_path.is_file() or not quant_path.is_file():
        return None
    n_pred = _count_csv_rows(pred_path)
    n_quant = _count_csv_rows(quant_path)
    if n_quant == 0:
        return None
    return n_pred / n_quant


def is_full_tissue_prediction(
    pred_path: Path,
    quant_path: Optional[Path],
    *,
    coverage_threshold: float = FULL_TISSUE_COVERAGE_THRESHOLD,
) -> bool:
    """Return True when predictions cover essentially the full quantification table."""
    if is_cross_validation_filename(pred_path.name):
        return False
    if quant_path is None or not quant_path.is_file():
        return True
    coverage = prediction_coverage_ratio(pred_path, quant_path)
    if coverage is None:
        return True
    return coverage >= coverage_threshold


def should_apply_spatial_smoothing(
    pred_path: Path,
    quant_path: Optional[Path],
) -> bool:
    """Spatial smoothing is only valid on dense, full-tissue prediction files."""
    if is_spatial_smoothed_file(pred_path):
        return False
    return is_full_tissue_prediction(pred_path, quant_path)


def prediction_output_granularity(path: Path) -> Optional[str]:
    """Return granularity dir when predictions are nested under a resolution folder.

    Standard supervised outputs live at ``{method}/level3/predictions_fold_N.csv``
    and should be evaluated at all hierarchy levels. Clustering methods write
    separate files under ``.../level3/{resolution}/level{1,2,3}/predictions_*.csv``.
    """
    parent = path.parent.name
    if parent not in _GRANULARITY_DIRS:
        return None
    grandparents = path.parent.parent
    if grandparents.parent is None:
        return None
    if grandparents.parent.name in _GRANULARITY_DIRS:
        return parent
    return None


def evaluation_levels_for_file(
    path: Path,
    all_levels: list[str],
) -> list[str]:
    """Levels to evaluate for one prediction file."""
    gran = prediction_output_granularity(path)
    if gran is not None:
        return [gran] if gran in all_levels else []
    return list(all_levels)

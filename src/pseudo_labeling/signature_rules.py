"""
signature_rules.py
==================
Pure-Python marker signature / threshold pseudo-labeler.

Reads a TACIT-style decision matrix (rows = cell types, columns = markers,
``1`` = required positive, ``-1`` = required negative, empty = don't care)
and assigns each cell the type with the highest rule-satisfaction score.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

UNDEFINED_LABEL = "undefined"


def load_decision_matrix(path: str | Path) -> pd.DataFrame:
    """Load a decision matrix CSV and normalise marker column names."""
    dm = pd.read_csv(path)
    if "cell_type" not in dm.columns:
        raise ValueError(f"Decision matrix must have a 'cell_type' column: {path}")
    dm = dm.set_index("cell_type")
    # Coerce rule values: 1 / -1 / 0 / NaN
    for col in dm.columns:
        dm[col] = pd.to_numeric(dm[col], errors="coerce")
    return dm


def _rule_mask(
    marker_values: np.ndarray,
    rules: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Boolean mask of which rules are satisfied for each cell (n_cells, n_rules)."""
    n_cells = marker_values.shape[0]
    satisfied = np.ones((n_cells, len(rules)), dtype=bool)

    for j, rule in enumerate(rules):
        if np.isnan(rule) or rule == 0:
            continue
        col_vals = marker_values[:, j]
        if rule > 0:
            satisfied[:, j] = col_vals >= threshold
        else:
            satisfied[:, j] = col_vals < threshold

    return satisfied


def apply_signature_rules(
    df: pd.DataFrame,
    decision_matrix: Union[pd.DataFrame, str, Path],
    marker_cols: Optional[List[str]] = None,
    threshold: float = 0.0,
    min_score: float = 1.0,
    undefined_label: str = UNDEFINED_LABEL,
) -> pd.Series:
    """Assign pseudo-labels via marker signature rules.

    Parameters
    ----------
    df:
        Quantification table with arcsinh-transformed marker columns.
    decision_matrix:
        TACIT-style matrix (path or DataFrame).
    marker_cols:
        Marker columns to use.  When ``None``, uses the intersection of
        ``df.columns`` and ``decision_matrix.columns``.
    threshold:
        Expression cutoff for positive/negative gating (post-arcsinh).
    min_score:
        Minimum fraction of applicable rules that must pass for assignment.
        Cells below this for all types receive *undefined_label*.
    undefined_label:
        Label for unassigned cells.

    Returns
    -------
    pd.Series
        Predicted cell types, indexed like *df*.
    """
    if isinstance(decision_matrix, (str, Path)):
        decision_matrix = load_decision_matrix(decision_matrix)

    dm = decision_matrix
    if marker_cols is None:
        marker_cols = [c for c in dm.columns if c in df.columns]
    else:
        marker_cols = [c for c in marker_cols if c in df.columns and c in dm.columns]

    missing_markers = [c for c in dm.columns if c not in df.columns]
    if missing_markers:
        logger.warning(
            "%d marker(s) in decision matrix not found in data (skipped): %s",
            len(missing_markers),
            missing_markers[:10],
        )

    if not marker_cols:
        raise ValueError("No overlapping marker columns between data and decision matrix.")

    marker_data = df[marker_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    cell_types = dm.index.tolist()
    n_cells = len(df)
    scores = np.zeros((n_cells, len(cell_types)))
    n_rules_applicable = np.zeros(len(cell_types))

    for i, ctype in enumerate(cell_types):
        rules = dm.loc[ctype, marker_cols].values.astype(float)
        applicable = ~np.isnan(rules) & (rules != 0)
        n_applicable = applicable.sum()
        n_rules_applicable[i] = n_applicable
        if n_applicable == 0:
            continue
        mask = _rule_mask(marker_data, rules, threshold)
        scores[:, i] = mask[:, applicable].sum(axis=1) / n_applicable

    # Break ties: higher score first, then more specific signatures (more rules).
    tie_breaker = n_rules_applicable / max(n_rules_applicable.max(), 1)
    combined = scores + tie_breaker * 1e-6
    best_idx = combined.argmax(axis=1)
    best_score = scores[np.arange(n_cells), best_idx]
    predictions = np.array(
        [cell_types[i] if best_score[j] >= min_score else undefined_label for j, i in enumerate(best_idx)]
    )

    n_undefined = (predictions == undefined_label).sum()
    logger.info(
        "Signature rules: %d cells labelled, %d → '%s'.",
        n_cells - n_undefined,
        n_undefined,
        undefined_label,
    )
    return pd.Series(predictions, index=df.index, name="predicted_phenotype")


def add_hierarchy_from_predictions(
    df: pd.DataFrame,
    prediction_col: str = "predicted_phenotype",
    hierarchy_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Derive coarse hierarchy columns from pseudo-labels."""
    import sys
    from pathlib import Path as P

    utils_dir = P(__file__).resolve().parents[1] / "preprocessing" / "utils"
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))

    from label_hierarchy import add_hierarchy_columns  # noqa: WPS433

    out = df.copy()
    out["cell_type"] = out[prediction_col]
    kwargs: Dict = {"cell_type_col": "cell_type", "inplace": True}
    if hierarchy_path:
        kwargs["hierarchy_path"] = hierarchy_path
    add_hierarchy_columns(out, **kwargs)
    return out

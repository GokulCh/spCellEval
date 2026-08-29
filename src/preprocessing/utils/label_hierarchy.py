"""
label_hierarchy.py
==================
Utilities for deriving hierarchical cell-type labels
(level_1_cell_type, level_2_cell_type) from the project-wide
``cell_type_hierarchy.txt`` indent tree.

Hierarchy file format (2-space indent per level)::

    Immune                     ← level-0  (root / level_1)
      Myeloid                  ← level-1  (subclass / level_2)
        Macrophage             ← level-2  (leaf / cell_type)
          M1_Macrophage        ← level-3+ (fine leaf / cell_type)

Column conventions
------------------
* ``cell_type``        – finest available label (level 2+ in the tree)
* ``level_2_cell_type``– immediate child of the root  (level-1 in the tree)
* ``level_1_cell_type``– root class                   (level-0 in the tree)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default path (resolved relative to *this* file so it works regardless of
# the working directory when the script is invoked).
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent   # src/preprocessing/utils/
_DEFAULT_HIERARCHY = (
    _HERE.parents[1] / "evaluation" / "cell_type_hierarchy.txt"
    # _HERE.parents[0] = src/preprocessing
    # _HERE.parents[1] = src
    # → src/evaluation/cell_type_hierarchy.txt
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_hierarchy(file_path: str | Path = _DEFAULT_HIERARCHY) -> Dict[str, Tuple[str, str]]:
    """Parse the indent-tree hierarchy file.

    Returns a dict mapping every *leaf* label to its ancestors::

        {
            "M1_Macrophage": ("Immune", "Myeloid"),
            "CD8+_T_cell":   ("Immune", "Lymphoid"),
            ...
        }

    The tuple is ``(level_1, level_2)`` i.e. (root, root's child).
    For nodes that sit at depth < 2 we still emit reasonable values
    so that every label in the tree has a mapping.

    Parameters
    ----------
    file_path:
        Path to ``cell_type_hierarchy.txt``.

    Returns
    -------
    dict[str, tuple[str, str]]
        ``{label: (level_1_cell_type, level_2_cell_type)}``
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Hierarchy file not found: {file_path}\n"
            "Expected at src/evaluation/cell_type_hierarchy.txt"
        )

    label_to_ancestors: Dict[str, Tuple[str, str]] = {}
    path_stack: list[str] = []

    with file_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.rstrip("\n").rstrip("\r")
            if not stripped.strip():
                continue  # skip blank lines

            indent = len(stripped) - len(stripped.lstrip(" "))
            depth = indent // 2          # 2 spaces per level
            label = stripped.strip()

            # Trim stack to current depth and push new node
            path_stack = path_stack[:depth]
            path_stack.append(label)

            # Resolve level_1 and level_2 for this node
            level_1 = path_stack[0] if len(path_stack) >= 1 else label
            level_2 = path_stack[1] if len(path_stack) >= 2 else level_1

            label_to_ancestors[label] = (level_1, level_2)

    logger.debug("Parsed %d labels from %s", len(label_to_ancestors), file_path)
    return label_to_ancestors


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def add_hierarchy_columns(
    df: pd.DataFrame,
    cell_type_col: str = "cell_type",
    hierarchy_path: str | Path = _DEFAULT_HIERARCHY,
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """Add ``level_1_cell_type`` and ``level_2_cell_type`` columns to *df*.

    Any label that is **not** found in the hierarchy file is assigned
    ``"undefined"`` at both coarser levels and a warning is emitted.

    Parameters
    ----------
    df:
        DataFrame that must contain *cell_type_col*.
    cell_type_col:
        Column name holding the finest-grained cell-type labels.
    hierarchy_path:
        Path to ``cell_type_hierarchy.txt``.
    inplace:
        If ``True`` mutate *df*; otherwise return a copy.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``level_1_cell_type`` and ``level_2_cell_type``
        columns appended (or updated in place).
    """
    if not inplace:
        df = df.copy()

    mapping = parse_hierarchy(hierarchy_path)

    # Vectorised lookup via map
    level_1_series = df[cell_type_col].map(
        {k: v[0] for k, v in mapping.items()}
    )
    level_2_series = df[cell_type_col].map(
        {k: v[1] for k, v in mapping.items()}
    )

    # Warn about unmapped labels
    unknown = df.loc[level_1_series.isna(), cell_type_col].unique()
    if len(unknown):
        logger.warning(
            "%d label(s) not found in hierarchy → mapped to 'undefined': %s",
            len(unknown),
            list(unknown),
        )

    df["level_1_cell_type"] = level_1_series.fillna("undefined")
    df["level_2_cell_type"] = level_2_series.fillna("undefined")

    return df


def get_level_mapping(
    hierarchy_path: str | Path = _DEFAULT_HIERARCHY,
) -> Dict[str, Dict[str, str]]:
    """Return a two-key dict of flat label → coarser-level mappings.

    Useful for quick look-ups without a DataFrame::

        mapping = get_level_mapping()
        mapping["level_1"]["CD8+_T_cell"]   # → "Immune"
        mapping["level_2"]["CD8+_T_cell"]   # → "Lymphoid"

    Returns
    -------
    dict with keys ``"level_1"`` and ``"level_2"``.
    """
    raw = parse_hierarchy(hierarchy_path)
    return {
        "level_1": {k: v[0] for k, v in raw.items()},
        "level_2": {k: v[1] for k, v in raw.items()},
    }


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_HIERARCHY
    mapping = get_level_mapping(path)
    print(json.dumps(mapping, indent=2))

"""
hierarchy.py
============
Cell-type hierarchy parsing for multi-level evaluation (level1/2/3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

_DEFAULT_HIERARCHY = Path(__file__).resolve().parent / "cell_type_hierarchy.txt"


def parse_ancestor_map(file_path: str | Path = _DEFAULT_HIERARCHY) -> Dict[str, List[str]]:
    """Parse indent-tree file → ``{label: [ancestor, …, label]}``."""
    file_path = Path(file_path)
    label_to_ancestors: Dict[str, List[str]] = {}
    path_stack: List[str] = []

    with file_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.rstrip("\n").rstrip("\r")
            if not stripped.strip():
                continue
            indent = len(stripped) - len(stripped.lstrip(" "))
            depth = indent // 2
            label = stripped.strip()
            path_stack = path_stack[:depth]
            path_stack.append(label)
            label_to_ancestors[label] = list(path_stack)

    return label_to_ancestors


def build_level_mappings(
    file_path: str | Path = _DEFAULT_HIERARCHY,
) -> Dict[str, Dict[str, str]]:
    """Map each label to ``level_1_cell_type`` and ``level_2_cell_type``."""
    ancestors = parse_ancestor_map(file_path)
    return {
        label: {
            "level_1_cell_type": path[0] if len(path) >= 1 else label,
            "level_2_cell_type": path[1] if len(path) >= 2 else (path[0] if path else label),
        }
        for label, path in ancestors.items()
    }


LEVEL_COLUMN = {
    "level1": "level_1_cell_type",
    "level2": "level_2_cell_type",
    "level3": "cell_type",
}

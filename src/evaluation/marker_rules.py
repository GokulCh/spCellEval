"""Load dataset-specific marker purity rules for evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "configs" / "marker_purity_rules.yaml"

# Fallback when no dataset-specific rules exist
IMMUCAN_FALLBACK: Dict[str, Dict[str, list]] = {
    "CD8+_T_cell": {"positive": ["CD3", "CD8a", "CD8"], "negative": ["CD20", "Ecad", "CD163"]},
    "CD4+_T_cell": {"positive": ["CD3", "CD4"], "negative": ["CD20", "Ecad"]},
    "B_cell": {"positive": ["CD20"], "negative": ["CD3", "Ecad"]},
    "Cancer": {"positive": ["Ecad", "PanCK"], "negative": ["CD3", "CD45"]},
}


def load_marker_purity_rules(
    dataset_name: str,
    rules_path: Optional[Path] = None,
) -> Dict[str, Dict[str, list]]:
    """Return marker purity rules for *dataset_name*."""
    path = rules_path or _DEFAULT_RULES_PATH
    if not path.is_file():
        return IMMUCAN_FALLBACK

    with path.open("r", encoding="utf-8") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh) or {}

    rules = cfg.get(dataset_name)
    if rules:
        return rules
    return IMMUCAN_FALLBACK

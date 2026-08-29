"""Stage 2 — automated pseudo-labeling and ground-truth management."""

from .ground_truth import (
    DEFAULT_EVAL_DROP_COLUMNS,
    LABEL_COLUMNS,
    extract_ground_truth,
    strip_ground_truth,
)
from .signature_rules import apply_signature_rules, load_decision_matrix

__all__ = [
    "DEFAULT_EVAL_DROP_COLUMNS",
    "LABEL_COLUMNS",
    "apply_signature_rules",
    "extract_ground_truth",
    "load_decision_matrix",
    "strip_ground_truth",
]

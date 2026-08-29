"""Stage 4 — automated evaluation framework."""

from .evaluator import evaluate_dataset, evaluate_predictions_file
from .metrics import compute_supervised_metrics, compute_unsupervised_metrics

__all__ = [
    "compute_supervised_metrics",
    "compute_unsupervised_metrics",
    "evaluate_dataset",
    "evaluate_predictions_file",
]

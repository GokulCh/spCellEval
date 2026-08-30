"""Tests for Stage 2 ground-truth management."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "pseudo_labeling"))

from ground_truth import LABEL_COLUMNS, strip_ground_truth  # noqa: E402


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "CD3": [1.0, 0.1],
        "CD8a": [0.5, 0.2],
        "Cell_ID": [1, 2],
        "x": [10.0, 20.0],
        "y": [30.0, 40.0],
        "cell_labels": ["CD8", "Tumor"],
        "cell_type": ["CD8+_T_cell", "Cancer"],
        "level_1_cell_type": ["Immune", "Cancer"],
        "level_2_cell_type": ["Lymphoid", "Cancer"],
    })


def test_strip_ground_truth_removes_all_label_columns(sample_df):
    features, labels = strip_ground_truth(sample_df)
    for col in LABEL_COLUMNS:
        assert col not in features.columns
    assert "cell_type" not in features.columns
    assert "CD3" in features.columns
    assert "cell_type" in labels.columns
    assert len(labels) == 2


def test_strip_ground_truth_drop_metadata(sample_df):
    features, _ = strip_ground_truth(sample_df, drop_metadata=True)
    assert "x" not in features.columns
    assert "y" not in features.columns
    assert "Cell_ID" in features.columns

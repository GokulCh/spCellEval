"""Tests for Stage 2 signature-rule pseudo-labeling."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "pseudo_labeling"))

from signature_rules import apply_signature_rules, load_decision_matrix  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
IMMUCAN_DM = _REPO / "configs/artifacts/IMMUcan/tacit_decision_matrix_level3.csv"


@pytest.fixture
def decision_matrix():
    return load_decision_matrix(IMMUCAN_DM)


def test_load_decision_matrix_has_cell_types(decision_matrix):
    assert "Cancer" in decision_matrix.index
    assert "CD8+_T_cell" in decision_matrix.index


def test_apply_signature_rules_basic(decision_matrix):
    # Synthetic cell strongly expressing CD8+ T cell markers
    df = pd.DataFrame({
        "Ecad": [0.0],
        "CD3": [5.0],
        "CD4": [0.0],
        "CD8a": [5.0],
        "FOXP3": [0.0],
        "CD7": [0.0],
        "CD38": [0.0],
        "CD20": [0.0],
        "CD15": [0.0],
        "MPO": [0.0],
        "CD163": [0.0],
        "CD68": [0.0],
        "CD206": [0.0],
        "CD11c": [0.0],
        "HLADR": [0.0],
        "CD303": [0.0],
        "SMA": [0.0],
        "PDGFRb": [0.0],
    })
    preds = apply_signature_rules(df, decision_matrix, threshold=1.0, min_score=1.0)
    assert preds.iloc[0] == "CD8+_T_cell"


def test_apply_signature_rules_cancer(decision_matrix):
    df = pd.DataFrame({
        "Ecad": [5.0],
        "SMA": [0.0],
        "CD3": [0.0], "CD4": [0.0], "CD8a": [0.0], "FOXP3": [0.0],
        "CD7": [0.0], "CD38": [0.0], "CD20": [0.0], "CD15": [0.0],
        "MPO": [0.0], "CD163": [0.0], "CD68": [0.0], "CD206": [0.0],
        "CD11c": [0.0], "HLADR": [0.0], "CD303": [0.0], "PDGFRb": [0.0],
    })
    preds = apply_signature_rules(df, decision_matrix, threshold=1.0, min_score=1.0)
    assert preds.iloc[0] == "Cancer"

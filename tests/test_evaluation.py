"""Tests for Stage 4 evaluation metrics."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "evaluation"))

from metrics import (  # noqa: E402
    compute_supervised_metrics,
    compute_unsupervised_metrics,
    hierarchical_f1_score,
    neighborhood_consistency_score,
)
from hierarchy import parse_ancestor_map, build_level_mappings  # noqa: E402
from prediction_io import normalize_columns, resolve_labels_for_level  # noqa: E402


def test_hierarchical_f1_perfect_match():
    anc = {"CD8+_T_cell": ["Immune", "Lymphoid", "T_cell", "CD8+_T_cell"]}
    score = hierarchical_f1_score(
        ["CD8+_T_cell", "CD8+_T_cell"],
        ["CD8+_T_cell", "CD8+_T_cell"],
        anc,
    )
    assert score == pytest.approx(1.0)


def test_compute_supervised_metrics():
    yt = pd.Series(["A", "A", "B", "B"])
    yp = pd.Series(["A", "B", "B", "B"])
    m = compute_supervised_metrics(yt, yp, level="level1")
    assert m.accuracy == 0.75
    assert m.macro_f1 > 0
    assert m.ari >= 0


def test_normalize_columns():
    df = pd.DataFrame({"cell_type": ["A"], "predicted_phenotype": ["A"]})
    out = normalize_columns(df.rename(columns={"cell_type": "true_phenotype"}))
    assert "true_phenotype" in out.columns


def test_level_mappings_exist():
    mappings = build_level_mappings()
    assert "CD8+_T_cell" in mappings
    assert mappings["CD8+_T_cell"]["level_1_cell_type"] == "Immune"


def test_unsupervised_metrics_extended():
    df = pd.DataFrame({
        "x": [0.0, 1.0, 2.0, 3.0],
        "y": [0.0, 0.0, 1.0, 1.0],
        "predicted_phenotype": ["A", "A", "B", "B"],
        "CD3": [0.9, 0.8, 0.1, 0.2],
        "CD20": [0.1, 0.2, 0.9, 0.8],
    })
    m = compute_unsupervised_metrics(df, "predicted_phenotype", ["CD3", "CD20"])
    assert m.n_cells == 4
    assert m.neighborhood_consistency is not None
    assert m.neighborhood_consistency >= 0


def test_evaluate_predictions_file_dual_metrics(tmp_path):
    pred = tmp_path / "predictions_0.csv"
    pd.DataFrame({
        "Cell_ID": [1, 2, 3, 4],
        "true_phenotype": ["A", "A", "B", "B"],
        "predicted_phenotype": ["A", "B", "B", "B"],
        "x": [0.0, 0.1, 10.0, 10.1],
        "y": [0.0, 0.1, 0.0, 0.1],
        "CD3": [0.9, 0.8, 0.1, 0.2],
        "CD20": [0.1, 0.2, 0.9, 0.8],
    }).to_csv(pred, index=False)

    from evaluator import evaluate_predictions_file  # noqa: E402

    row = evaluate_predictions_file(pred, level="level3", marker_cols=["CD3", "CD20"])
    assert row["accuracy"] == 0.75
    assert row["silhouette"] is not None or row["neighborhood_consistency"] is not None
    assert "marker_purity" in row


def test_neighborhood_consistency_high_for_homogeneous_clusters():
    df = pd.DataFrame({
        "x": [0.0, 0.1, 0.2, 10.0, 10.1, 10.2],
        "y": [0.0, 0.1, 0.2, 0.0, 0.1, 0.2],
        "predicted_phenotype": ["A", "A", "A", "B", "B", "B"],
    })
    score = neighborhood_consistency_score(df, "predicted_phenotype", k=2)
    assert score >= 0.8


def test_aggregate_results_adds_stability(tmp_path):
    from evaluator import aggregate_results  # noqa: E402

    per_fold = pd.DataFrame({
        "dataset": ["D", "D"],
        "method": ["m1", "m1"],
        "level": ["level3", "level3"],
        "weighted_f1": [0.8, 0.9],
        "accuracy": [0.7, 0.8],
        "macro_f1": [0.6, 0.7],
    })
    summary = aggregate_results(per_fold)
    assert "stability" in summary.columns
    assert "notebook_overall_score" in summary.columns

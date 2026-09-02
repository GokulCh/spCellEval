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


def test_rare_type_metrics():
    from metrics import compute_rare_type_metrics, per_class_metrics_table  # noqa: E402

    yt = pd.Series(["Common"] * 97 + ["Rare"] * 3)
    yp = pd.Series(["Common"] * 95 + ["Rare"] * 2 + ["Common"] * 2 + ["Rare"] * 1)
    rare = compute_rare_type_metrics(yt, yp, rare_fraction=0.05, common_fraction=0.10)
    assert rare["n_rare_types"] == 1
    assert rare["n_common_types"] == 1
    assert rare["most_common_type"] == "Common"
    assert rare["rarest_type"] == "Rare"
    assert rare["rare_macro_f1"] is not None

    table = per_class_metrics_table(yt, yp, rare_fraction=0.05, common_fraction=0.10)
    assert set(table["frequency_tier"]) <= {"rare", "common", "intermediate"}
    assert len(table) == 2


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


def test_discover_prediction_files_excludes_spatial_by_default(tmp_path):
    from prediction_io import discover_prediction_files  # noqa: E402

    (tmp_path / "predictions_1.csv").write_text("a\n", encoding="utf-8")
    (tmp_path / "predictions_1_spatial.csv").write_text("a\n", encoding="utf-8")

    raw_only = discover_prediction_files(tmp_path)
    assert [p.name for p in raw_only] == ["predictions_1.csv"]

    both = discover_prediction_files(tmp_path, include_spatial=True)
    assert {p.name for p in both} == {"predictions_1.csv", "predictions_1_spatial.csv"}


def test_should_apply_spatial_smoothing_skips_cv_subsets(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "utils"))
    from prediction_files import should_apply_spatial_smoothing  # noqa: E402

    quant = tmp_path / "quant.csv"
    pd.DataFrame({"Cell_ID": range(10)}).to_csv(quant, index=False)

    full_pred = tmp_path / "predictions_1.csv"
    pd.DataFrame({"Cell_ID": range(10)}).to_csv(full_pred, index=False)

    fold_pred = tmp_path / "predictions_fold_1.csv"
    pd.DataFrame({"Cell_ID": range(2)}).to_csv(fold_pred, index=False)

    sparse_pred = tmp_path / "predictions_2.csv"
    pd.DataFrame({"Cell_ID": range(2)}).to_csv(sparse_pred, index=False)

    assert should_apply_spatial_smoothing(full_pred, quant) is True
    assert should_apply_spatial_smoothing(fold_pred, quant) is False
    assert should_apply_spatial_smoothing(sparse_pred, quant) is False


def test_evaluation_levels_for_nested_clustering_outputs(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "utils"))
    from prediction_files import evaluation_levels_for_file  # noqa: E402

    rf_fold = tmp_path / "random_forest" / "level3" / "predictions_fold_1.csv"
    rf_fold.parent.mkdir(parents=True)
    rf_fold.touch()

    leiden_l1 = tmp_path / "leiden" / "level3" / "res1" / "level1" / "predictions_1.csv"
    leiden_l1.parent.mkdir(parents=True)
    leiden_l1.touch()

    all_levels = ["level1", "level2", "level3"]
    assert evaluation_levels_for_file(rf_fold, all_levels) == all_levels
    assert evaluation_levels_for_file(leiden_l1, all_levels) == ["level1"]

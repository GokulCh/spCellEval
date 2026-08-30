"""Integration tests for signature → evaluation → spatial post-processing."""

import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "e2e"

for _p in [
    str(_REPO / "src" / "evaluation"),
    str(_REPO / "src" / "pseudo_labeling"),
    str(_REPO / "src" / "spatial"),
    str(_REPO / "src" / "benchmark"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from signature_rules import apply_signature_rules  # noqa: E402
from evaluator import evaluate_predictions_file  # noqa: E402
from marker_rules import load_marker_purity_rules  # noqa: E402
from postprocess import postprocess_predictions_dir  # noqa: E402
from dataset_context import DatasetContext  # noqa: E402
from executors import MethodExecutionError, run_image_pipeline  # noqa: E402
from method_registry import METHOD_REGISTRY  # noqa: E402


def test_marker_purity_rules_per_dataset():
    crc = load_marker_purity_rules("CRC_TMA")
    assert "Macrophage" in crc
    assert "CD163" in crc["Macrophage"]["positive"]
    immu = load_marker_purity_rules("IMMUcan")
    assert "CD8+_T_cell" in immu


def test_signature_eval_spatial_chain(tmp_path):
    quant = pd.read_csv(_FIXTURES / "tiny_quant.csv")
    dm = _FIXTURES / "decision_matrix.csv"
    preds = apply_signature_rules(quant, dm, threshold=0.5, min_score=0.5)

    out = quant.copy()
    out["predicted_phenotype"] = preds.values
    out["true_phenotype"] = out["cell_type"]
    pred_path = tmp_path / "predictions_1.csv"
    out.to_csv(pred_path, index=False)

    quant_path = tmp_path / "quant.csv"
    quant.to_csv(quant_path, index=False)

    row = evaluate_predictions_file(
        pred_path,
        level="level3",
        marker_cols=["CD3", "CD20", "Ecad"],
        marker_rules=load_marker_purity_rules("IMMUcan"),
    )
    assert row["accuracy"] == pytest.approx(1.0)
    assert row["marker_purity"] is not None

    result_dir = tmp_path / "signature" / "level3"
    result_dir.mkdir(parents=True)
    shutil.move(str(pred_path), str(result_dir / "predictions_1.csv"))
    n = postprocess_predictions_dir(result_dir, quant_path)
    assert n == 1
    assert (result_dir / "predictions_1_spatial.csv").is_file()


def test_unlabeled_method_filter():
    from method_registry import filter_unlabeled_methods, is_unlabeled_compatible

    assert is_unlabeled_compatible("signature")
    assert is_unlabeled_compatible("leiden")
    assert not is_unlabeled_compatible("random_forest")
    assert not is_unlabeled_compatible("stellar")
    filtered = filter_unlabeled_methods(
        ["random_forest", "signature", "xgboost", "leiden", "louvain", "spade", "singler"]
    )
    assert filtered == ["signature", "leiden", "louvain", "spade"]


def test_evaluate_predictions_without_ground_truth(tmp_path):
    pred = tmp_path / "predictions_1.csv"
    pd.DataFrame({
        "Cell_ID": [1, 2, 3, 4],
        "predicted_phenotype": ["A", "A", "B", "B"],
        "x": [0.0, 0.1, 10.0, 10.1],
        "y": [0.0, 0.1, 0.0, 0.1],
        "CD3": [0.9, 0.8, 0.1, 0.2],
        "CD20": [0.1, 0.2, 0.9, 0.8],
    }).to_csv(pred, index=False)

    row = evaluate_predictions_file(
        pred,
        level="level3",
        marker_cols=["CD3", "CD20"],
        marker_rules=load_marker_purity_rules("IMMUcan"),
    )
    assert row["has_ground_truth"] is False
    assert "accuracy" not in row
    assert row["silhouette"] is not None or row["neighborhood_consistency"] is not None


def test_image_methods_reject_crc_tma(tmp_path):
    cfg = {
        "dataset_name": "CRC_TMA",
        "output": {"processed_dir": "data/processed/CRC_TMA", "output_filename": "CRC_TMA_quantification.csv"},
        "column_mappings": {"batch_id": "batch_id", "image_id": "Image_ID"},
        "protein_markers": [],
    }
    cfg_path = tmp_path / "crc_tma.yaml"
    import yaml

    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")
    ctx = DatasetContext.from_config(
        cfg_path,
        root=_REPO,
        benchmark_overrides={"image_data_dir": str(tmp_path / "images")},
    )
    ctx.image_data_dir = tmp_path / "images"
    ctx.image_data_dir.mkdir()

    spec = METHOD_REGISTRY["virtues_supervised"]
    with pytest.raises(MethodExecutionError, match="not configured for image-based"):
        run_image_pipeline(ctx, spec)

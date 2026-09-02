"""Tests for k-fold strategies and cell-type representation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "methods" / "utils"))

from kfold_strategies import (  # noqa: E402
    build_cell_type_representation,
    classify_frequency_tier,
    progressive_kfold_splits,
    result_method_id,
)
from data_handler import DataSetHandler  # noqa: E402


def test_result_method_id_mapping():
    from kfold_strategies import infer_kfold_strategy_from_result_method, result_method_id  # noqa: E402

    assert result_method_id("random_forest", "StratifiedKFold") == "random_forest"
    assert result_method_id("random_forest", "ProgressiveKFold") == "random_forest_progressive"
    assert infer_kfold_strategy_from_result_method("random_forest") == "StratifiedKFold"
    assert infer_kfold_strategy_from_result_method("random_forest_progressive") == "ProgressiveKFold"
    assert classify_frequency_tier(0.005, rare_fraction=0.01, common_fraction=0.05) == "rare"
    assert classify_frequency_tier(0.03, rare_fraction=0.01, common_fraction=0.05) == "intermediate"
    assert classify_frequency_tier(0.10, rare_fraction=0.01, common_fraction=0.05) == "common"


def test_progressive_kfold_grows_training_sets():
    # 3 classes, imbalanced
    y = np.array([0] * 50 + [1] * 30 + [2] * 20)
    n_splits = 5
    folds = progressive_kfold_splits(y, n_splits=n_splits, random_state=42)
    assert len(folds) == n_splits

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    standard_tests = [test for _, test in skf.split(np.zeros(len(y)), y)]

    train_sizes = []
    for i, (train_idx, test_idx) in enumerate(folds):
        assert np.array_equal(test_idx, standard_tests[i])
        train_sizes.append(len(train_idx))
        assert len(test_idx) > 0
        # Every class present in test fold (stratified)
        assert len(np.unique(y[test_idx])) == 3

    assert train_sizes == sorted(train_sizes)
    assert train_sizes[-1] > train_sizes[0]


def test_build_cell_type_representation_ranks_types():
    labels = pd.DataFrame({"label": [0, 1, 2], "phenotype": ["Common", "Mid", "Rare"]})
    y = np.array([0] * 90 + [1] * 9 + [2] * 1)
    repr_df = build_cell_type_representation(labels, y, rare_fraction=0.02, common_fraction=0.05)
    assert repr_df.iloc[0]["phenotype"] == "Common"
    assert repr_df.iloc[0]["is_most_common"]
    rare_row = repr_df[repr_df["phenotype"] == "Rare"].iloc[0]
    assert rare_row["is_rare"]
    assert rare_row["frequency_tier"] == "rare"


def test_data_handler_progressive_kfold(tmp_path):
    quant = tmp_path / "tiny_quant.csv"
    rows = []
    for i in range(100):
        rows.append({"marker_a": float(i), "marker_b": float(i % 3), "cell_type": "A" if i < 70 else "B"})
    pd.DataFrame(rows).to_csv(quant, index=False)

    handler = DataSetHandler(str(quant), random_state=42)
    handler.preprocess(dropna=False, phenotype_column="cell_type")
    handler.createFolds(5, "ProgressiveKFold", rare_fraction=0.01, common_fraction=0.05)

    train_sizes = [len(train) for train, _ in handler.fold_indices]
    assert train_sizes == sorted(train_sizes)
    assert len(handler.fold_indices) == 5

    handler.save_labels(str(tmp_path))
    handler.save_folds(str(tmp_path))
    repr_df = handler.save_cell_type_representation(str(tmp_path))
    assert not repr_df.empty
    assert (tmp_path / "cell_type_representation.csv").is_file()
    kdir = tmp_path / "kfolds_ProgressiveKFold_level3"
    assert (kdir / "fold_cell_type_representation.csv").is_file()
    assert (kdir / "fold_indices.json").is_file()

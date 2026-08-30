"""Tests for SingleR / scArches reference mapping and new method registry entries."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
_UTILS = _REPO / "src" / "methods" / "utils"
_BENCH = _REPO / "src" / "benchmark"
for _p in [str(_UTILS), str(_BENCH)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from reference_mapping import (  # noqa: E402
    build_reference_profiles,
    run_kfold_label_transfer,
    singler_predict,
)
from method_registry import METHOD_REGISTRY, is_unlabeled_compatible  # noqa: E402


def test_new_methods_registered():
    for method_id in ("louvain", "spade", "singler", "scarches"):
        assert method_id in METHOD_REGISTRY
    assert is_unlabeled_compatible("louvain")
    assert is_unlabeled_compatible("spade")
    assert not is_unlabeled_compatible("singler")
    assert not is_unlabeled_compatible("scarches")


def test_singler_correlation_assigns_expected_type():
    markers = ["A", "B"]
    train = pd.DataFrame({
        "A": [10.0, 10.1, 0.1, 0.0],
        "B": [0.0, 0.1, 10.0, 10.2],
        "encoded_phenotype": [0, 0, 1, 1],
    })
    test = pd.DataFrame({
        "A": [9.8, 0.2],
        "B": [0.2, 9.9],
        "encoded_phenotype": [0, 1],
    })
    label_dict = {0: "TypeA", 1: "TypeB"}
    preds = singler_predict(train, test, markers, label_dict)
    assert list(preds) == ["TypeA", "TypeB"]


def test_build_reference_profiles_shape():
    train = pd.DataFrame({
        "M1": [1.0, 2.0, 8.0],
        "M2": [1.0, 1.0, 9.0],
        "encoded_phenotype": [0, 0, 1],
    })
    profiles = build_reference_profiles(train, ["M1", "M2"])
    assert profiles.shape == (2, 2)
    assert profiles.loc[1, "M1"] == pytest.approx(8.0)


def test_run_kfold_label_transfer_writes_predictions(tmp_path):
    kdir = tmp_path / "kfolds"
    kdir.mkdir()
    markers = ["M1", "M2"]
    train = pd.DataFrame({
        "M1": [10.0, 10.1, 0.1, 0.0],
        "M2": [0.0, 0.1, 10.0, 10.2],
        "encoded_phenotype": [0, 0, 1, 1],
    })
    test = pd.DataFrame({
        "M1": [9.8, 0.2],
        "M2": [0.2, 9.9],
        "encoded_phenotype": [0, 1],
    })
    train.to_csv(kdir / "fold_1_train.csv", index=False)
    test.to_csv(kdir / "fold_1_test.csv", index=False)
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"label": [0, 1], "phenotype": ["TypeA", "TypeB"]}).to_csv(labels, index=False)
    out = tmp_path / "singler" / "level3"

    run_kfold_label_transfer(kdir, labels, out, markers, singler_predict)

    pred_file = out / "predictions_1.csv"
    assert pred_file.is_file()
    preds = pd.read_csv(pred_file)
    assert "predicted_phenotype" in preds.columns
    assert "true_phenotype" in preds.columns
    assert list(preds["predicted_phenotype"]) == ["TypeA", "TypeB"]

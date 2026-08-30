"""Tests for spatial integration hooks."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "spatial"))

from integration import build_quant_knn_edges  # noqa: E402


def test_build_quant_knn_edges(tmp_path):
    quant = tmp_path / "quant.csv"
    pd.DataFrame({
        "Cell_ID": [1, 2, 3, 4],
        "Image_ID": ["a", "a", "a", "a"],
        "x": [0.0, 1.0, 0.0, 1.0],
        "y": [0.0, 0.0, 1.0, 1.0],
    }).to_csv(quant, index=False)

    edges = build_quant_knn_edges(quant, k=2)
    assert not edges.empty
    assert {"source", "target", "distance", "graph_method"}.issubset(edges.columns)

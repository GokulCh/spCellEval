"""Tests for spatial graph and smoothing (Module 3)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "spatial"))

from graphs import build_knn_graph, graph_summary  # noqa: E402
from smoothing import apply_spatial_smoothing  # noqa: E402


def _toy_df():
    rng = np.random.default_rng(0)
    n = 30
    labels = ["A"] * 15 + ["B"] * 15
    return pd.DataFrame({
        "x": rng.uniform(0, 100, n),
        "y": rng.uniform(0, 100, n),
        "predicted_phenotype": labels,
        "Image_ID": ["img1"] * n,
    })


def test_build_knn_graph_single_cell():
    edges, dists = build_knn_graph(np.array([1.0]), np.array([2.0]), k=5)
    assert edges.shape == (2, 0)
    assert len(dists) == 0
    df = _toy_df()
    edges, dists = build_knn_graph(df["x"].values, df["y"].values, k=5)
    assert edges.shape[0] == 2
    assert len(dists) == edges.shape[1]


def test_graph_summary():
    df = _toy_df()
    stats = graph_summary(df, method="knn", k=5)
    assert stats["n_nodes"] == len(df)
    assert stats["n_edges"] > 0


def test_spatial_smoothing_runs():
    df = _toy_df()
    out = apply_spatial_smoothing(df, k_neighbors=5, min_neighbors=2)
    assert "spatial_smoothed_phenotype" in out.columns
    assert len(out) == len(df)

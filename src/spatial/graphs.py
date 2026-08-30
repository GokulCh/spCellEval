"""
graphs.py
=========
Spatial neighbor graph construction for GNN and microenvironment analysis.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, cKDTree


def build_knn_graph(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build undirected k-NN graph. Returns ``(edge_index [2,E], distances [E])``."""
    coords = np.column_stack([x, y])
    if len(coords) <= 1:
        empty = np.zeros((2, 0), dtype=np.int64)
        return empty, np.array([])

    tree = cKDTree(coords)
    k_query = min(k + 1, len(coords))
    dists, indices = tree.query(coords, k=k_query)

    if k_query <= 1:
        empty = np.zeros((2, 0), dtype=np.int64)
        return empty, np.array([])

    dists = np.atleast_2d(dists)
    indices = np.atleast_2d(indices)

    edges_i, edges_j, edge_d = [], [], []
    for i in range(len(coords)):
        for j_idx, d in zip(indices[i], dists[i]):
            if j_idx == i:
                continue
            edges_i.append(i)
            edges_j.append(int(j_idx))
            edge_d.append(float(d))

    edge_index = np.array([edges_i, edges_j], dtype=np.int64)
    return edge_index, np.array(edge_d)


def build_delaunay_edges(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Delaunay triangulation edges. Returns ``edge_index [2, E]``."""
    coords = np.column_stack([x, y])
    if len(coords) < 4:
        return build_knn_graph(x, y, k=min(3, len(coords) - 1))[0]

    tri = Delaunay(coords)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = sorted((simplex[i], simplex[(i + 1) % 3]))
            edges.add((a, b))

    if not edges:
        return build_knn_graph(x, y, k=10)[0]

    ei = np.array(list(edges), dtype=np.int64).T
    return np.vstack([ei, ei[::-1]])


def graph_summary(df: pd.DataFrame, method: str = "knn", k: int = 12) -> dict:
    """Quick stats for a tissue graph."""
    x, y = df["x"].values, df["y"].values
    if method == "delaunay":
        edges = build_delaunay_edges(x, y)
    else:
        edges, _ = build_knn_graph(x, y, k=k)
    n_nodes = len(df)
    n_edges = edges.shape[1] if edges.size else 0
    return {"n_nodes": n_nodes, "n_edges": n_edges, "mean_degree": 2 * n_edges / max(n_nodes, 1)}

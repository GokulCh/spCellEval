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


def _neighbor_lookup(df: pd.DataFrame, k: int = 12, radius: float = None):
    """Shared cKDTree neighbor indices for the diagnostic helpers below."""
    coords = df[["x", "y"]].to_numpy(dtype=float)
    if len(coords) < 2:
        return np.empty((0, 0), dtype=int)
    tree = cKDTree(coords)
    if radius is not None:
        _, idx_list = tree.query_ball_point(coords, r=radius, return_length=False)
        max_n = max((int(len(i)) for i in idx_list), default=0)
        out = np.zeros((len(coords), max_n), dtype=int)
        for row, idx in enumerate(idx_list):
            idx = np.asarray([int(i) for i in idx if i != row], dtype=int)
            if idx.size:
                out[row, : idx.size] = idx
        return out
    k_query = min(k + 1, len(coords))
    _, indices = tree.query(coords, k=k_query)
    indices = np.atleast_2d(indices)
    return np.asarray(
        [[int(j) for j in row if int(j) != i] for i, row in enumerate(indices)],
        dtype=int,
    )


def neighborhood_composition(
    df: pd.DataFrame,
    label_col: str,
    k: int = 12,
    radius: float = None,
) -> pd.DataFrame:
    """Objective 5 — fraction of each cell's k-NN/radius neighborhood per phenotype.

    Returns a phenotype × phenotype matrix where entry ``(cell_type, neighbor)``
    is the mean fraction of a cell's neighbors belonging to *neighbor*.
    """
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError("Spatial columns 'x'/'y' are required for neighborhood diagnostics.")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not in data.")

    labels_arr = df[label_col].astype(str).to_numpy()
    labels = sorted(set(labels_arr))
    idx = {lab: i for i, lab in enumerate(labels)}
    neighbors = _neighbor_lookup(df, k=k, radius=radius)

    mean_frac = np.zeros((len(labels), len(labels)), dtype=float)
    for cell, nbrs in enumerate(neighbors):
        if nbrs.size == 0:
            continue
        for lab in nbrs:
            mean_frac[idx[labels_arr[lab]]] += 1.0 / nbrs.size
    mean_frac /= max(len(df), 1)

    return pd.DataFrame(
        mean_frac,
        index=pd.Index(labels, name=label_col),
        columns=pd.Index(labels, name="neighbor"),
    )


def local_niche_proportions(
    df: pd.DataFrame,
    label_col: str,
    k: int = 12,
    radius: float = None,
) -> pd.DataFrame:
    """Objective 5 — per-cell local niche (neighbor label proportions).

    Returns a DataFrame with one row per cell: the fraction of its k-NN/radius
    neighbors in each phenotype, plus ``n_neighbors``. Useful for measuring how
    homogenous a cell's micro-environment is and for spatial-context features.
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not in data.")

    labels_arr = df[label_col].astype(str).to_numpy()
    labels = sorted(set(labels_arr))
    idx = {lab: i for i, lab in enumerate(labels)}
    neighbors = _neighbor_lookup(df, k=k, radius=radius)

    rows = []
    for cell, nbrs in enumerate(neighbors):
        props = {lab: 0.0 for lab in labels}
        if nbrs.size:
            for lab in nbrs:
                props[labels_arr[lab]] += 1.0 / nbrs.size
        props["n_neighbors"] = int(nbrs.size)
        rows.append(props)
    out = pd.DataFrame(rows, index=df.index)
    if "x" in df.columns:
        out["x"] = df["x"].values
        out["y"] = df["y"].values
    return out


def spatial_lag_features(
    df: pd.DataFrame,
    marker_cols: list,
    k: int = 12,
    radius: float = None,
) -> pd.DataFrame:
    """Objective 5 — locally averaged (lag) marker expression for each cell.

    For every cell, the mean expression of each marker across its k-NN/radius
    neighbors; cells without neighbors keep their own expression. This is the
    classic spatial-lag feature used as an input to improve TMA classification.
    """
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError("Spatial columns 'x'/'y' are required for lag features.")
    markers = [c for c in marker_cols if c in df.columns]
    if not markers:
        raise ValueError(f"No marker columns found among: {marker_cols}")

    neighbors = _neighbor_lookup(df, k=k, radius=radius)
    expression = df[markers].to_numpy(dtype=float)
    out = np.empty_like(expression, dtype=float)
    for cell, nbrs in enumerate(neighbors):
        if nbrs.size:
            out[cell] = expression[nbrs].mean(axis=0)
        else:
            out[cell] = expression[cell]

    return pd.DataFrame(
        out,
        index=df.index,
        columns=[f"{m}_spatial_lag" for m in markers],
    )

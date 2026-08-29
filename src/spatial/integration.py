"""
integration.py
==============
Shared spatial hooks for image/GNN pipelines (STELLAR, VirTues, benchmark runner).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from graphs import build_delaunay_edges, build_knn_graph
from postprocess import export_spatial_graph_summary

logger = logging.getLogger(__name__)


def load_spatial_coords(quant_path: Path) -> pd.DataFrame:
    """Load cell coordinates (+ IDs) from a quantification CSV."""
    usecols = ["Cell_ID", "cell_id", "Image_ID", "x", "y"]
    df = pd.read_csv(quant_path, usecols=lambda c: c in usecols)
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError(f"Spatial coordinates missing in {quant_path}")
    return df


def build_quant_knn_edges(
    quant_path: Path,
    *,
    k: int = 12,
    per_image: bool = True,
) -> pd.DataFrame:
    """Build k-NN edge list from quantification spatial coordinates."""
    df = load_spatial_coords(quant_path)
    rows = []
    groups = df.groupby("Image_ID") if per_image and "Image_ID" in df.columns else [("_all", df)]
    for image_id, grp in groups:
        x, y = grp["x"].values.astype(float), grp["y"].values.astype(float)
        edge_index, dists = build_knn_graph(x, y, k=k)
        id_col = "Cell_ID" if "Cell_ID" in grp.columns else "cell_id"
        cell_ids = grp[id_col].values
        for ei, (i, j) in enumerate(zip(edge_index[0], edge_index[1])):
            rows.append({
                "Image_ID": image_id,
                "source": cell_ids[i],
                "target": cell_ids[j],
                "distance": float(dists[ei]),
                "graph_method": "knn",
            })
    return pd.DataFrame(rows)


def attach_spatial_artifacts(
    quant_path: Path,
    method_result_dir: Path,
    *,
    k: int = 12,
    graph_method: str = "both",
    export_edges: bool = True,
) -> Optional[Path]:
    """Attach spatial graph summaries (and optional edge list) to a method result dir."""
    if not quant_path.is_file():
        logger.warning("Quant path not found for spatial artifacts: %s", quant_path)
        return None

    spatial_dir = method_result_dir / "spatial"
    summary_path = export_spatial_graph_summary(
        quant_path,
        spatial_dir,
        k=k,
        graph_method=graph_method,
    )

    if export_edges:
        try:
            edges = build_quant_knn_edges(quant_path, k=k)
            edge_path = spatial_dir / "knn_edges.csv"
            spatial_dir.mkdir(parents=True, exist_ok=True)
            edges.to_csv(edge_path, index=False)
            logger.info("Wrote spatial edge list: %s", edge_path)
        except Exception as exc:
            logger.warning("Could not export k-NN edges: %s", exc)

    return summary_path


def delaunay_edge_index(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Re-export Delaunay edges for GNN pipelines (STELLAR/VirTues compatibility)."""
    return build_delaunay_edges(x, y)

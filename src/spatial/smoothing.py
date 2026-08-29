"""
smoothing.py
============
Sliding-window spatial inference to refine cell-type predictions (Module 3).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def _majority_vote(labels: list) -> str:
    counts = Counter(labels)
    return counts.most_common(1)[0][0]


def apply_spatial_smoothing(
    df: pd.DataFrame,
    label_col: str = "predicted_phenotype",
    output_col: str = "spatial_smoothed_phenotype",
    *,
    image_col: str = "Image_ID",
    x_col: str = "x",
    y_col: str = "y",
    radius: Optional[float] = None,
    k_neighbors: int = 15,
    min_neighbors: int = 3,
) -> pd.DataFrame:
    """Refine predictions using local neighborhood majority voting.

    For each cell, gather neighbors within *radius* (auto-estimated from k-NN
    if ``None``) and assign the majority label. Operates per ``image_col``
    group to respect tissue boundaries.
    """
    out = df.copy()
    if label_col not in out.columns:
        raise ValueError(f"Column '{label_col}' not found.")

    smoothed = out[label_col].astype(str).copy()

    groups = [("_all", out)] if image_col not in out.columns else out.groupby(image_col)
    for _, grp in groups:
        idx = grp.index
        coords = grp[[x_col, y_col]].values.astype(float)
        labels = grp[label_col].astype(str).values

        if len(coords) < min_neighbors:
            continue

        tree = cKDTree(coords)
        if radius is None:
            dists, _ = tree.query(coords, k=min(k_neighbors + 1, len(coords)))
            if dists.ndim == 1:
                dists = dists.reshape(-1, 1)
            radius = float(np.median(dists[:, -1])) * 1.5

        for i, (coord, label) in enumerate(zip(coords, labels)):
            neighbor_idx = tree.query_ball_point(coord, r=radius)
            if len(neighbor_idx) < min_neighbors:
                smoothed.loc[idx[i]] = label
                continue
            neighbor_labels = [labels[j] for j in neighbor_idx]
            smoothed.loc[idx[i]] = _majority_vote(neighbor_labels)

    out[output_col] = smoothed.values
    changed = (out[label_col].astype(str) != out[output_col].astype(str)).sum()
    logger.info(
        "Spatial smoothing: %d / %d cells changed (radius=%.1f, k=%d).",
        changed, len(out), radius or 0, k_neighbors,
    )
    return out


def smooth_predictions_file(
    predictions_path: str,
    quant_path: Optional[str] = None,
    output_path: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """Load predictions, merge spatial coords from quant CSV if needed, smooth."""
    pred = pd.read_csv(predictions_path)
    if quant_path and ("x" not in pred.columns or "y" not in pred.columns):
        quant = pd.read_csv(quant_path, usecols=lambda c: c in {
            "Cell_ID", "x", "y", "Image_ID", "cell_id", "image",
        })
        id_col = "Cell_ID" if "Cell_ID" in pred.columns else "cell_id"
        qid = "Cell_ID" if "Cell_ID" in quant.columns else "cell_id"
        pred = pred.merge(quant, left_on=id_col, right_on=qid, how="left", suffixes=("", "_q"))

    result = apply_spatial_smoothing(pred, **kwargs)
    if output_path:
        result.to_csv(output_path, index=False)
    return result

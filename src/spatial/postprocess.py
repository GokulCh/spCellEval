"""
postprocess.py
==============
Spatial post-processing hooks for benchmark outputs (Module 3).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from smoothing import smooth_predictions_file

_UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(_UTILS) not in sys.path:
    sys.path.insert(0, str(_UTILS))

from prediction_files import should_apply_spatial_smoothing  # noqa: E402

logger = logging.getLogger(__name__)


def postprocess_predictions_dir(
    result_dir: Path,
    quant_path: Path,
    *,
    overwrite: bool = False,
    k_neighbors: int = 15,
    parallel_jobs: int = 1,
) -> int:
    """Apply spatial smoothing to all ``predictions_*.csv`` under *result_dir*."""
    if not result_dir.is_dir():
        return 0

    pred_paths = [
        pred_path
        for pred_path in sorted(result_dir.rglob("predictions_*.csv"))
        if not pred_path.name.endswith("_spatial.csv")
        and should_apply_spatial_smoothing(pred_path, quant_path)
    ]
    if not pred_paths:
        return 0

    def _smooth_one(pred_path: Path) -> int:
        out_path = pred_path if overwrite else pred_path.with_name(pred_path.stem + "_spatial.csv")
        smooth_predictions_file(
            str(pred_path),
            quant_path=str(quant_path),
            output_path=str(out_path),
            k_neighbors=k_neighbors,
        )
        df = pd.read_csv(out_path)
        if "spatial_smoothed_phenotype" in df.columns:
            df["predicted_phenotype"] = df["spatial_smoothed_phenotype"]
            df.to_csv(out_path, index=False)
        logger.info("Spatial post-process: %s -> %s", pred_path.name, out_path.name)
        return 1

    if parallel_jobs <= 1 or len(pred_paths) == 1:
        return sum(_smooth_one(path) for path in pred_paths)

    from parallel import parallel_map

    return sum(
        parallel_map(
            _smooth_one,
            pred_paths,
            max_workers=parallel_jobs,
            description="spatial smoothing files",
        )
    )


def export_spatial_graph_summary(
    quant_path: Path,
    output_dir: Path,
    *,
    k: int = 12,
    per_image: bool = True,
    graph_method: str = "both",
) -> Optional[Path]:
    """Write k-NN and/or Delaunay graph summaries per tissue."""
    from graphs import graph_summary

    if not quant_path.is_file():
        logger.warning("Quant file not found: %s", quant_path)
        return None

    df = pd.read_csv(quant_path, usecols=lambda c: c in {"x", "y", "Image_ID"})
    if "x" not in df.columns or "y" not in df.columns:
        logger.warning("Spatial coordinates missing in %s", quant_path)
        return None

    methods = ["knn", "delaunay"] if graph_method == "both" else [graph_method]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def _append_stats(grp: pd.DataFrame, image_id: str | None, method: str) -> None:
        stats = graph_summary(grp, method=method, k=k)
        if image_id is not None:
            stats["Image_ID"] = image_id
        stats["graph_method"] = method
        rows.append(stats)

    for method in methods:
        if per_image and "Image_ID" in df.columns:
            for image_id, grp in df.groupby("Image_ID"):
                _append_stats(grp, str(image_id), method)
        else:
            _append_stats(df, None, method)

    out_path = output_dir / "spatial_graph_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info("Wrote spatial graph summary: %s", out_path)
    return out_path

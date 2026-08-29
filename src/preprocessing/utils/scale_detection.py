"""
scale_detection.py
==================
Heuristics to detect whether marker data needs arcsinh transformation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def needs_arcsinh_transform(
    marker_data: pd.DataFrame,
    *,
    already_transformed_p99: float = 10.0,
    already_transformed_median: float = 3.0,
    raw_count_p99: float = 50.0,
) -> bool:
    """Return True when marker values look like raw counts (not yet arcsinh-scaled).

    Skips transform when the bulk of values already sit in a typical
    post-arcsinh range (small positive values, low dynamic range).
    """
    vals = marker_data.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return False

    p99 = float(np.percentile(vals, 99))
    p50 = float(np.percentile(vals, 50))
    vmax = float(np.max(vals))

    if p99 <= already_transformed_p99 and p50 <= already_transformed_median and vals.min() >= -1:
        return False

    if p99 >= raw_count_p99 or (vmax > 20 and p50 > 5):
        return True

    # Borderline: apply transform when enabled and values are strictly non-negative counts
    if vals.min() >= 0 and p99 > already_transformed_p99:
        return True

    return False

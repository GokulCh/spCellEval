"""Tests for auto scale detection (Module 1 ETL)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "preprocessing" / "utils"))

from scale_detection import needs_arcsinh_transform  # noqa: E402


def test_skip_already_transformed():
    data = pd.DataFrame({"CD3": np.random.uniform(0, 2, 100)})
    assert needs_arcsinh_transform(data) is False


def test_detect_raw_counts():
    data = pd.DataFrame({"CD3": np.random.uniform(100, 5000, 100)})
    assert needs_arcsinh_transform(data) is True

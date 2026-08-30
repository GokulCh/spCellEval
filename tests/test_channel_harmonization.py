"""Tests for channel harmonization (Module 1 ETL)."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "preprocessing" / "utils"))

from channel_harmonization import (  # noqa: E402
    clean_channel_name,
    harmonize_column_names,
    strip_acquisition_columns,
)


def test_clean_cyc_suffix():
    raw = "CD8 - cytotoxic T cells:Cyc_3_ch_2"
    assert clean_channel_name(raw) == "CD8"


def test_clean_isotope_suffix():
    assert clean_channel_name("CD4_154Sm") == "CD4"


def test_harmonize_columns():
    df = pd.DataFrame({
        "CD3 - T cells:Cyc_16_ch_4": [1.0],
        "flag_no_cells": [0],
        "x": [10.0],
    })
    out = harmonize_column_names(df)
    assert "CD3" in out.columns
    assert "CD3 - T cells:Cyc_16_ch_4" not in out.columns


def test_strip_acquisition_columns():
    df = pd.DataFrame({"CD3": [1.0], "flag_no_cells": [1], "x": [2.0]})
    out = strip_acquisition_columns(df)
    assert "flag_no_cells" not in out.columns
    assert "CD3" in out.columns

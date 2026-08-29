"""
channel_harmonization.py
========================
Channel name cleaning and acquisition-metadata stripping for Module 1 ETL.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Set

import pandas as pd

# CRC-TMA cycle suffix:  "CD8 - cytotoxic T cells:Cyc_3_ch_2"
_CYC_SUFFIX_RE = re.compile(r":Cyc_\d+_ch_\d+$", re.IGNORECASE)

# Isotope / metal tag suffixes:  "CD4_154Sm", "CD45RO_169Tm", "PanCK_141Pr"
_ISOTOPE_SUFFIX_RE = re.compile(r"[_\s](\d{2,3})(Sm|Nd|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Ir|Pt|Au|Ag|Cu|Zn|Fe|Co|Ni|Mn|Cr|V|Ti|Sc|Ca|K|Cl|S|P|O|N|C|H)$", re.IGNORECASE)
# Simpler pattern: trailing _<mass><element>
_ISOTOPE_SIMPLE_RE = re.compile(r"[_-](\d{1,3}[A-Z][a-z]?)$")

# Default acquisition / QC columns to strip from analytical matrices
DEFAULT_STRIP_COLUMNS: Set[str] = {
    "flag_no_cells", "flag_no_ROI", "flag_total_area", "flag_percent_covered",
    "flag_tumor", "small_cell", "ROIonSlide", "includeImage",
    "distToCells", "tumor_patches", "CD20_patches",
    "PD1_pos", "Ki67_pos", "cleavedPARP_pos", "GrzB_pos",
    "Box.Description", "Box.Description.1",
    "SubBatchId", "SampleId", "csv", "orig.ident",
}


def clean_channel_name(raw: str) -> str:
    """Normalize a raw channel/column name to a canonical protein identifier."""
    name = str(raw).strip()
    name = _CYC_SUFFIX_RE.sub("", name)
    name = _ISOTOPE_SUFFIX_RE.sub("", name)
    name = _ISOTOPE_SIMPLE_RE.sub("", name)
    # Common aliases
    aliases = {
        "HLADR": "HLADR", "HLA-DR": "HLADR", "HLA-DR - MHC-II": "HLADR",
        "Ecad": "Ecad", "Cytokeratin - epithelia": "PanCK",
        "CD8a": "CD8a", "CD8 - cytotoxic T cells": "CD8",
        "CD4 - T helper cells": "CD4", "CD3 - T cells": "CD3",
        "CD20 - B cells": "CD20",         "CD163 - macrophages": "CD163",
        "FOXP3 - regulatory T cells": "FOXP3",
        "CD138 - plasma cells": "CD138",
        "aSMA - smooth muscle": "SMA",
        "Vimentin - cytoplasm": "Vimentin",
        "CD31 - vasculature": "CD31",
        "Podoplanin - lymphatics": "Podoplanin",
        "CD68 - macrophages": "CD68",
        "CD11c - DCs": "CD11c",
        "CD11b - macrophages": "CD11b",
        "CD56 - NK cells": "CD56",
        "CD15 - granulocytes": "CD15",
        "PD-L1 - checkpoint": "PDL1", "PD-1 - checkpoint": "PD1",
    }
    return aliases.get(name, name.strip())


def harmonize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename all columns using :func:`clean_channel_name` (deduplicate collisions)."""
    new_names = {}
    used: Set[str] = set()
    for col in df.columns:
        clean = clean_channel_name(col)
        if clean in used and clean != col:
            clean = f"{clean}__{col[:20]}"
        used.add(clean)
        if clean != col:
            new_names[col] = clean
    if new_names:
        df = df.rename(columns=new_names)
    return df


def strip_acquisition_columns(
    df: pd.DataFrame,
    extra_strip: Optional[Iterable[str]] = None,
    keep: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Remove non-analytical acquisition/QC flag columns."""
    keep_set = set(keep or [])
    to_drop = [c for c in DEFAULT_STRIP_COLUMNS if c in df.columns and c not in keep_set]
    if extra_strip:
        to_drop.extend(c for c in extra_strip if c in df.columns and c not in keep_set)
    to_drop = list(dict.fromkeys(to_drop))
    if to_drop:
        df = df.drop(columns=to_drop)
    return df


def auto_detect_markers(
    df: pd.DataFrame,
    exclude: Optional[Set[str]] = None,
) -> List[str]:
    """Heuristic: numeric columns that are not spatial/identity/label metadata."""
    exclude = exclude or set()
    meta = {
        "Cell_ID", "Image_ID", "Patient_ID", "x", "y", "batch_id",
        "cell_type", "level_1_cell_type", "level_2_cell_type", "cell_labels",
        "ObjectNumber", "Pos_X", "Pos_Y", "image", "sample_id", "BatchId",
        "celltypes", "ClusterName", "CellID", "File Name", "patients", "TMA_AB",
    } | exclude
    markers = [
        c for c in df.columns
        if c not in meta
        and pd.api.types.is_numeric_dtype(df[c])
        and not c.lower().startswith("flag_")
        and c not in DEFAULT_STRIP_COLUMNS
    ]
    return markers

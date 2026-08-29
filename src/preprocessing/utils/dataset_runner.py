"""
dataset_runner.py
=================
Core preprocessing engine for the spCellEval benchmark.

Given a loaded YAML configuration dict (see ``configs/<dataset>.yaml``),
``DatasetRunner`` performs the following steps in order:

1. Load the raw CSV.
2. Rename columns to the standardised schema.
3. (Optional) Apply arcsinh transform to protein marker channels.
4. Derive ``level_1_cell_type`` and ``level_2_cell_type`` from the shared
   hierarchy file via ``label_hierarchy.add_hierarchy_columns()``.
5. Reorder columns so that markers come first, then metadata, then labels.
6. Save the processed CSV to the configured output path.

Standardised output column order
---------------------------------
<marker_1>, ..., <marker_N>,          ← arcsinh-transformed protein channels
Cell_ID, Image_ID, Patient_ID,        ← spatial / identity
x, y,                                 ← spatial coordinates
<morphology_features...>,             ← optional shape descriptors
batch_id,                             ← batch / slide
<additional_obs...>,                  ← dataset-specific extra metadata
cell_labels,                          ← raw label string (pre-harmonisation)
level_1_cell_type,                    ← coarsest: Immune / Stromal / Cancer
level_2_cell_type,                    ← mid: Myeloid / Lymphoid / Fibroblast
cell_type                             ← finest (level 3): CD8+_T_cell …
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from label_hierarchy import add_hierarchy_columns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper: strip cycle / channel suffixes from CRC-TMA-style marker names
# e.g.  "CD8 - cytotoxic T cells:Cyc_3_ch_2"  →  "CD8 - cytotoxic T cells"
# ---------------------------------------------------------------------------
_CYC_SUFFIX_RE = re.compile(r":Cyc_\d+_ch_\d+$", re.IGNORECASE)


def _clean_marker_name(raw: str) -> str:
    """Strip the ':Cyc_N_ch_M' suffix if present."""
    return _CYC_SUFFIX_RE.sub("", raw).strip()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DatasetRunner:
    """Preprocessing engine driven by a YAML config dict.

    Parameters
    ----------
    config:
        Parsed YAML dict (output of ``yaml.safe_load``).
    root_dir:
        Workspace root – all relative paths in the config are resolved
        against this directory.  Defaults to the current working directory.
    hierarchy_path:
        Override path to ``cell_type_hierarchy.txt``.  When ``None`` the
        default path bundled with ``label_hierarchy`` is used.
    """

    # ------------------------------------------------------------------
    # Standard output column names (the schema contract)
    # ------------------------------------------------------------------
    OUT_CELL_ID    = "Cell_ID"
    OUT_IMAGE_ID   = "Image_ID"
    OUT_PATIENT_ID = "Patient_ID"
    OUT_X          = "x"
    OUT_Y          = "y"
    OUT_BATCH_ID   = "batch_id"
    OUT_CELL_LABELS = "cell_labels"
    OUT_L1         = "level_1_cell_type"
    OUT_L2         = "level_2_cell_type"
    OUT_CELL_TYPE  = "cell_type"

    def __init__(
        self,
        config: Dict[str, Any],
        root_dir: str | Path = ".",
        hierarchy_path: Optional[str | Path] = None,
    ) -> None:
        self.cfg = config
        self.root = Path(root_dir).resolve()
        self.hierarchy_path = hierarchy_path
        self.df: Optional[pd.DataFrame] = None
        self._marker_cols: List[str] = []   # output (cleaned) marker names

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Execute the full pipeline and return the processed DataFrame."""
        self._load()
        self._rename_columns()
        self._apply_label_remap()
        self._arcsinh_transform()
        self._add_hierarchy_labels()
        self._reorder_columns()
        return self.df

    def save(self, df: Optional[pd.DataFrame] = None) -> Path:
        """Write the processed DataFrame to the configured output path.

        Parameters
        ----------
        df:
            DataFrame to save.  If ``None``, uses ``self.df`` (i.e. the
            result of the most recent ``run()`` call).

        Returns
        -------
        Path
            Absolute path of the written file.
        """
        if df is not None:
            self.df = df
        if self.df is None:
            raise RuntimeError("Call run() before save().")

        out_cfg = self.cfg.get("output", {})
        out_dir = self.root / out_cfg.get("processed_dir", "data/processed")
        out_dir.mkdir(parents=True, exist_ok=True)

        dataset_name = self.cfg.get("dataset_name", "dataset")
        filename = out_cfg.get("output_filename", f"{dataset_name}_quantification.csv")
        out_path = out_dir / filename

        self.df.to_csv(out_path, index=False)
        logger.info("Saved %d × %d to %s", *self.df.shape, out_path)
        return out_path

    # ------------------------------------------------------------------
    # Pipeline steps (private)
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the raw data file."""
        raw_cfg = self.cfg.get("raw_data", {})
        file_path = self.root / raw_cfg.get("file_path", "")
        fmt = raw_cfg.get("format", "csv").lower()

        logger.info("Loading %s (%s) …", file_path, fmt)

        if fmt == "csv":
            self.df = pd.read_csv(file_path, low_memory=False)
        elif fmt == "parquet":
            self.df = pd.read_parquet(file_path)
        elif fmt == "h5ad":
            import anndata
            adata = anndata.read_h5ad(file_path)
            # Flatten to DataFrame: obs metadata + var (marker) expression
            expr = pd.DataFrame(
                adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
                columns=adata.var_names,
                index=adata.obs_names,
            )
            self.df = pd.concat([expr, adata.obs.reset_index(drop=True)], axis=1)
        else:
            raise ValueError(
                f"Unsupported format '{fmt}'. "
                "Pre-convert .rds / .RData to CSV using the R Markdown "
                "scripts in src/preprocessing/datasets/."
            )

        logger.info("Loaded %d rows × %d columns.", *self.df.shape)

    def _rename_columns(self) -> None:
        """Rename raw columns to standardised schema names."""
        mappings: Dict[str, str] = self.cfg.get("column_mappings", {})

        rename_map: Dict[str, str] = {}

        if "cell_id" in mappings:
            rename_map[mappings["cell_id"]] = self.OUT_CELL_ID
        if "image_id" in mappings:
            rename_map[mappings["image_id"]] = self.OUT_IMAGE_ID
        if "patient_id" in mappings and mappings["patient_id"]:
            rename_map[mappings["patient_id"]] = self.OUT_PATIENT_ID
        if "batch_id" in mappings and mappings["batch_id"]:
            rename_map[mappings["batch_id"]] = self.OUT_BATCH_ID
        if "spatial_x" in mappings:
            rename_map[mappings["spatial_x"]] = self.OUT_X
        if "spatial_y" in mappings:
            rename_map[mappings["spatial_y"]] = self.OUT_Y

        # Rename marker columns (strip cycle suffixes, create cleaned names)
        raw_markers: List[str] = self.cfg.get("protein_markers", [])
        self._marker_cols = []
        for raw in raw_markers:
            clean = _clean_marker_name(raw)
            if raw != clean:
                rename_map[raw] = clean
            self._marker_cols.append(clean)

        # Only rename columns that actually exist
        actual_rename = {k: v for k, v in rename_map.items() if k in self.df.columns}
        missing = [k for k in rename_map if k not in self.df.columns]
        if missing:
            logger.warning("The following expected raw columns were not found: %s", missing)

        self.df = self.df.rename(columns=actual_rename)
        logger.debug("Renamed %d columns.", len(actual_rename))

    def _apply_label_remap(self) -> None:
        """Apply label_remap from the config to the ground-truth label column.

        Also copies the original label into ``cell_labels`` for traceability.
        """
        mappings = self.cfg.get("column_mappings", {})
        raw_label_col = mappings.get("ground_truth_label")

        if not raw_label_col:
            logger.warning("No ground_truth_label mapping found; skipping label remap.")
            return

        # After _rename_columns the marker columns are clean; the label
        # column may still have its original name if it was not in rename_map.
        if raw_label_col not in self.df.columns:
            logger.warning(
                "Label column '%s' not found in DataFrame; skipping remap.", raw_label_col
            )
            return

        # Store original labels
        self.df[self.OUT_CELL_LABELS] = self.df[raw_label_col].astype(str)

        # Apply harmonisation map
        remap: Dict[str, str] = self.cfg.get("label_remap", {})
        if remap:
            self.df[self.OUT_CELL_TYPE] = (
                self.df[raw_label_col].astype(str).replace(remap)
            )
        else:
            self.df[self.OUT_CELL_TYPE] = self.df[raw_label_col].astype(str)

        # Drop the raw label column if it differs from OUT_CELL_TYPE
        if raw_label_col != self.OUT_CELL_TYPE and raw_label_col in self.df.columns:
            self.df = self.df.drop(columns=[raw_label_col])

        unique_before = self.df[self.OUT_CELL_LABELS].nunique()
        unique_after  = self.df[self.OUT_CELL_TYPE].nunique()
        logger.info(
            "Label remap: %d raw labels → %d harmonised cell_type values.",
            unique_before, unique_after,
        )

    def _arcsinh_transform(self) -> None:
        """Apply arcsinh(x / cofactor) to all protein marker columns."""
        transforms = self.cfg.get("transformations", {})
        if not transforms.get("arcsinh_transform", True):
            return

        cofactor = float(transforms.get("cofactor", 5))
        qc_min   = float(transforms.get("qc_min_expression", 0))

        # Only transform columns that actually exist in the DataFrame
        present = [c for c in self._marker_cols if c in self.df.columns]
        missing = [c for c in self._marker_cols if c not in self.df.columns]
        if missing:
            logger.warning(
                "%d marker column(s) missing from data (skipped): %s",
                len(missing), missing,
            )

        if not present:
            logger.warning("No marker columns found to transform.")
            return

        marker_data = self.df[present].apply(pd.to_numeric, errors="coerce")

        if qc_min > 0:
            # Drop cells where every marker is below the QC threshold
            mask = (marker_data >= qc_min).any(axis=1)
            n_dropped = (~mask).sum()
            if n_dropped:
                logger.info("QC: dropping %d cells (all markers < %.3g).", n_dropped, qc_min)
                self.df = self.df[mask].reset_index(drop=True)
                marker_data = marker_data[mask].reset_index(drop=True)

        self.df[present] = np.arcsinh(marker_data / cofactor)
        logger.info(
            "arcsinh(x / %.3g) applied to %d marker columns.", cofactor, len(present)
        )

    def _add_hierarchy_labels(self) -> None:
        """Derive level_1_cell_type and level_2_cell_type from the hierarchy."""
        if self.OUT_CELL_TYPE not in self.df.columns:
            logger.warning(
                "'%s' column not found; cannot derive hierarchy levels.",
                self.OUT_CELL_TYPE,
            )
            return

        kwargs: Dict[str, Any] = {"cell_type_col": self.OUT_CELL_TYPE, "inplace": True}
        if self.hierarchy_path:
            kwargs["hierarchy_path"] = self.hierarchy_path

        add_hierarchy_columns(self.df, **kwargs)

        logger.info(
            "Hierarchy levels added — level_1: %d unique, level_2: %d unique.",
            self.df[self.OUT_L1].nunique(),
            self.df[self.OUT_L2].nunique(),
        )

    def _reorder_columns(self) -> None:
        """Reorder DataFrame columns to the standardised output contract."""
        extra_obs: List[str] = self.cfg.get("additional_obs", [])
        morph: List[str] = self.cfg.get("morphology_features", [])

        # Build desired column order
        meta_cols = [
            self.OUT_CELL_ID,
            self.OUT_IMAGE_ID,
            self.OUT_PATIENT_ID,
            self.OUT_X,
            self.OUT_Y,
        ]
        label_cols = [
            self.OUT_CELL_LABELS,
            self.OUT_L1,
            self.OUT_L2,
            self.OUT_CELL_TYPE,
        ]

        # Filter to columns that actually exist
        def _exists(cols: List[str]) -> List[str]:
            return [c for c in cols if c in self.df.columns]

        marker_block  = _exists(self._marker_cols)
        meta_block    = _exists(meta_cols)
        morph_block   = _exists(morph)
        batch_block   = _exists([self.OUT_BATCH_ID])
        extra_block   = _exists(extra_obs)
        label_block   = _exists(label_cols)

        ordered = (
            marker_block + meta_block + morph_block
            + batch_block + extra_block + label_block
        )

        # Append any columns we haven't explicitly placed
        remaining = [c for c in self.df.columns if c not in ordered]
        if remaining:
            logger.debug(
                "Additional columns appended after label block: %s", remaining
            )

        self.df = self.df[ordered + remaining]
        logger.debug("Column order finalised (%d columns).", len(self.df.columns))

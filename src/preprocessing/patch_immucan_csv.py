"""
patch_immucan_csv.py
====================
One-shot script that adds ``level_1_cell_type`` and ``level_2_cell_type``
to the existing IMMUcan_quantification.csv in-place, without re-running the
full preprocessing pipeline.

Run from the repo root::

    python src/preprocessing/patch_immucan_csv.py

What it does
------------
1. Reads  data/processed/IMMUcan/IMMUcan_quantification.csv
2. Applies the label remap that was previously hardcoded in
   process_IMMUcan.ipynb (Tumor→Cancer, CD8→CD8+_T_cell, etc.)
3. Calls label_hierarchy.add_hierarchy_columns() to derive level_1 / level_2
4. Overwrites the original CSV (or a new path if --out is given)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ── Path bootstrap ─────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent          # src/preprocessing/
_UTILS_DIR = _HERE / "utils"
_REPO_ROOT = _HERE.parents[1]

for _p in [str(_UTILS_DIR), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from label_hierarchy import add_hierarchy_columns  # noqa: E402

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("patch_immucan")

# ── Label remap (copied from process_IMMUcan.ipynb) ────────────────────────
LABEL_REMAP = {
    "Tumor":    "Cancer",
    "Mural":    "Stroma",
    "CD8":      "CD8+_T_cell",
    "MacCD163": "M2_Macrophage",
    "CD4":      "CD4+_T_cell",
    "plasma":   "Plasma_cell",
    "DC":       "Dendritic_cell",
    "B":        "B_cell",
    "NK":       "NK_cell",
    "pDC":      "Plasmacytoid_dendritic_cell",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=_REPO_ROOT / "data/processed/IMMUcan/IMMUcan_quantification.csv",
        help="Path to IMMUcan_quantification.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (defaults to overwriting --csv).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print summary without writing.",
    )
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    out_path = (args.out or args.csv).resolve()

    if not csv_path.exists():
        log.error("File not found: %s", csv_path)
        sys.exit(1)

    log.info("Reading %s …", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    log.info("Loaded %d × %d", *df.shape)

    # ── 1. Apply label remap to cell_type ──────────────────────────────────
    if "cell_type" not in df.columns:
        log.error("'cell_type' column not found.")
        sys.exit(1)

    if "cell_labels" not in df.columns:
        df["cell_labels"] = df["cell_type"].copy()

    before = df["cell_type"].nunique()
    df["cell_type"] = df["cell_type"].replace(LABEL_REMAP)
    after = df["cell_type"].nunique()
    log.info("Label remap: %d → %d unique cell_type values.", before, after)

    # ── 2. Derive hierarchy columns ───────────────────────────────────────
    add_hierarchy_columns(df, cell_type_col="cell_type", inplace=True)
    log.info(
        "level_1_cell_type: %d unique | level_2_cell_type: %d unique",
        df["level_1_cell_type"].nunique(),
        df["level_2_cell_type"].nunique(),
    )

    # ── 3. Reorder columns so labels are at the end ────────────────────────
    LABEL_COLS = ["cell_labels", "level_1_cell_type", "level_2_cell_type", "cell_type"]
    other_cols = [c for c in df.columns if c not in LABEL_COLS]
    df = df[other_cols + [c for c in LABEL_COLS if c in df.columns]]

    if args.dry_run:
        log.info("[DRY-RUN] Would write %d × %d to %s", *df.shape, out_path)
        print(df[["cell_labels", "level_1_cell_type", "level_2_cell_type", "cell_type"]].value_counts().head(20))
        return

    log.info("Writing %s …", out_path)
    df.to_csv(out_path, index=False)
    log.info("Done. Final shape: %d × %d", *df.shape)


if __name__ == "__main__":
    main()

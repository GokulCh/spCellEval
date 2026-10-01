"""
Convert a raw CODEX quantification table (CSV/TSV/Excel/Parquet) into the standard
spCellEval quantification CSV:

    <main_dir>/datasets/<dataset_name>/quantification/processed/<dataset_name>_quantification.csv

Output columns: marker columns first, then image, cell_id, patient, region, x, y, followed by
cell_type (and level_2_cell_type / level_1_cell_type if a hierarchy file is given).

Raw columns that are derived from the cell types (cluster ID, neighborhoods), nuclear stains and QC
columns are not carried over, so they cannot leak the label into the features.

Marker columns are recognised by the pattern "<name> - <description>:Cyc_<n>_ch_<n>" and are
renamed to the part before " - " with all non-alphanumeric characters removed (HLA-DR -> HLADR).
Use these cleaned names in marker definition files.

Usage:
    python process_crc_codex.py --input raw.csv --main_dir ~/spCellEval/data --dataset_name crc_codex
    python process_crc_codex.py --input raw.csv --main_dir ... --label_map map.json --hierarchy hier.json

--label_map  JSON {"raw label": "benchmark label", ...}
--hierarchy  JSON {"benchmark label": ["level_2 label", "level_1 label"], ...}
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

UNDEFINED = "undefined"


def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv",):
        return pd.read_csv(path)
    if ext in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format '{ext}'. Use csv, tsv, xlsx or parquet.")


def clean_marker_name(column: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", column.split(" - ")[0].split(":")[0])


def find_markers(columns, marker_regex: str, stains) -> list:
    return [c for c in columns if re.search(marker_regex, c) and not c.startswith(tuple(stains))]


def load_json(path):
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def process(args) -> pd.DataFrame:
    raw = read_table(args.input)
    print(f"Read {len(raw)} cells and {raw.shape[1]} columns from {args.input}")

    required = [args.cell_id, args.image, args.patient, args.region, args.x, args.y, args.label_column]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Columns not found in the input: {missing}")

    marker_cols = find_markers(raw.columns, args.marker_regex, args.stains)
    if not marker_cols:
        raise ValueError(f"No marker columns matched the pattern '{args.marker_regex}'")
    names = {c: clean_marker_name(c) for c in marker_cols}
    if len(set(names.values())) != len(names):
        dupes = pd.Series(list(names.values())).value_counts()
        raise ValueError(f"Marker names are not unique after cleaning: {dupes[dupes > 1].index.tolist()}")

    markers = raw[marker_cols].rename(columns=names)
    if not np.isfinite(markers.to_numpy(dtype=float)).all():
        raise ValueError("Marker columns contain NaN or infinite values. Fix or impute them before running.")
    if args.cofactor and args.cofactor > 0:
        markers = np.arcsinh(markers / args.cofactor)

    meta = raw[[args.image, args.cell_id, args.patient, args.region, args.x, args.y]].copy()
    meta.columns = ["image", "cell_id", "patient", "region", "x", "y"]
    out = pd.concat([markers, meta], axis=1)

    # labels
    labels = raw[args.label_column].astype("string")
    labels = labels.fillna(UNDEFINED).str.strip().replace("", UNDEFINED)
    label_map = load_json(args.label_map)
    if label_map:
        labels = labels.map(lambda v: label_map.get(v, v))
    out["cell_type"] = labels.astype(str).values

    hierarchy = load_json(args.hierarchy)
    if hierarchy:
        unknown = sorted(set(out["cell_type"]) - set(hierarchy) - {UNDEFINED})
        if unknown:
            raise ValueError(f"These cell types are missing from the hierarchy file: {unknown}")
        out["level_2_cell_type"] = out["cell_type"].map(lambda v: hierarchy[v][0] if v in hierarchy else UNDEFINED)
        out["level_1_cell_type"] = out["cell_type"].map(lambda v: hierarchy[v][1] if v in hierarchy else UNDEFINED)
        out = out[[c for c in out.columns if c not in ("cell_type", "level_2_cell_type", "level_1_cell_type")]
                  + ["level_1_cell_type", "level_2_cell_type", "cell_type"]]

    report(out, len(marker_cols), args)
    return out


def report(out: pd.DataFrame, n_markers: int, args) -> None:
    print(f"{n_markers} marker columns, {len(out)} cells, {out['image'].nunique()} images, "
          f"{out['patient'].nunique()} patients")
    counts = out["cell_type"].value_counts()
    print("Cells per cell_type:")
    print(counts.to_string())
    small = counts[counts < args.n_splits]
    if len(small):
        print(f"WARNING: these cell types have fewer than {args.n_splits} cells, so stratified folds will "
              f"fail or warn: {small.index.tolist()}")
    if out["patient"].nunique() < args.n_splits:
        print(f"WARNING: only {out['patient'].nunique()} patients for {args.n_splits} folds. "
              "Group-aware splitting will fall back to StratifiedKFold.")


def main():
    p = argparse.ArgumentParser(description="Convert a raw CODEX quantification table into the standard spCellEval CSV")
    p.add_argument("--input", required=True, help="Raw table (csv, tsv, xlsx, parquet)")
    p.add_argument("--main_dir", required=True, help="Main directory holding 'datasets' and 'results'")
    p.add_argument("--dataset_name", default="crc_codex", help="Name of the dataset folder. Default: crc_codex")
    p.add_argument("--label_column", default="ClusterName", help="Raw column with the cell type labels. Default: ClusterName")
    p.add_argument("--cell_id", default="CellID")
    p.add_argument("--image", default="File Name", help="Image/sample column, used as the image identifier")
    p.add_argument("--patient", default="patients", help="Column used for group-aware folds")
    p.add_argument("--region", default="Region")
    p.add_argument("--x", default="X")
    p.add_argument("--y", default="Y")
    p.add_argument("--marker_regex", default=r":Cyc_\d+_ch_\d+$", help="Regex that identifies marker columns")
    p.add_argument("--stains", nargs="*", default=["HOECHST", "DRAQ5"], help="Column prefixes to exclude (nuclear stains)")
    p.add_argument("--cofactor", type=float, default=5.0, help="arcsinh cofactor. 0 disables normalization. Default: 5")
    p.add_argument("--label_map", default=None, help="JSON mapping raw labels to benchmark labels")
    p.add_argument("--hierarchy", default=None, help="JSON mapping cell_type to [level_2, level_1]")
    p.add_argument("--n_splits", type=int, default=5, help="Number of folds you plan to use, for the sanity checks")
    args = p.parse_args()

    try:
        out = process(args)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")

    out_dir = os.path.join(args.main_dir, "datasets", args.dataset_name, "quantification", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.dataset_name}_quantification.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    marker_list = [c for c in out.columns[: out.columns.get_loc("image")]]
    with open(os.path.join(out_dir, "markers.txt"), "w") as f:
        f.write("\n".join(marker_list) + "\n")
    print(f"Saved marker names: {os.path.join(out_dir, 'markers.txt')}")


if __name__ == "__main__":
    main()

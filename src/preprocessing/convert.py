"""Convert a raw quantification table with the repository's own ``datasets/process_crc_codex.py``.

That script renames the marker columns, drops label-derived / stain / QC columns, applies arcsinh and writes
``<main_dir>/datasets/<name>/quantification/processed/<name>_quantification.csv``. This module only (a) recognises a raw
table, (b) suggests the script's column options from the file's header, and (c) runs the script.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

CONVERTER = Path(__file__).resolve().parent / "datasets" / "process_crc_codex.py"
DEFAULT_REGEX = r":Cyc_\d+_ch_\d+$"

# converter option -> (the script's default column name, accepted aliases)
OPTIONS: dict[str, tuple[str, list[str]]] = {
    "label_column": ("ClusterName", ["clustername", "celltype", "cluster", "annotation", "label", "phenotype"]),
    "cell_id": ("CellID", ["cellid", "id", "labelid", "objectid", "cell"]),
    "image": ("File Name", ["filename", "image", "imageid", "sample", "sampleid", "fov", "core", "tma"]),
    "patient": ("patients", ["patients", "patient", "patientid", "donor", "case", "subject"]),
    "region": ("Region", ["region", "roi", "tissueregion"]),
    "x": ("X", ["x", "xcentroid", "centroidx", "xcoord", "xum", "xposition"]),
    "y": ("Y", ["y", "ycentroid", "centroidy", "ycoord", "yum", "yposition"]),
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def header(path: str | Path) -> list[str]:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".xlsx", ".xls"):
        return list(pd.read_excel(p, nrows=0).columns)
    if ext == ".parquet":
        return list(pd.read_parquet(p).columns)
    return list(pd.read_csv(p, nrows=0, sep="\t" if ext in (".tsv", ".txt") else ",").columns)


def n_marker_columns(cols: list[str], regex: str = DEFAULT_REGEX) -> int:
    return sum(bool(re.search(regex, c)) for c in cols)


def looks_raw(cols: list[str], regex: str = DEFAULT_REGEX) -> bool:
    """A raw CODEX table: many ``<marker> - ...:Cyc_<n>_ch_<n>`` columns."""
    return n_marker_columns(cols, regex) >= 5


def suggest(cols: list[str]) -> dict[str, str | None]:
    """Best column for each converter option: the script's own default name if present, else an alias match."""
    out = {}
    byn = {_norm(c): c for c in cols}
    for opt, (default, aliases) in OPTIONS.items():
        out[opt] = default if default in cols else next((byn[a] for a in aliases if a in byn), None)
    return out


def dataset_name(path: str | Path) -> str:
    return re.sub(r"_(expression|quantification|raw|data)$", "", Path(path).stem, flags=re.I)


def run_converter(raw: str | Path, main_dir: str | Path, name: str, opts: dict) -> Path:
    """Run the repo script; ``opts`` keys are OPTIONS names plus cofactor / marker_regex / label_map / hierarchy / stains.

    Returns the processed CSV path. Raises RuntimeError with the script's message if it fails.
    """
    cmd = [sys.executable, str(CONVERTER), "--input", str(raw), "--main_dir", str(main_dir), "--dataset_name", name]
    for opt in OPTIONS:
        if opts.get(opt):
            cmd += [f"--{opt}", opts[opt]]
    if opts.get("cofactor") is not None:
        cmd += ["--cofactor", str(opts["cofactor"])]
    for k in ("marker_regex", "label_map", "hierarchy"):
        if opts.get(k):
            cmd += [f"--{k}", str(opts[k])]
    if opts.get("stains") is not None:
        cmd += ["--stains", *opts["stains"]]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout).strip().splitlines()[-1] if (p.stderr or p.stdout).strip() else f"exit {p.returncode}")
    print(p.stdout.strip())
    out = Path(main_dir) / "datasets" / name / "quantification" / "processed" / f"{name}_quantification.csv"
    if not out.exists():
        raise RuntimeError(f"converter finished but {out} was not written")
    return out

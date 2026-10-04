"""Dataset loading, column normalisation and transforms.

Input convention follows ``datasets/process_crc_codex.py``: marker columns first, then metadata
(``image, cell_id, patient, region, x, y``) and ``cell_type`` / ``level_2_cell_type`` /
``level_1_cell_type`` labels. Aliases are normalised so other tables also load.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LEVEL_COLUMN = {"level3": "cell_type", "level2": "level_2_cell_type", "level1": "level_1_cell_type"}
PROTEIN = {"codex", "mibi", "imc", "crc_tma", "cycif"}
RNA = {"scrna", "spatial_rna"}
MODALITIES = sorted(PROTEIN | RNA)

_ALIASES = {
    "x": ["x", "x_centroid", "centroid_x", "x_um", "center_x", "xcoord"],
    "y": ["y", "y_centroid", "centroid_y", "y_um", "center_y", "ycoord"],
    "cell_type": ["cell_type", "celltype", "cell_types", "level_3_cell_type", "annotation", "label"],
    "level_2_cell_type": ["level_2_cell_type", "level2_cell_type"],
    "level_1_cell_type": ["level_1_cell_type", "level1_cell_type"],
    "image": ["image", "sample", "sample_id", "fov", "region_id", "batch"],
}
_META = {"image", "cell_id", "patient", "region", "x", "y", "cell_type", "level_2_cell_type",
         "level_1_cell_type", "leiden", "cluster", "index", "unnamed: 0"}


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame                      # cells x markers/genes
    markers: list[str]
    y: np.ndarray | None = None          # labels at the chosen granularity (None if unlabeled)
    xy: np.ndarray | None = None         # cells x 2 spatial coordinates
    groups: np.ndarray | None = None     # image / sample id (batch + spatial graph scope)
    modality: str = "codex"
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __len__(self) -> int:
        return len(self.X)

    def subsample(self, n: int, seed: int = 0) -> "Dataset":
        if n <= 0 or n >= len(self):
            return self
        idx = np.sort(np.random.default_rng(seed).choice(len(self), n, replace=False))
        pick = lambda a: None if a is None else a[idx]
        return Dataset(self.name, self.X.iloc[idx].reset_index(drop=True), self.markers, pick(self.y),
                       pick(self.xy), pick(self.groups), self.modality, self.meta.iloc[idx].reset_index(drop=True))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip names and rename known aliases (x/y/labels/image) to the canonical spCellEval names."""
    df = df.rename(columns=lambda c: str(c).strip())
    lower = {c.lower(): c for c in df.columns}
    ren = {}
    for canon, names in _ALIASES.items():
        if canon in df.columns:
            continue
        hit = next((lower[n] for n in names if n in lower and lower[n] not in ren), None)
        if hit:
            ren[hit] = canon
    return df.rename(columns=ren)


def _read(path: Path) -> pd.DataFrame:
    s = path.suffix.lower()
    if s == ".h5ad":
        import anndata as ad
        a = ad.read_h5ad(path)
        df = a.to_df().reset_index(drop=True)
        obs = a.obs.reset_index(drop=True)
        if "spatial" in a.obsm:
            sp = np.asarray(a.obsm["spatial"])
            obs["x"], obs["y"] = sp[:, 0], sp[:, 1]
        return pd.concat([df, obs], axis=1)
    if s == ".parquet":
        return pd.read_parquet(path)
    if s in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def load_dataset(path: str | Path, name: str | None = None, modality: str = "codex",
                 level: str = "level3", marker_cols: list[str] | None = None,
                 label_col: str | None = None) -> Dataset:
    path = Path(path)
    df = normalize_columns(_read(path))
    col = label_col or LEVEL_COLUMN[level]
    if col not in df.columns and level != "level3":
        if "cell_type" not in df.columns:
            raise ValueError(f"{path.name}: label column '{col}' for {level} not found")
        from ..evaluation.repo_assets import to_level       # lazy: evaluation imports this package
        df[col] = to_level(df["cell_type"], level)          # derive via the repo's hierarchy_mappings.pkl
    markers = marker_cols or [c for c in df.columns
                              if c.lower() not in _META and c != col and pd.api.types.is_numeric_dtype(df[c])]
    if not markers:
        raise ValueError(f"{path.name}: no numeric marker columns found")
    y = df[col].astype(str).to_numpy() if col in df.columns else None
    if y is not None:                    # drop unlabeled cells
        ok = ~pd.Series(y).isin(["nan", "None", ""]).to_numpy()
        df, y = df[ok].reset_index(drop=True), y[ok]
    X = df[markers].astype("float32").fillna(0.0).reset_index(drop=True)
    xy = df[["x", "y"]].to_numpy(float) if {"x", "y"} <= set(df.columns) else None
    groups = df["image"].astype(str).to_numpy() if "image" in df.columns else None
    meta = df.drop(columns=markers, errors="ignore").reset_index(drop=True)
    return Dataset(name or path.stem.replace("_quantification", ""), X, markers, y, xy, groups, modality, meta)


def transform(ds: Dataset, method: str = "auto", cofactor: float = 5.0, normalize: str = "none",
              batch_correct: str = "none") -> Dataset:
    """Return a copy with transformed X.

    method: auto (arcsinh for proteomics, depth-normalise+log1p for RNA) | arcsinh | log1p | none
    normalize: none | zscore | minmax | robust (per-column, "column auto-normalisation")
    batch_correct: none | median (per-image median centring; harmony needs scanpy, not wired)
    """
    X = ds.X.to_numpy(np.float32, copy=True)
    if method == "auto":
        method = "arcsinh" if ds.modality in PROTEIN else "log1p"
    if method == "arcsinh":
        X = np.arcsinh(X / cofactor)
    elif method == "log1p":
        if ds.modality in RNA:
            X = X / np.maximum(X.sum(1, keepdims=True), 1e-9) * 1e4
        X = np.log1p(np.clip(X, 0, None))
    elif method != "none":
        raise ValueError(f"unknown transform '{method}'")

    if batch_correct == "median":
        if ds.groups is None:
            raise ValueError("batch_correct=median needs an image/sample column")
        glob = np.median(X, 0)
        for g in np.unique(ds.groups):
            m = ds.groups == g
            X[m] += glob - np.median(X[m], 0)
    elif batch_correct != "none":
        raise ValueError(f"unknown batch correction '{batch_correct}'")

    if normalize == "zscore":
        X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    elif normalize == "minmax":
        lo, hi = X.min(0), X.max(0)
        X = (X - lo) / np.where(hi - lo == 0, 1, hi - lo)
    elif normalize == "robust":
        med = np.median(X, 0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X = (X - med) / np.where(iqr == 0, 1, iqr)
    elif normalize != "none":
        raise ValueError(f"unknown normalization '{normalize}'")

    return Dataset(ds.name, pd.DataFrame(X, columns=ds.markers), ds.markers, ds.y, ds.xy, ds.groups,
                   ds.modality, ds.meta)

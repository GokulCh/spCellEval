"""Marker-prior helpers and a plain marker-score annotator.

TACIT / Scyan / Astir / Tribus are NOT implemented here: they run the repository's own scripts (``scripts.py``)
on the decision matrices the repo already ships (``methods/scyan``, ``methods/TACIT``, ``methods/astir``,
``methods/tribus``). What lives here is only what the repo lacks: locating those files, converting a matrix to
Astir's YAML when no bundled one exists, and the marker score used to make pseudo-labels in unsupervised mode.

Matrix format (repo's): first column = cell type, one column per marker, +1 positive / -1 negative.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..repo import METHODS
from .base import MethodUnavailable, Result, Task, register


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def load_marker_matrix(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0).apply(pd.to_numeric, errors="coerce").fillna(0)


def find_marker_matrix(dataset: str, level: str = "level3", prefer: str = "scyan") -> Path | None:
    """The repo's bundled decision matrix (CSV) for a dataset, looking in ``prefer`` first (scyan | TACIT)."""
    for d in (prefer, *[x for x in ("scyan", "TACIT") if x != prefer]):
        hits = sorted((METHODS / d).glob(f"{dataset}_decision_matrix_{level}*.csv"), key=lambda p: ("merged" in p.name, p.name))
        hits = hits or sorted((METHODS / d).glob(f"{dataset.lower()}_decision_matrix_{level}*.csv"))
        if hits:
            return hits[0]
    return None


def find_bundled(dataset: str, folder: str, pattern: str) -> Path | None:
    """Case-insensitive lookup of e.g. astir/cell_types_<ds>.yml or tribus/logic_table_<ds>_level3.xlsx."""
    want = pattern.format(ds=dataset).lower()
    return next((p for p in (METHODS / folder).iterdir() if p.name.lower() == want), None)


def to_astir_yaml(matrix: pd.DataFrame, path: str | Path) -> Path:
    """Astir marker file (``cell_types: {Name: [markers]}``, positive markers only) from a decision matrix."""
    lines = ["cell_types:"]
    for ct, row in matrix.iterrows():
        pos = [c for c, v in row.items() if v > 0]
        if pos:
            lines += [f"  {ct}:"] + [f"    - {m}" for m in pos]
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)


def align(matrix: pd.DataFrame, markers: list[str]) -> tuple[list[str], np.ndarray]:
    """Return (cell types, M[types x markers]) with the matrix columns matched to marker names."""
    col = {_key(c): c for c in matrix.columns}
    M = np.zeros((len(matrix), len(markers)))
    hits = 0
    for j, m in enumerate(markers):
        if _key(m) in col:
            M[:, j] = matrix[col[_key(m)]].to_numpy(float)
            hits += 1
    if hits == 0:
        raise MethodUnavailable("no decision-matrix column matches the dataset's markers")
    keep = np.abs(M).sum(1) > 0
    return list(matrix.index[keep]), M[keep]


def marker_scores(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Cells x types: mean z-score of positive markers minus that of negative markers."""
    Z = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    return Z @ M.T / np.maximum(np.abs(M).sum(1), 1)


def pseudo_labels(X, markers, matrix, min_score: float = 0.5, min_margin: float = 0.25):
    """Marker-based pseudo-labels. Returns (labels, confident_mask)."""
    types, M = align(matrix, markers)
    S = marker_scores(X, M)
    top2 = np.sort(S, 1)[:, -2:] if S.shape[1] > 1 else np.c_[np.full(len(S), -np.inf), S[:, 0]]
    best, margin = top2[:, 1], top2[:, 1] - top2[:, 0]
    labels = np.array(types, dtype=object)[S.argmax(1)]
    ok = (best >= min_score) & (margin >= min_margin)
    return np.where(best > 0, labels, "Unknown").astype(str), ok


@register("marker_score", 2, "prior", ())
def marker_score(t: Task) -> Result:
    """Plain marker-score annotation (argmax of signed marker z-scores; 'Unknown' if no type scores > 0).

    Lightweight stand-in that needs no R/torch; it is NOT TACIT (run ``tacit`` for the real R implementation).
    """
    if t.marker_matrix is None:
        raise MethodUnavailable("needs a marker decision matrix (--marker-matrix or a bundled <dataset> matrix)")
    types, M = align(t.marker_matrix, t.markers)
    S = marker_scores(t.X_test, M)
    lab = np.array(types, dtype=object)[S.argmax(1)]
    return Result(np.where(S.max(1) > 0, lab, "Unknown").astype(str))


PRIOR_METHODS = {"tacit", "scyan", "astir", "marker_score"}      # the methods that need a marker decision matrix
AUTO_NOTE = "auto-draft from the dataset's own labels (circular: not independent prior knowledge)"


def draft_matrix(X, markers, y, top: int = 3, min_z: float = 1.0) -> pd.DataFrame:
    """Decision matrix in the repo's format drafted from labelled data: +1 for each type's ``top`` markers whose mean
    z-score is >= ``min_z``. Derived from the labels, so it is NOT independent prior knowledge."""
    X = np.asarray(X, float)
    z = pd.DataFrame((X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0)), columns=list(markers)).groupby(np.asarray(y)).mean()
    m = pd.DataFrame(np.nan, index=pd.Index(sorted(z.index), name="Populations"), columns=list(markers))
    for t in m.index:
        best = z.loc[t].sort_values(ascending=False)
        for mk in best[best >= min_z].index[:top]:
            m.loc[t, mk] = 1.0
    return m


def clean_marker_name(col: str) -> str:
    """The cleaning rule of datasets/process_crc_codex.py: 'CD31 - vasculature:Cyc_19_ch_3' -> 'CD31'."""
    return re.sub(r"[^A-Za-z0-9]", "", str(col).split(" - ")[0].split(":")[0])


def conform_matrix(matrix: pd.DataFrame, markers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Rename the matrix's marker columns to the dataset's exact marker names (tolerating case / punctuation and the raw
    '<marker> - <description>:Cyc_<n>_ch_<n>' names used in published signature files). Columns that match no marker
    are dropped and returned. Columns that map to the same marker are merged (max)."""
    target = {_key(m): m for m in markers}
    rename, dropped = {}, []
    for c in matrix.columns:
        hit = next((target[_key(x)] for x in (c, clean_marker_name(c)) if _key(x) in target), None)
        if hit is None:
            dropped.append(str(c))
        else:
            rename[c] = hit
    m = matrix[list(rename)].rename(columns=rename)
    if m.columns.duplicated().any():
        m = m.T.groupby(level=0, sort=False).max().T
    return m, dropped

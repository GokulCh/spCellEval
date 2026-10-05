r"""Materialise a dataset in the benchmark's own on-disk layout and create folds with the repo's code.

    <root>/datasets/<name>/quantification/processed/
        <name>_quantification.csv                  markers first, then metadata (spc_row, image, x, y, labels ...)
        kfolds_<method>_<level>/fold_<i>_{train,validation,test}.csv     \  written by
        labels_<method>_<level>.csv                                       /  methods/utils/run_kfold_creator.py

Folds, label encoding and the validation split are all produced by the repository's ``run_fold_creation``
(``DataSetHandler``); nothing here re-implements them. Every row carries ``spc_row`` (its position in the
loaded dataset) so predictions written by any script can be mapped back to the in-memory matrix.

Splits store. The folds can live permanently next to the data, in the repo's own ``<main_dir>/datasets/<name>/
quantification/processed/`` (``store``). Only the fold directory, the labels file and a small fingerprint file are
written there - the user's own ``<name>_quantification.csv`` is never touched. A run first looks for matching folds
in the store (data values, labels, seed, fold method, fold count and validation fraction must all match) and links
them; otherwise it creates the folds and publishes them to the store for next time.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..repo import UTILS, load_module
from .data import LEVEL_COLUMN, Dataset

LEVELS = ["level3", "level2", "level1"]


@dataclass
class Workspace:
    root: Path
    name: str
    level: str
    kfold_method: str
    markers: list[str]
    n_rows: int
    has_folds: bool
    reused: bool = False                      # the folds came from the splits store instead of being created

    @property
    def dataset_dir(self) -> Path:            # what run_classic_ml_default.py calls --dataset_path
        return self.root / "datasets" / self.name

    @property
    def proc(self) -> Path:
        return self.dataset_dir / "quantification" / "processed"

    @property
    def quant(self) -> Path:
        return self.proc / f"{self.name}_quantification.csv"

    @property
    def kdir(self) -> Path:
        return self.proc / f"kfolds_{self.kfold_method}_{self.level}"

    @property
    def labels(self) -> Path:
        return self.proc / f"labels_{self.kfold_method}_{self.level}.csv"

    @property
    def fingerprint_file(self) -> Path:
        return self.proc / f"spc_fingerprint_{self.kfold_method}_{self.level}.json"

    @property
    def split_col(self) -> str:               # first non-marker column (scyan / TACIT / starling convention)
        return "spc_row"

    @property
    def last_marker(self) -> str:             # astir convention
        return self.markers[-1]

    def folds(self) -> list[dict]:
        """[{train, validation, test}] as arrays of spc_row ids, read from the files the repo code wrote."""
        out = []
        for i in range(1, 100):
            f = {k: self.kdir / f"fold_{i}_{k}.csv" for k in ("train", "validation", "test")}
            if not f["train"].exists():
                break
            out.append({k: pd.read_csv(p, usecols=["spc_row"]).spc_row.to_numpy() for k, p in f.items() if p.exists()})
        return out


# ----------------------------------------------------------------------------- splits store
def infer_main_dir(data_path: str | Path) -> Path | None:
    """<main_dir> if the file sits in the repo's ``<main_dir>/datasets/<name>/quantification/processed/`` layout."""
    p = Path(data_path).resolve()
    parts = p.parent.parts
    if len(parts) >= 5 and parts[-1] == "processed" and parts[-2] == "quantification" and parts[-4] == "datasets":
        return p.parents[4]
    return None


def resolve_store(splits_dir: str | None, data_path: str | Path, ds_name: str, fallback_root: str | Path | None = None) -> Path | None:
    """Where the folds should live.

    splits_dir: None/'auto' -> next to the data if it is in the repo layout, else a ``preprocess`` run's workspace under
                ``fallback_root/<ds>/workspace`` if one exists; 'out' -> nowhere (folds stay inside the output folder);
                a path -> that folder (a <main_dir>, or an older ``preprocessed`` folder holding <ds>/workspace).
    """
    if splits_dir == "out":
        return None
    if splits_dir and splits_dir != "auto":
        p = Path(splits_dir).resolve()
        return p / ds_name / "workspace" if (p / ds_name / "workspace" / "datasets").is_dir() else p
    m = infer_main_dir(data_path)
    if m is not None:
        return m
    if fallback_root is not None and (Path(fallback_root) / ds_name / "workspace" / "datasets").is_dir():
        return (Path(fallback_root) / ds_name / "workspace").resolve()
    return None


def _link(src: Path, dst: Path) -> None:
    """Symlink (cheap, read-only use) with a copy fallback where symlinks are unavailable."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def _link_tree(src: Path, dst: Path) -> None:
    for f in src.rglob("*"):
        if f.is_file():
            _link(f, dst / f.relative_to(src))


def fingerprint(ds: Dataset, labels, rows, level, kfold_method, seed, n_splits, val_pct) -> dict:
    """Everything that determines the folds: data values, labels and the split settings."""
    X = ds.X.iloc[rows]
    h = lambda a: hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
    return dict(n_rows=int(len(rows)), markers=list(ds.markers), level=level, kfold_method=kfold_method, seed=int(seed),
                n_splits=int(n_splits), val_pct=float(val_pct),
                x_md5=h(pd.util.hash_pandas_object(X, index=False).values),
                labels_md5=h(pd.util.hash_array(np.asarray(labels)[rows].astype(str))),
                rows_md5=h(np.asarray(rows, dtype=np.int64)))


def _legacy_mismatch(src: Workspace, ds: Dataset, labels, rows, level: str, n_splits: int) -> str | None:
    """None if an older split directory without a fingerprint file provably belongs to this data, else the reason."""
    cols = list(ds.markers) + ["spc_row", LEVEL_COLUMN[level]]
    try:
        t = pd.read_csv(src.quant, usecols=cols)
    except ValueError:
        return "no fingerprint and the stored table has no spc_row column, so it cannot be verified"
    if len(t) != len(rows) or not np.array_equal(t.spc_row.to_numpy(), rows):
        return "different cells"
    if not np.array_equal(t[LEVEL_COLUMN[level]].astype(str).to_numpy(), np.asarray(labels)[rows].astype(str)):
        return "different labels"
    if not np.allclose(t[list(ds.markers)].to_numpy(float), ds.X.iloc[rows].to_numpy(float), rtol=1e-4, atol=1e-6):
        return "different marker values (another transform?)"
    folds = src.folds()
    if len(folds) != n_splits:
        return f"{len(folds)} folds, not {n_splits}"
    if not np.array_equal(np.sort(np.concatenate([f["test"] for f in folds])), np.sort(rows)):
        return "folds do not cover the cells exactly once"
    return None


def _try_reuse(ws: Workspace, st: Workspace, fp: dict, ds: Dataset, labels, rows, level, n_splits) -> bool:
    """Link matching folds from the store into the workspace. Prints why not when it cannot."""
    if not (st.kdir.exists() and st.labels.exists()):
        print(f"{ds.name}: no folds in {st.proc} yet; creating them")
        return False
    if st.fingerprint_file.exists():
        old = json.loads(st.fingerprint_file.read_text())
        diff = [k for k in fp if old.get(k) != fp[k]]
        if diff:
            print(f"{ds.name}: folds in {st.proc} do not match this run ({', '.join(diff)} differ); "
                  "creating new folds and replacing them")
            return False
        how = "data, labels and settings match"
    else:
        why = _legacy_mismatch(st, ds, labels, rows, level, n_splits)
        if why is not None:
            print(f"{ds.name}: older folds in {st.proc} cannot be reused ({why}); creating new folds and replacing them")
            return False
        how = f"older folds: table, labels and {n_splits} folds verified; the seed they were made with cannot be checked"
    _link_tree(st.kdir, ws.kdir)
    _link(st.labels, ws.labels)
    ws.reused = True
    print(f"{ds.name}: reusing the folds already created in {st.proc} ({how})")
    return True


def _publish(ws: Workspace, st: Workspace) -> None:
    """Move freshly created folds into the store and link them back into the workspace (no duplicate copies)."""
    st.proc.mkdir(parents=True, exist_ok=True)
    for src, dst in ((ws.kdir, st.kdir), (ws.labels, st.labels), (ws.fingerprint_file, st.fingerprint_file)):
        if dst.is_dir():
            shutil.rmtree(dst)
        elif dst.exists() or dst.is_symlink():
            dst.unlink()
        shutil.move(str(src), str(dst))
    _link_tree(st.kdir, ws.kdir)
    _link(st.labels, ws.labels)
    _link(st.fingerprint_file, ws.fingerprint_file)
    print(f"{ws.name}: folds saved in {st.kdir} (labels: {st.labels.name})")


# ----------------------------------------------------------------------------- workspace
def build_workspace(ds: Dataset, labels: np.ndarray | None, root: str | Path, level: str = "level3",
                    kfold_method: str = "StratifiedKFold", seed: int = 0, n_splits: int = 5,
                    val_pct: float = 0.15, rows: np.ndarray | None = None, make_folds: bool = True,
                    extra: dict[str, np.ndarray] | None = None, store: str | Path | None = None) -> Workspace:
    """Write the quantification table (+ folds via the repo's ``run_fold_creation``).

    labels: label at ``level`` for every cell (true labels, or marker pseudo-labels in unsupervised mode).
    rows:   positions of the cells to include (default all); ``extra`` adds metadata columns (full-length arrays).
    store:  a <main_dir> holding the splits (see module docstring): matching folds are linked from it, otherwise the
            new folds are published to it. Its own quantification table is never modified.
    """
    rows = np.arange(len(ds)) if rows is None else np.asarray(rows)
    ws = Workspace(Path(root).resolve(), ds.name, level, kfold_method, list(ds.markers), len(rows), make_folds and labels is not None)
    X = ds.X.iloc[rows].reset_index(drop=True)
    meta = ds.meta.iloc[rows].reset_index(drop=True).copy()
    meta = meta.drop(columns=[c for c in meta if c in ("spc_row", *LEVEL_COLUMN.values())], errors="ignore")
    keep_levels = {}
    if labels is not None:
        keep_levels[LEVEL_COLUMN[level]] = np.asarray(labels)[rows]
        if ds.y is not None and np.array_equal(np.asarray(labels), ds.y):      # true labels: also keep coarser levels
            from ..evaluation.analysis import level_labels
            for lv, y in level_labels(ds, level).items():
                keep_levels[LEVEL_COLUMN[lv]] = np.asarray(y)[rows]
    if "cell_id" not in meta:
        meta["cell_id"] = np.arange(len(rows))
    if ds.xy is not None and "x" not in meta:
        meta["x"], meta["y"] = ds.xy[rows, 0], ds.xy[rows, 1]
    if ds.groups is not None and "image" not in meta:
        meta["image"] = ds.groups[rows]
    for k, v in keep_levels.items():
        meta[k] = v
    for k, v in (extra or {}).items():
        meta[k] = np.asarray(v)[rows]
    df = pd.concat([X, pd.DataFrame({"spc_row": rows}), meta], axis=1)
    ws.proc.mkdir(parents=True, exist_ok=True)
    df.to_csv(ws.quant, index=False)
    if not ws.has_folds:
        return ws

    fp = fingerprint(ds, labels, rows, level, kfold_method, seed, n_splits, val_pct)
    st = None
    if store is not None and Path(store).resolve() != ws.root:
        st = Workspace(Path(store).resolve(), ds.name, level, kfold_method, list(ds.markers), len(rows), True)
        if _try_reuse(ws, st, fp, ds, labels, rows, level, n_splits):
            return ws
    run_fold_creation = load_module(UTILS / "run_kfold_creator.py", "run_kfold_creator").run_fold_creation
    run_fold_creation(str(ws.root), ws.name, dropna=False, impute_value=None,
                      phenotype_column=LEVEL_COLUMN[level],
                      batch_identifier_column="image" if "image" in df else None, drop_columns=None,
                      drop_non_numerical=False, n_splits=n_splits, method=kfold_method,
                      group_shuffle_split_size=0.5, swap_train_test=False, random_state=seed,
                      percentage_validation=val_pct)
    ws.fingerprint_file.write_text(json.dumps(fp))
    if st is not None:
        _publish(ws, st)
    return ws


def make_variant(ws: Workspace, name: str, fold: int = 0, train_rows: np.ndarray | None = None,
                 variants_root: str | Path | None = None) -> Path:
    """A single-fold copy of the dataset dir (for hold-out / progressive runs of the repo's classic-ML script).

    Returns the variant root to pass as ``--main_dir``. ``train_rows`` (spc_row ids) subsamples the fold's train file.
    ``variants_root`` (default ``<workspace>/v``) is where the variants are written.
    """
    base = Path(variants_root).resolve() if variants_root else ws.root / "v"
    vroot = base / name / "datasets" / ws.name               # short names: Windows MAX_PATH
    kd = vroot / "quantification" / "processed" / ws.kdir.name
    kd.mkdir(parents=True, exist_ok=True)
    shutil.copy(ws.labels, kd.parent / ws.labels.name)
    for part in ("train", "validation", "test"):
        src = ws.kdir / f"fold_{fold + 1}_{part}.csv"
        if part == "train" and train_rows is not None:
            t = pd.read_csv(src)
            t[t.spc_row.isin(train_rows)].to_csv(kd / "fold_1_train.csv", index=False)
        else:
            shutil.copy(src, kd / f"fold_1_{part}.csv")
    return vroot.parent.parent                 # the variant root, usable as --main_dir

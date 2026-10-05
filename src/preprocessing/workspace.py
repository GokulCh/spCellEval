r"""Materialise a dataset in the benchmark's own on-disk layout and create folds with the repo's code.

    <root>/datasets/<name>/quantification/processed/
        <name>_quantification.csv                  markers first, then metadata (spc_row, image, x, y, labels ...)
        kfolds_<method>_<level>/fold_<i>_{train,validation,test}.csv     \  written by
        labels_<method>_<level>.csv                                       /  methods/utils/run_kfold_creator.py

Folds, label encoding and the validation split are all produced by the repository's ``run_fold_creation``
(``DataSetHandler``); nothing here re-implements them. Every row carries ``spc_row`` (its position in the
loaded dataset) so predictions written by any script can be mapped back to the in-memory matrix.
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
    reused: bool = False                      # folds were linked from an earlier run (see build_workspace reuse_from)
    reuse_src: Path | None = None

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


def build_workspace(ds: Dataset, labels: np.ndarray | None, root: str | Path, level: str = "level3",
                    kfold_method: str = "StratifiedKFold", seed: int = 0, n_splits: int = 5,
                    val_pct: float = 0.15, rows: np.ndarray | None = None, make_folds: bool = True,
                    extra: dict[str, np.ndarray] | None = None, reuse_from: str | Path | None = None) -> Workspace:
    """Write the quantification table (+ folds via the repo's ``run_fold_creation``).

    labels: label at ``level`` for every cell (true labels, or marker pseudo-labels in unsupervised mode).
    rows:   positions of the cells to include (default all); ``extra`` adds metadata columns (full-length arrays).
    reuse_from: a workspace root from an earlier ``preprocess`` run. If its fingerprint (data, labels, seed, fold method,
                folds) matches, its table, folds and labels are linked here instead of being recreated.
    """
    rows = np.arange(len(ds)) if rows is None else np.asarray(rows)
    ws = Workspace(Path(root).resolve(), ds.name, level, kfold_method, list(ds.markers), len(rows), make_folds and labels is not None)
    fp = fingerprint(ds, labels, rows, level, kfold_method, seed, n_splits, val_pct) if ws.has_folds else None
    if reuse_from and ws.has_folds:
        src = Workspace(Path(reuse_from).resolve(), ds.name, level, kfold_method, list(ds.markers), len(rows), True)
        fpf = src.proc / "spc_fingerprint.json"
        if fpf.exists() and src.quant.exists() and src.kdir.exists() and src.labels.exists():
            old = json.loads(fpf.read_text())
            if old == fp:
                _link_tree(src.proc, ws.proc)
                ws.reused, ws.reuse_src = True, src.root
                print(f"{ds.name}: reusing the folds already created in {src.root} (data, labels and settings match)")
                return ws
            diff = [k for k in fp if old.get(k) != fp[k]]
            print(f"{ds.name}: splits in {src.root} do not match this run ({', '.join(diff)} differ); creating new folds")
        else:
            print(f"{ds.name}: no usable splits found in {src.root}; creating new folds")
    X = ds.X.iloc[rows].reset_index(drop=True)
    meta = ds.meta.iloc[rows].reset_index(drop=True).copy()
    meta = meta.drop(columns=[c for c in meta if c in ("spc_row", *LEVEL_COLUMN.values())], errors="ignore")
    keep_levels = {}
    if labels is not None:
        keep_levels[LEVEL_COLUMN[level]] = np.asarray(labels)[rows]
        if ds.y is not None and np.array_equal(np.asarray(labels), ds.y):      # true labels: also keep coarser levels
            from ..evaluation.repo_assets import to_level
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
    if ws.has_folds:
        run_fold_creation = load_module(UTILS / "run_kfold_creator.py", "run_kfold_creator").run_fold_creation
        run_fold_creation(str(ws.root), ws.name, dropna=False, impute_value=None,
                          phenotype_column=LEVEL_COLUMN[level],
                          batch_identifier_column="image" if "image" in df else None, drop_columns=None,
                          drop_non_numerical=False, n_splits=n_splits, method=kfold_method,
                          group_shuffle_split_size=0.5, swap_train_test=False, random_state=seed,
                          percentage_validation=val_pct)
        (ws.proc / "spc_fingerprint.json").write_text(json.dumps(fp))
    return ws


def make_variant(ws: Workspace, name: str, fold: int = 0, train_rows: np.ndarray | None = None) -> Path:
    """A single-fold copy of the dataset dir (for hold-out / progressive runs of the repo's classic-ML script).

    Returns the variant root to pass as ``--main_dir``. ``train_rows`` (spc_row ids) subsamples the fold's train file.
    """
    vroot = ws.root / "v" / name / "datasets" / ws.name      # short names: Windows MAX_PATH
    kd = vroot / "quantification" / "processed" / ws.kdir.name
    if ws.reuse_src is not None:                             # variant already written by the earlier preprocess run?
        done = ws.reuse_src / "v" / name
        if (done / "datasets" / ws.name / "quantification" / "processed" / ws.kdir.name / "fold_1_train.csv").exists():
            _link_tree(done, ws.root / "v" / name)
            return ws.root / "v" / name
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

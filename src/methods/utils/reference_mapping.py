"""Reference-based label transfer (SingleR-style correlation, scanpy ingest)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def _list_kfold_files(kdir: Path) -> Dict[str, List[str]]:
    fold_dict: Dict[str, List[str]] = {"train": [], "validation": [], "test": []}
    for file in os.listdir(kdir):
        if not file.endswith(".csv"):
            continue
        if "train" in file:
            fold_dict["train"].append(file)
        elif "validation" in file:
            fold_dict["validation"].append(file)
        elif "test" in file:
            fold_dict["test"].append(file)
    for key in fold_dict:
        fold_dict[key].sort()
    if len(fold_dict["train"]) != len(fold_dict["test"]):
        raise ValueError("Train and test fold counts do not match.")
    return fold_dict


def build_reference_profiles(
    train_df: pd.DataFrame,
    markers: Sequence[str],
    label_col: str = "encoded_phenotype",
) -> pd.DataFrame:
    """Mean marker expression per reference label."""
    return train_df.groupby(label_col, observed=True)[list(markers)].mean()


def singler_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    markers: Sequence[str],
    label_dict: Dict[int, str],
    label_col: str = "encoded_phenotype",
) -> np.ndarray:
    """Assign each test cell to the reference type with highest Pearson correlation."""
    profiles = build_reference_profiles(train_df, markers, label_col=label_col)
    test_matrix = test_df[list(markers)].to_numpy(dtype=float)
    profile_matrix = profiles.to_numpy(dtype=float)
    labels = profiles.index.to_numpy()

    predictions: List[str] = []
    for row in test_matrix:
        best_label = label_dict[int(labels[0])]
        best_corr = -2.0
        for idx, profile in enumerate(profile_matrix):
            corr, _ = pearsonr(row, profile)
            if np.isnan(corr):
                corr = -1.0
            if corr > best_corr:
                best_corr = corr
                best_label = label_dict[int(labels[idx])]
        predictions.append(best_label)
    return np.array(predictions, dtype=object)


def _pearson_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized Pearson correlations between rows of ``a`` and rows of ``b``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a0 = a - a.mean(axis=1, keepdims=True)
    b0 = b - b.mean(axis=1, keepdims=True)
    an = np.linalg.norm(a0, axis=1, keepdims=True)
    bn = np.linalg.norm(b0, axis=1, keepdims=True)
    sim = (a0 @ b0.T) / (an @ bn.T + 1e-12)
    sim[~np.isfinite(sim)] = -1.0
    return sim


def hungarian_capacity_assignment(
    cost_matrix: np.ndarray,
    capacities: Sequence[int],
) -> np.ndarray:
    """Assign each row to a column under per-column capacity (Hungarian/Munkres).

    Columns are repeated according to their capacity so that
    :func:`scipy.optimize.linear_sum_assignment` solves the linear assignment
    with quotas. Excess capacity (or rows beyond capacity) are clipped, and the
    returned array maps every row to a column index.
    """
    from scipy.optimize import linear_sum_assignment

    cost_matrix = np.asarray(cost_matrix, dtype=float)
    n_rows, n_types = cost_matrix.shape
    total_cap = int(sum(capacities))
    if n_rows == 0 or total_cap == 0:
        return np.zeros(n_rows, dtype=int)

    repeat = max(1, int(np.ceil(n_rows / total_cap)))
    duplicated_cols = np.repeat(np.arange(n_types, dtype=int), [int(c) * repeat for c in capacities])
    duplicated_cols = duplicated_cols[:n_rows]
    sub_cost = cost_matrix[:, duplicated_cols]
    row_idx, col_idx = linear_sum_assignment(sub_cost)
    return duplicated_cols[col_idx]


def ribca_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    markers: Sequence[str],
    label_dict: Dict[int, str],
    label_col: str = "encoded_phenotype",
    *,
    min_corr: Optional[float] = None,
    use_hungarian: bool = True,
    hungarian_max_cells: int = 20000,
) -> np.ndarray:
    """RIBCA_adapted — reference-guided cell typing for TMA benchmarking.

    Pipeline (adapted from RIBCA's reference-informed ideas):
      1. Build per-cell-type mean expression reference profiles on training data.
      2. Score every test cell against all profiles via vectorized Pearson
         correlation.
      3. Optionally threshold low-confidence ``min_corr`` cells (they revert to
         their nearest reference type).
      4. Refine the hard assignment with a capacity-constrained Hungarian joint
         assignment (Munkres) to keep predicted type frequencies coherent with
         the correlation evidence. The Hungarian step is only used when the test
         matrix is small enough to keep the joint assignment tractable.
    """
    profiles = build_reference_profiles(train_df, markers, label_col=label_col)
    test_matrix = test_df[list(markers)].to_numpy(dtype=float)
    profile_matrix = profiles.to_numpy(dtype=float)
    labels = profiles.index.to_numpy()
    n_types = len(labels)

    sim = _pearson_sim_matrix(test_matrix, profile_matrix)
    if min_corr is not None:
        below = sim < min_corr
        if below.any():
            sim[below] = np.min(sim, axis=1, keepdims=True)[below]

    if use_hungarian and len(test_matrix) <= hungarian_max_cells and n_types <= 32:
        hard = sim.argmax(axis=1)
        capacities = np.bincount(hard, minlength=n_types).tolist()
        assignments = hungarian_capacity_assignment(-sim, capacities)
    else:
        assignments = sim.argmax(axis=1)

    return np.array([label_dict[int(labels[i])] for i in assignments], dtype=object)


def scarches_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    markers: Sequence[str],
    label_dict: Dict[int, str],
    label_col: str = "encoded_phenotype",
) -> np.ndarray:
    """Label transfer via scanpy ``ingest`` (scArches-compatible reference mapping)."""
    try:
        import anndata as ad
        import scanpy as sc

        train_obs = train_df.copy()
        train_obs["reference_label"] = train_obs[label_col].map(label_dict)

        ref = ad.AnnData(
            X=train_df[list(markers)].to_numpy(dtype=float),
            obs=train_obs[["reference_label"]],
        )
        query = ad.AnnData(X=test_df[list(markers)].to_numpy(dtype=float))

        n_pcs = min(30, len(markers) - 1, ref.n_obs - 1)
        if n_pcs < 1:
            return singler_predict(train_df, test_df, markers, label_dict, label_col=label_col)

        sc.pp.pca(ref, n_comps=n_pcs)
        sc.pp.neighbors(ref, n_neighbors=min(15, ref.n_obs - 1))
        if ref.n_obs >= 5:
            sc.tl.umap(ref)
        sc.tl.ingest(query, ref, obs="reference_label")
        return query.obs["reference_label"].to_numpy(dtype=object)
    except Exception:
        return singler_predict(train_df, test_df, markers, label_dict, label_col=label_col)


def run_kfold_label_transfer(
    kdir: Path,
    labels_path: Path,
    output_dir: Path,
    markers: Sequence[str],
    predict_fn: Callable[..., np.ndarray],
    dumb_columns: Optional[Sequence[str]] = None,
) -> Path:
    """Run reference mapping across saved k-folds and write ``predictions_*.csv``."""
    fold_dict = _list_kfold_files(kdir)
    labels = pd.read_csv(labels_path)
    label_dict = dict(zip(labels["label"], labels["phenotype"]))
    drop_cols = set(dumb_columns or [])

    output_dir.mkdir(parents=True, exist_ok=True)
    times_path = output_dir / "fold_times.txt"

    for fold_idx, (train_file, test_file) in enumerate(
        zip(fold_dict["train"], fold_dict["test"]), start=1
    ):
        train_df = pd.read_csv(kdir / train_file)
        test_df = pd.read_csv(kdir / test_file)

        feature_cols = [
            c
            for c in train_df.columns
            if c != "encoded_phenotype" and c not in drop_cols
        ]
        use_markers = [m for m in markers if m in feature_cols]
        if not use_markers:
            raise ValueError(f"No marker columns found in fold data under {kdir}")

        import time

        start = time.time()
        preds = predict_fn(train_df, test_df, use_markers, label_dict)
        elapsed = time.time() - start

        out = test_df.copy()
        out["predicted_phenotype"] = preds
        out["true_phenotype"] = out["encoded_phenotype"].map(label_dict)
        out = out.drop(columns=["encoded_phenotype"])
        out.to_csv(output_dir / f"predictions_{fold_idx}.csv", index=False)

        mode = "w" if fold_idx == 1 else "a"
        with open(times_path, mode, encoding="utf-8") as fh:
            fh.write(f"Fold {fold_idx} train_time, {elapsed:.2f}\n")

    return output_dir

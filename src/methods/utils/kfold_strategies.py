"""
kfold_strategies.py
===================
K-fold split strategies and cell-type representation helpers.

Supported split methods (``createFolds`` in ``data_handler.py``):

* **StratifiedKFold** — standard stratified cell-level CV (default).
* **ProgressiveKFold** — same test folds as stratified CV, but each fold
  trains on a progressively larger stratified subset of the training pool
  (fold 1 → 1/k of train pool, fold k → full train pool).
* **StratifiedGroupKFold** / **GroupShuffleSplit** — group-aware variants.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

KFOLD_METHODS = (
    "StratifiedKFold",
    "ProgressiveKFold",
    "StratifiedGroupKFold",
    "GroupShuffleSplit",
)

# Default supervised benchmarking runs both standard and progressive CV.
DEFAULT_SUPERVISED_KFOLD_METHODS = ("StratifiedKFold", "ProgressiveKFold")


def progressive_kfold_splits(
    y: np.ndarray,
    n_splits: int,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build progressive training folds with fixed stratified test partitions.

    For fold *i* (1-indexed), the training set contains a stratified
    ``i / n_splits`` fraction of the non-test cells.  Test indices match
    standard :class:`~sklearn.model_selection.StratifiedKFold` splits so
    results are comparable across methods.
    """
    if n_splits < 2:
        raise ValueError("ProgressiveKFold requires n_splits >= 2")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []

    for fold_i, (train_pool_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        fraction = (fold_i + 1) / n_splits
        pool_y = y[train_pool_idx]

        if fraction >= 1.0:
            train_idx = train_pool_idx
        else:
            # Stratified subsample of the training pool.
            try:
                train_idx, _ = train_test_split(
                    train_pool_idx,
                    train_size=fraction,
                    stratify=pool_y,
                    random_state=random_state + fold_i,
                )
            except ValueError:
                # Too few samples per class for stratify — fall back to random.
                rng = np.random.RandomState(random_state + fold_i)
                n_take = max(1, int(round(len(train_pool_idx) * fraction)))
                train_idx = rng.choice(train_pool_idx, size=n_take, replace=False)

        folds.append((np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)))

    return folds


def stratified_80_20_split(
    y: np.ndarray,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Objective 1 — baseline stratified 80% train / 20% test single split.

    Ratios are stratifed on the cell-type labels so every phenotype keeps its
    population share in both partitions (following the same ``train_test_split``
    convention used elsewhere in this package).
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be a float in (0, 1).")
    try:
        train_idx, test_idx = train_test_split(
            np.arange(len(y)),
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )
    except ValueError:
        # A class too small to stratify — fall back to a shuffled split.
        rng = np.random.RandomState(random_state)
        idx = rng.permutation(len(y))
        n_test = int(round(len(y) * test_size))
        test_idx, train_idx = idx[:n_test], idx[n_test:]
    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


# Objective 4 — training-amount efficiency sweep (fractions of the 80% train pool).
SUBSAMPLING_FRACTIONS: Tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80)


def subsample_stratified_indices(
    y: np.ndarray,
    train_idx: Sequence[int],
    fraction: float,
    random_state: int = 42,
) -> np.ndarray:
    """Stratified subsample of the (80%) training pool for a given fraction.

    Keeps a fixed 20% holdout untouched; only the training pool is shrunk, so
    test metrics remain comparable across the fractions sweep.
    """
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError(f"fraction must be in (0, 1]; got {fraction}.")
    pool = np.asarray(train_idx, dtype=int)
    if fraction >= 1.0:
        return pool
    pool_y = y[pool]
    try:
        sub, _ = train_test_split(
            pool,
            train_size=fraction,
            stratify=pool_y,
            random_state=random_state,
        )
    except ValueError:
        rng = np.random.RandomState(random_state)
        n_take = max(1, int(round(len(pool) * fraction)))
        sub = rng.choice(pool, size=n_take, replace=False)
    return np.asarray(sub, dtype=int)


def classify_frequency_tier(
    fraction: float,
    *,
    rare_fraction: float = 0.01,
    common_fraction: float = 0.05,
) -> str:
    """Label a phenotype as rare, intermediate, or common by dataset fraction."""
    if fraction < rare_fraction:
        return "rare"
    if fraction >= common_fraction:
        return "common"
    return "intermediate"


def classify_abundance_tier(
    fraction: float,
    *,
    abundant_fraction: float = 0.05,
    rare_fraction: float = 0.01,
) -> str:
    """Objective 2 — abundance tiers exactly as specified:
    ``abundant`` = >5% of total cells, ``rare`` = <=1%, else ``intermediate``.
    """
    if fraction > abundant_fraction:
        return "abundant"
    if fraction <= rare_fraction:
        return "rare"
    return "intermediate"


def build_cell_type_representation(
    labels: pd.DataFrame,
    y_encoded: np.ndarray,
    *,
    rare_fraction: float = 0.01,
    common_fraction: float = 0.05,
) -> pd.DataFrame:
    """Rank phenotypes by abundance and assign frequency tiers.

    Parameters
    ----------
    labels:
        DataFrame with columns ``label`` (int code) and ``phenotype`` (name).
    y_encoded:
        Encoded phenotype array aligned with the quantification table.
    """
    label_to_name = dict(zip(labels["label"], labels["phenotype"]))
    counts = pd.Series(y_encoded).value_counts().sort_values(ascending=False)
    total = int(counts.sum()) if len(counts) else 0

    rows = []
    for rank, (code, count) in enumerate(counts.items(), start=1):
        fraction = float(count) / total if total else 0.0
        phenotype = label_to_name.get(int(code), str(code))
        rows.append({
            "rank": rank,
            "label": int(code),
            "phenotype": phenotype,
            "count": int(count),
            "fraction": fraction,
            "percent": fraction * 100.0,
            "frequency_tier": classify_frequency_tier(
                fraction,
                rare_fraction=rare_fraction,
                common_fraction=common_fraction,
            ),
            "is_most_common": rank == 1,
            "is_rare": fraction < rare_fraction,
        })

    return pd.DataFrame(rows)


def fold_representation_table(
    y_encoded: np.ndarray,
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    labels: pd.DataFrame,
    *,
    fold: int,
    rare_fraction: float = 0.01,
    common_fraction: float = 0.05,
) -> pd.DataFrame:
    """Per-fold train/test counts and fractions for every phenotype."""
    label_to_name = dict(zip(labels["label"], labels["phenotype"]))
    global_counts = pd.Series(y_encoded).value_counts()
    total = int(global_counts.sum()) if len(global_counts) else 0

    rows = []
    for code in sorted(label_to_name):
        phenotype = label_to_name[code]
        global_count = int(global_counts.get(code, 0))
        train_count = int(np.sum(y_encoded[train_idx] == code))
        test_count = int(np.sum(y_encoded[test_idx] == code))
        global_fraction = global_count / total if total else 0.0
        rows.append({
            "fold": fold,
            "label": int(code),
            "phenotype": phenotype,
            "global_count": global_count,
            "global_fraction": global_fraction,
            "frequency_tier": classify_frequency_tier(
                global_fraction,
                rare_fraction=rare_fraction,
                common_fraction=common_fraction,
            ),
            "train_count": train_count,
            "test_count": test_count,
            "train_fraction_of_global": train_count / global_count if global_count else 0.0,
            "test_fraction_of_global": test_count / global_count if global_count else 0.0,
        })

    return pd.DataFrame(rows)


def frequency_tiers_from_series(
    y_true: pd.Series,
    *,
    rare_fraction: float = 0.01,
    common_fraction: float = 0.05,
) -> Dict[str, str]:
    """Map each observed phenotype to rare / intermediate / common."""
    counts = y_true.dropna().astype(str).value_counts(normalize=True)
    return {
        label: classify_frequency_tier(
            float(frac),
            rare_fraction=rare_fraction,
            common_fraction=common_fraction,
        )
        for label, frac in counts.items()
    }


def result_method_id(method_id: str, kfold_method: str) -> str:
    """Map a base method id + k-fold strategy to a results directory name."""
    if kfold_method == "StratifiedKFold":
        return method_id
    if kfold_method == "ProgressiveKFold":
        return f"{method_id}_progressive"
    slug = kfold_method.replace("KFold", "").lower()
    return f"{method_id}_{slug}"


def infer_kfold_strategy_from_result_method(method_id: str) -> str:
    """Recover the k-fold strategy from a results method folder name."""
    if method_id.endswith("_progressive"):
        return "ProgressiveKFold"
    return "StratifiedKFold"

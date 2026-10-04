"""Progressive training-set subsampling.

Hold-out (80/20) and 5-fold CV come from the repository's ``DataSetHandler`` (see ``workspace.py``): one fold of a
stratified 5-fold split *is* a stratified 80/20 split. The repo has no equivalent for the data-scarcity sweep
(only the ``training_amount/xgb_amount.ipynb`` experiment), so subsampling lives here.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

DEFAULT_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.80)


def _strat(y: np.ndarray, min_count: int = 2) -> np.ndarray:
    """Merge classes with < min_count cells into one bucket so stratification never errors."""
    vals, inv, cnt = np.unique(y, return_inverse=True, return_counts=True)
    s = np.where(cnt[inv] < min_count, "__rare__", y)
    if (s == "__rare__").sum() == 1:     # a lone bucket member cannot be stratified either
        s = np.where(s == "__rare__", vals[cnt.argmax()], s)
    return s


def progressive(train_idx, y, n_total: int, fractions=DEFAULT_FRACTIONS, seed: int = 0):
    """Return [(fraction, subset)]: fraction*n_total training cells, class-stratified.

    A fraction at/above the available training size returns the whole training split.
    """
    train_idx = np.asarray(train_idx)
    out = []
    for f in sorted(fractions):
        n = max(int(round(f * n_total)), 2)
        if n >= len(train_idx):
            out.append((f, train_idx))
            continue
        try:
            sub, _ = train_test_split(train_idx, train_size=n, random_state=seed,
                                      stratify=_strat(np.asarray(y)[train_idx]))
        except ValueError:
            sub = np.random.default_rng(seed).choice(train_idx, n, replace=False)
        out.append((f, np.sort(sub)))
    return out

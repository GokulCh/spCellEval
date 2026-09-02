"""
parallel.py
===========
Shared helpers for bounded parallel execution across pipeline stages.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, List, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def resolve_worker_count(requested: Optional[int] = None, *, cap: int = 8) -> int:
    """Return a safe worker count for CPU-bound / mixed parallel tasks."""
    cpu = os.cpu_count() or 1
    if requested is None or requested <= 0:
        return max(1, min(cap, max(1, cpu - 1)))
    return max(1, min(int(requested), cpu))


def resolve_ml_n_jobs(parallel_workers: int) -> int:
    """Cap per-model sklearn/xgboost threads when running methods in parallel."""
    cpu = os.cpu_count() or 1
    if parallel_workers <= 1:
        return -1
    return max(1, cpu // parallel_workers)


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    *,
    max_workers: int = 1,
    description: str = "tasks",
) -> List[R]:
    """Run *func* over *items* with a thread pool (I/O-heavy or subprocess work)."""
    work = list(items)
    if not work:
        return []
    if max_workers <= 1 or len(work) == 1:
        return [func(item) for item in work]

    results: List[Optional[R]] = [None] * len(work)
    with ThreadPoolExecutor(max_workers=min(max_workers, len(work))) as pool:
        future_to_idx = {pool.submit(func, item): idx for idx, item in enumerate(work)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
    return [r for r in results if r is not None]

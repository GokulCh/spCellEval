"""Tests for parallel execution helpers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "utils"))

from parallel import parallel_map, resolve_ml_n_jobs, resolve_worker_count


def test_resolve_worker_count_auto():
    import os

    cpu = os.cpu_count() or 1
    assert 1 <= resolve_worker_count(0) <= max(1, cpu)


def test_resolve_worker_count_explicit():
    assert resolve_worker_count(2) == 2


def test_resolve_ml_n_jobs_scales_with_parallelism():
    assert resolve_ml_n_jobs(1) == -1
    assert resolve_ml_n_jobs(4) >= 1


def test_parallel_map_preserves_order():
    out = parallel_map(lambda x: x * 2, [1, 2, 3], max_workers=2)
    assert out == [2, 4, 6]

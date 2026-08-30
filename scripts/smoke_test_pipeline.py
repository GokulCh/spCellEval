#!/usr/bin/env python3
"""
smoke_test_pipeline.py
======================
Fast end-to-end check (~2–8 min) that mirrors::

    python src/run_pipeline.py --dataset CRC_TMA --recreate_kfolds

Uses a 300-cell CRC_TMA fixture (``tests/fixtures/smoke/``) and a trimmed
method list so wiring issues surface quickly without a multi-hour run.

Usage
-----
    python scripts/smoke_test_pipeline.py
    python scripts/smoke_test_pipeline.py --keep-results
    python scripts/smoke_test_pipeline.py --regenerate-fixture
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_FIXTURE_RAW = _REPO / "tests" / "fixtures" / "smoke" / "CRC_TMA_expression.csv"
_FULL_RAW = _REPO / "data" / "raw" / "CRC_TMA" / "CRC_TMA_expression.csv"
_PROCESSED = _REPO / "data" / "processed" / "CRC_TMA_SMOKE"
_RESULTS = _REPO / "results" / "CRC_TMA_SMOKE"


def _regenerate_fixture(n_per_label: int = 20, max_rows: int = 300) -> None:
    """Rebuild tiny raw CSV from full CRC_TMA (stratified by ClusterName)."""
    import pandas as pd

    if not _FULL_RAW.is_file():
        raise FileNotFoundError(
            f"Cannot regenerate fixture — full raw file missing:\n  {_FULL_RAW}"
        )

    df = pd.read_csv(_FULL_RAW)
    if "ClusterName" not in df.columns:
        tiny = df.head(max_rows)
    else:
        parts = [grp.head(n_per_label) for _, grp in df.groupby("ClusterName")]
        tiny = pd.concat(parts, ignore_index=True).head(max_rows)

    _FIXTURE_RAW.parent.mkdir(parents=True, exist_ok=True)
    tiny.to_csv(_FIXTURE_RAW, index=False)
    print(f"Wrote fixture: {_FIXTURE_RAW} ({len(tiny)} cells, {len(tiny.columns)} columns)")


def _clean_smoke_artifacts() -> None:
    for path in (_PROCESSED, _RESULTS):
        if path.is_dir():
            shutil.rmtree(path)
            print(f"Removed {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast CRC_TMA pipeline smoke test.")
    parser.add_argument(
        "--regenerate-fixture",
        action="store_true",
        help="Resample tests/fixtures/smoke/CRC_TMA_expression.csv from full raw data.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete data/processed/CRC_TMA_SMOKE and results/CRC_TMA_SMOKE before running.",
    )
    parser.add_argument(
        "--keep-results",
        action="store_true",
        help="Do not delete smoke outputs before run (default).",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Deprecated alias for default behavior (outputs are kept).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.regenerate_fixture:
        _regenerate_fixture()

    if not _FIXTURE_RAW.is_file():
        print(f"Fixture missing: {_FIXTURE_RAW}")
        print("Run with --regenerate-fixture (requires full CRC_TMA raw CSV).")
        return 1

    print("=" * 60)
    print("CRC_TMA smoke test (tiny dataset, fast methods)")
    print("Mirrors: run_pipeline.py --dataset CRC_TMA --recreate_kfolds")
    print("=" * 60)

    if args.fresh and not (args.keep_results or args.skip_clean):
        _clean_smoke_artifacts()

    cmd = [
        sys.executable,
        str(_REPO / "src" / "run_pipeline.py"),
        "--dataset", "CRC_TMA_SMOKE",
        "--recreate_kfolds",
        "--benchmark_config", "configs/benchmark_smoke.yaml",
        "--skip_viz",
    ]
    if args.verbose:
        cmd.append("-v")

    print("Command:")
    print(" ", " ".join(cmd))
    print("=" * 60)

    start = time.time()
    result = subprocess.run(cmd, cwd=str(_REPO))
    elapsed = time.time() - start

    summary = _REPO / "results" / "CRC_TMA_SMOKE" / "summary" / "final_results.csv"
    print("\n" + "=" * 60)
    if result.returncode == 0:
        print(f"SMOKE TEST PASSED in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
        if summary.is_file():
            print(f"  Summary: {summary}")
    else:
        print(f"SMOKE TEST FAILED (exit {result.returncode}) after {elapsed:.0f}s")
        print("  Fix errors above before running the full CRC_TMA pipeline.")
    print("=" * 60)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

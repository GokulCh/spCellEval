"""
performance.py
==============
Runtime and memory benchmarking from ``fold_times.txt`` and run manifests.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

_TIME_RE = re.compile(
    r"(?:Fold\s+)?(\d+)?\s*(train(?:ing)?|inference|prediction)[_\s]*time[:\s]+([0-9.]+)",
    re.IGNORECASE,
)
_SIMPLE_TIME_RE = re.compile(
    r"Fold\s+(\d+)\s+inference_time:\s+([0-9.]+)",
    re.IGNORECASE,
)


@dataclass
class PerformanceStats:
    train_time_mean: Optional[float] = None
    inference_time_mean: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    total_time_sec: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def parse_fold_times(path: Path) -> PerformanceStats:
    """Parse ``fold_times.txt`` written by method runners."""
    if not path.is_file():
        return PerformanceStats()

    train_times = []
    infer_times = []

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            m = _SIMPLE_TIME_RE.search(line)
            if m:
                infer_times.append(float(m.group(2)))
                continue
            m2 = _TIME_RE.search(line.replace(",", " "))
            if m2:
                val = float(m2.group(3))
                event = m2.group(2).lower()
                if "train" in event:
                    train_times.append(val)
                else:
                    infer_times.append(val)
            elif "inference_time:" in line.lower():
                try:
                    infer_times.append(float(line.split(":")[-1].strip().split()[0]))
                except ValueError:
                    pass

    return PerformanceStats(
        train_time_mean=float(sum(train_times) / len(train_times)) if train_times else None,
        inference_time_mean=float(sum(infer_times) / len(infer_times)) if infer_times else None,
    )


def load_run_manifest(path: Path) -> PerformanceStats:
    """Load ``benchmark_manifest.json`` written by Stage 3 runner."""
    if not path.is_file():
        return PerformanceStats()
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return PerformanceStats(
        train_time_mean=data.get("train_time_sec"),
        inference_time_mean=data.get("inference_time_sec"),
        peak_memory_mb=data.get("peak_memory_mb"),
        total_time_sec=data.get("elapsed_sec"),
    )


def find_performance_file(method_dir: Path) -> PerformanceStats:
    """Combine fold_times.txt and benchmark_manifest.json if present."""
    stats = parse_fold_times(method_dir / "fold_times.txt")
    manifest = load_run_manifest(method_dir / "benchmark_manifest.json")
    if manifest.total_time_sec is not None:
        stats.total_time_sec = manifest.total_time_sec
    if manifest.peak_memory_mb is not None:
        stats.peak_memory_mb = manifest.peak_memory_mb
    if stats.train_time_mean is None and manifest.train_time_mean is not None:
        stats.train_time_mean = manifest.train_time_mean
    if stats.inference_time_mean is None and manifest.inference_time_mean is not None:
        stats.inference_time_mean = manifest.inference_time_mean
    return stats


def write_run_manifest(
    path: Path,
    method_id: str,
    elapsed_sec: float,
    peak_memory_mb: Optional[float] = None,
) -> None:
    """Write performance metadata after a benchmark method run."""
    data: Dict = {
        "method": method_id,
        "elapsed_sec": round(elapsed_sec, 2),
    }
    if peak_memory_mb is not None:
        data["peak_memory_mb"] = round(peak_memory_mb, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

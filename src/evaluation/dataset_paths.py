"""Resolve dataset quant paths and marker columns for evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import yaml


def resolve_dataset_quant(
    dataset_name: str,
    root: Path,
    bench_config_path: Optional[Path] = None,
) -> Tuple[Optional[Path], List[str]]:
    """Return (quant_path, marker_columns) for a dataset."""
    bench_path = bench_config_path or (root / "configs" / "benchmark.yaml")
    ds_config_rel = f"configs/{dataset_name.lower()}.yaml"

    if bench_path.is_file():
        with bench_path.open("r", encoding="utf-8") as fh:
            bench = yaml.safe_load(fh) or {}
        ds_entry = bench.get("datasets", {}).get(dataset_name, {})
        ds_config_rel = ds_entry.get("config", ds_config_rel)

    ds_path = root / ds_config_rel
    if not ds_path.is_file():
        return None, []

    with ds_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    out = cfg.get("output", {})
    processed_dir = root / out.get("processed_dir", f"data/processed/{dataset_name}")
    filename = out.get("output_filename", f"{dataset_name}_quantification.csv")
    quant_path = processed_dir / filename

    markers = cfg.get("protein_markers", [])
    cleaned: List[str] = []
    for raw in markers:
        name = str(raw)
        if ":Cyc_" in name:
            name = name.rsplit(":", 1)[0]
        cleaned.append(name)
    return quant_path if quant_path.is_file() else None, cleaned

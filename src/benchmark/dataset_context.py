"""
dataset_context.py
====================
Resolves Stage 1 paths (processed CSV, k-folds, markers) from a dataset YAML config.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PSEUDO_DIR = _REPO_ROOT / "src" / "pseudo_labeling"
if str(_PSEUDO_DIR) not in sys.path:
    sys.path.insert(0, str(_PSEUDO_DIR))

from ground_truth import get_marker_columns, infer_separate_col  # noqa: E402


@dataclass
class DatasetContext:
    """Runtime paths and metadata for one benchmark dataset."""

    root: Path
    config_path: Path
    config: Dict[str, Any] = field(repr=False)
    dataset_name: str = ""
    quant_path: Path = field(default_factory=Path)
    processed_dir: Path = field(default_factory=Path)
    markers: List[str] = field(default_factory=list)
    split_col: str = "Image_ID"
    kfold_method: str = "StratifiedKFold"
    granularity: str = "level3"
    phenotype_column: str = "cell_type"
    batch_column: Optional[str] = None
    image_data_dir: Optional[Path] = None

    @classmethod
    def from_config(
        cls,
        config_path: Path,
        root: Optional[Path] = None,
        benchmark_overrides: Optional[Dict[str, Any]] = None,
    ) -> "DatasetContext":
        root = (root or _REPO_ROOT).resolve()
        config_path = config_path if config_path.is_absolute() else root / config_path
        with config_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        overrides = benchmark_overrides or {}
        out_cfg = cfg.get("output", {})
        processed_dir = root / out_cfg.get("processed_dir", f"data/processed/{cfg['dataset_name']}")
        filename = out_cfg.get("output_filename", f"{cfg['dataset_name']}_quantification.csv")

        batch_col = cfg.get("column_mappings", {}).get("batch_id")
        if batch_col == cfg.get("column_mappings", {}).get("image_id"):
            batch_col = "batch_id"  # standardised name in processed output

        img_dir = overrides.get("image_data_dir")
        return cls(
            root=root,
            config_path=config_path,
            config=cfg,
            dataset_name=cfg.get("dataset_name", config_path.stem),
            quant_path=processed_dir / filename,
            processed_dir=processed_dir,
            markers=get_marker_columns(cfg),
            split_col=infer_separate_col(cfg),
            kfold_method=overrides.get("kfold_method", "StratifiedKFold"),
            granularity=overrides.get("granularity", "level3"),
            phenotype_column=overrides.get("phenotype_column", "cell_type"),
            batch_column=overrides.get("batch_column", "batch_id"),
            image_data_dir=Path(img_dir).resolve() if img_dir else None,
        )

    def kfold_dir(self) -> Path:
        return self.processed_dir / f"kfolds_{self.kfold_method}_{self.granularity}"

    def labels_path(self) -> Path:
        return self.processed_dir / f"labels_{self.kfold_method}_{self.granularity}.csv"

    def results_dir(self, method_id: str) -> Path:
        return self.root / "results" / self.dataset_name / method_id / self.granularity

    def artifact(self, key: str, default: Optional[str] = None) -> Optional[Path]:
        """Look up a dataset-specific artifact path from benchmark or dataset config."""
        bench = self.config.get("benchmark", {})
        ds_art = bench.get("artifacts", {})
        val = ds_art.get(key, default)
        return (self.root / val).resolve() if val else None

    def ensure_kfolds(self, strip_labels: bool = True, dropna: bool = True) -> Path:
        """Create k-folds via run_kfold_creator if they do not yet exist."""
        kdir = self.kfold_dir()
        if kdir.is_dir() and any(kdir.glob("fold_*_train.csv")):
            return kdir

        script = self.root / "src" / "methods" / "utils" / "run_kfold_creator.py"
        cmd = [
            sys.executable,
            str(script),
            "--main_dir", str(self.root),
            "--dataset_name", self.dataset_name,
            "--n_splits", "5",
            "--method", self.kfold_method,
            "--phenotype_column", self.phenotype_column,
            "--batch_identifier_column", self.batch_column or "batch_id",
        ]
        if strip_labels:
            cmd.append("--strip_labels")
        if dropna:
            cmd.append("--dropna")

        print(f"[benchmark] Creating k-folds for {self.dataset_name} …")
        subprocess.run(cmd, check=True, cwd=str(self.root))
        return kdir

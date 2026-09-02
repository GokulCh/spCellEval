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
_METHODS_UTILS = _REPO_ROOT / "src" / "methods" / "utils"
for _p in (_PSEUDO_DIR, _METHODS_UTILS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ground_truth import get_marker_columns, infer_separate_col  # noqa: E402
from kfold_strategies import DEFAULT_SUPERVISED_KFOLD_METHODS, result_method_id  # noqa: E402


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
    kfold_methods: List[str] = field(default_factory=lambda: list(DEFAULT_SUPERVISED_KFOLD_METHODS))
    granularity: str = "level3"
    phenotype_column: str = "cell_type"
    batch_column: Optional[str] = None
    n_splits: int = 5
    rare_fraction: float = 0.01
    common_fraction: float = 0.05
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

        kfold_methods = list(
            overrides.get("kfold_methods")
            or ([overrides["kfold_method"]] if overrides.get("kfold_method") else list(DEFAULT_SUPERVISED_KFOLD_METHODS))
        )

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
            kfold_method=kfold_methods[0],
            kfold_methods=kfold_methods,
            granularity=overrides.get("granularity", "level3"),
            phenotype_column=overrides.get("phenotype_column", "cell_type"),
            batch_column=overrides.get("batch_column", "batch_id"),
            n_splits=int(overrides.get("n_splits", 5)),
            rare_fraction=float(overrides.get("rare_fraction", 0.01)),
            common_fraction=float(overrides.get("common_fraction", 0.05)),
            image_data_dir=Path(img_dir).resolve() if img_dir else None,
        )

    def kfold_dir(self, kfold_method: Optional[str] = None) -> Path:
        method = kfold_method or self.kfold_method
        return self.processed_dir / f"kfolds_{method}_{self.granularity}"

    def labels_path(self, kfold_method: Optional[str] = None) -> Path:
        method = kfold_method or self.kfold_method
        return self.processed_dir / f"labels_{method}_{self.granularity}.csv"

    def results_dir(self, method_id: str, kfold_method: Optional[str] = None) -> Path:
        result_id = result_method_id(method_id, kfold_method or self.kfold_method)
        return self.root / "results" / self.dataset_name / result_id / self.granularity

    def artifact(self, key: str, default: Optional[str] = None) -> Optional[Path]:
        """Look up a dataset-specific artifact path from benchmark or dataset config."""
        bench = self.config.get("benchmark", {})
        ds_art = bench.get("artifacts", {})
        val = ds_art.get(key, default)
        return (self.root / val).resolve() if val else None

    def _kfold_exists(self, kfold_method: str) -> bool:
        kdir = self.kfold_dir(kfold_method)
        return kdir.is_dir() and any(kdir.glob("fold_*_train.csv"))

    def ensure_kfold(
        self,
        kfold_method: str,
        *,
        strip_labels: bool = True,
        dropna: bool = True,
    ) -> Path:
        """Create one k-fold split set if it does not yet exist."""
        if self._kfold_exists(kfold_method):
            return self.kfold_dir(kfold_method)
        return self._run_kfold_creator([kfold_method], strip_labels=strip_labels, dropna=dropna)

    def _run_kfold_creator(
        self,
        methods: List[str],
        *,
        strip_labels: bool = True,
        dropna: bool = True,
    ) -> Path:
        script = self.root / "src" / "methods" / "utils" / "run_kfold_creator.py"
        cmd = [
            sys.executable,
            str(script),
            "--main_dir", str(self.root),
            "--dataset_name", self.dataset_name,
            "--n_splits", str(self.n_splits),
            "--phenotype_column", self.phenotype_column,
            "--batch_identifier_column", self.batch_column or "batch_id",
            "--rare_fraction", str(self.rare_fraction),
            "--common_fraction", str(self.common_fraction),
            "--methods", *methods,
        ]
        if strip_labels:
            cmd.append("--strip_labels")
        if dropna:
            cmd.append("--dropna")

        print(
            f"[benchmark] Creating k-folds for {self.dataset_name} "
            f"({', '.join(methods)}) in one preprocess pass …"
        )
        subprocess.run(cmd, check=True, cwd=str(self.root))
        return self.kfold_dir(methods[0])

    def ensure_kfolds(self, strip_labels: bool = True, dropna: bool = True) -> Path:
        """Create every configured k-fold split set (standard + progressive by default)."""
        missing = [m for m in self.kfold_methods if not self._kfold_exists(m)]
        if not missing:
            return self.kfold_dir(self.kfold_methods[0])
        return self._run_kfold_creator(missing, strip_labels=strip_labels, dropna=dropna)

    def remove_kfolds(self) -> None:
        """Delete on-disk fold CSVs/labels for all configured k-fold strategies."""
        import shutil

        for method in self.kfold_methods:
            kdir = self.kfold_dir(method)
            if kdir.is_dir():
                shutil.rmtree(kdir)
            labels = self.labels_path(method)
            if labels.is_file():
                labels.unlink()

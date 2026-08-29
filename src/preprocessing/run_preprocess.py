#!/usr/bin/env python
"""
run_preprocess.py
=================
CLI entry-point for the spCellEval preprocessing pipeline.

Usage
-----
Run a single dataset::

    python src/preprocessing/run_preprocess.py --config configs/immucan.yaml

Run all YAML configs in the configs/ directory::

    python src/preprocessing/run_preprocess.py --config_dir configs/

Full options::

    python src/preprocessing/run_preprocess.py --help

Pipeline
--------
For each YAML config the script:

1. Validates required fields.
2. Instantiates a ``DatasetRunner`` (see utils/dataset_runner.py).
3. Runs the preprocessing pipeline (load → rename → label remap →
   arcsinh transform → hierarchy derivation → column reorder).
4. Saves the output CSV to ``output.processed_dir / output.output_filename``.

The script always resolves file paths relative to ``--root_dir`` (default:
the repository root, detected automatically from this file's location).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Path bootstrap: ensure utils/ is importable regardless of how the script is
# invoked.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent         # src/preprocessing/
_UTILS_DIR  = _SCRIPT_DIR / "utils"
_REPO_ROOT  = _SCRIPT_DIR.parents[1]                 # spCellEval/

for _p in [str(_UTILS_DIR), str(_SCRIPT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset_runner import DatasetRunner  # noqa: E402 (after sys.path adjustment)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("run_preprocess")

# ---------------------------------------------------------------------------
# Required top-level YAML keys
# ---------------------------------------------------------------------------
_REQUIRED_KEYS = ["dataset_name", "raw_data", "column_mappings", "output"]
_REQUIRED_RAW  = ["file_path", "format"]
_REQUIRED_OUT  = ["processed_dir", "output_filename"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_config(cfg: dict, config_path: Path) -> None:
    """Raise ``ValueError`` if the config is missing mandatory fields."""
    errors: list[str] = []

    for key in _REQUIRED_KEYS:
        if key not in cfg:
            errors.append(f"  Missing top-level key: '{key}'")

    raw = cfg.get("raw_data", {})
    for key in _REQUIRED_RAW:
        if key not in raw:
            errors.append(f"  Missing raw_data.{key}")

    out = cfg.get("output", {})
    for key in _REQUIRED_OUT:
        if key not in out:
            errors.append(f"  Missing output.{key}")

    col = cfg.get("column_mappings", {})
    for key in ["cell_id", "spatial_x", "spatial_y", "image_id", "ground_truth_label"]:
        if key not in col:
            errors.append(f"  Missing column_mappings.{key}")

    if errors:
        msg = f"Config validation failed for {config_path}:\n" + "\n".join(errors)
        raise ValueError(msg)

    logger.info("Config validated: %s ✓", config_path.name)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_one(
    config_path: Path,
    root_dir: Path,
    dry_run: bool = False,
    override_dataset: str | None = None,
    override_output: Path | None = None,
) -> None:
    """Preprocess a single dataset described by *config_path*."""
    start_time = time.time()

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    validate_config(cfg, config_path)

    dataset = override_dataset or cfg.get("dataset_name", config_path.stem)
    # Update config with overrides
    cfg["dataset_name"] = dataset
    if override_output:
        cfg.setdefault("output", {})["processed_dir"] = str(override_output.parent)
        cfg["output"]["output_filename"] = override_output.name

    logger.info("=" * 60)
    logger.info("Dataset : %s", dataset)
    logger.info("Config  : %s", config_path)
    logger.info("Root    : %s", root_dir)

    runner = DatasetRunner(cfg, root_dir=root_dir)

    if dry_run:
        logger.info("[DRY-RUN] Would process %s — skipping actual load/save.", dataset)
        return

    df = runner.run()

    if df.empty:
        raise ValueError(f"Validation failed: Dataframe is empty after preprocessing {dataset}!")

    out_path = runner.save(df)

    if cfg.get("output", {}).get("export_features_only", True):
        runner.export_features_only()

    if not out_path.exists():
        raise FileNotFoundError(f"Validation failed: Output file was not created at {out_path}")

    elapsed = time.time() - start_time

    # Quick summary
    logger.info("-" * 60)
    logger.info("Output  : %s", out_path)
    logger.info("Shape   : %d rows × %d columns", *df.shape)
    for col in ["cell_type", "level_2_cell_type", "level_1_cell_type"]:
        if col in df.columns:
            logger.info("  %s: %d unique values", col, df[col].nunique())
    logger.info("Time    : %.2f seconds", elapsed)
    logger.info("=" * 60)


def run_all(config_dir: Path, root_dir: Path, dry_run: bool = False) -> None:
    """Iterate over all *.yaml files in *config_dir* and preprocess each."""
    yamls = sorted(config_dir.glob("*.yaml")) + sorted(config_dir.glob("*.yml"))
    if not yamls:
        logger.error("No YAML files found in %s", config_dir)
        sys.exit(1)

    logger.info("Found %d config file(s) in %s.", len(yamls), config_dir)
    failures: list[tuple[Path, Exception]] = []

    for path in yamls:
        try:
            run_one(path, root_dir=root_dir, dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001
            logger.error("FAILED %s: %s", path.name, exc)
            failures.append((path, exc))

    if failures:
        logger.error("%d dataset(s) failed:", len(failures))
        for p, e in failures:
            logger.error("  %s — %s", p.name, e)
        sys.exit(1)

    logger.info("All datasets processed successfully.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_preprocess",
        description=(
            "spCellEval preprocessing pipeline.\n\n"
            "Reads a YAML config, loads the raw data, applies arcsinh "
            "normalisation and label harmonisation, derives hierarchical "
            "cell-type levels, and writes a standardised quantification CSV."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config",
        type=Path,
        metavar="PATH",
        help="Path to a single dataset YAML config (e.g. configs/immucan.yaml).",
    )
    source.add_argument(
        "--config_dir",
        type=Path,
        metavar="DIR",
        help="Directory containing *.yaml configs. All are processed in order.",
    )

    parser.add_argument(
        "--root_dir",
        type=Path,
        default=_REPO_ROOT,
        metavar="DIR",
        help=(
            "Workspace root; all relative paths in YAML are resolved against "
            f"this directory. Default: auto-detected as {_REPO_ROOT}"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Override the dataset_name from the YAML config.",
    )
    parser.add_argument(
        "--override_output",
        type=Path,
        metavar="PATH",
        help="Override the output file path (e.g. data/processed/my_dataset.csv).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate configs and print the plan without loading or writing data.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    root = args.root_dir.resolve()

    if args.config:
        cfg_path = args.config if args.config.is_absolute() else root / args.config
        if not cfg_path.exists():
            logger.error("Config file not found: %s", cfg_path)
            sys.exit(1)
        run_one(
            cfg_path,
            root_dir=root,
            dry_run=args.dry_run,
            override_dataset=args.dataset,
            override_output=args.override_output,
        )
    else:
        cfg_dir = args.config_dir if args.config_dir.is_absolute() else root / args.config_dir
        if not cfg_dir.is_dir():
            logger.error("Config directory not found: %s", cfg_dir)
            sys.exit(1)
        run_all(cfg_dir, root_dir=root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

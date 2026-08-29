#!/usr/bin/env python
"""
run_benchmark.py
================
Stage 3 — unified multi-method annotation benchmark runner.

Runs any combination of the 22+ methods registered in ``method_registry.py``,
routing each to the correct executor based on its category:

* **supervised_kfold** — needs k-fold CSVs (Random Forest, XGBoost, MAPS)
* **unsupervised_quant** — full quantification CSV (Leiden, FlowSOM, Starling)
* **prior_knowledge** — decision matrix / rules (Scyan, TACIT, Signature)
* **image_pipeline** — raw images + segmentation (Stellar, EVA, KRONOS, …)

Usage
-----
Run default tabular methods on IMMUcan::

    python src/benchmark/run_benchmark.py --dataset IMMUcan

Run specific methods::

    python src/benchmark/run_benchmark.py --dataset IMMUcan \\
        --methods random_forest leiden scyan signature

List all registered methods::

    python src/benchmark/run_benchmark.py --list_methods

Ensure k-folds exist before supervised methods::

    python src/benchmark/run_benchmark.py --dataset IMMUcan \\
        --methods random_forest --ensure_kfolds
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import yaml

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dataset_context import DatasetContext  # noqa: E402
from executors import MethodExecutionError, execute_method  # noqa: E402
from method_registry import METHOD_REGISTRY, MethodCategory, list_methods  # noqa: E402

logger = logging.getLogger("run_benchmark")


def _load_benchmark_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _resolve_dataset_entry(bench_cfg: dict, dataset_name: str) -> dict:
    datasets = bench_cfg.get("datasets", {})
    if dataset_name not in datasets:
        raise ValueError(
            f"Dataset '{dataset_name}' not in benchmark config. "
            f"Available: {list(datasets.keys())}"
        )
    return datasets[dataset_name]


def _default_methods(bench_cfg: dict, ds_entry: dict, tabular_only: bool) -> List[str]:
    if "methods" in ds_entry:
        return ds_entry["methods"]
    if tabular_only:
        return [s.id for s in list_methods(tabular_only=True)]
    return list(METHOD_REGISTRY.keys())


def run_benchmark(
    dataset_name: str,
    methods: List[str],
    bench_config_path: Path,
    root: Path,
    ensure_kfolds: bool = False,
    recreate_kfolds: bool = False,
    iterations: int = 1,
    skip_missing_deps: bool = True,
    fail_fast: bool = False,
    spatial_smooth: Optional[bool] = None,
    spatial_k_neighbors: int = 15,
) -> dict:
    bench_cfg = _load_benchmark_config(bench_config_path)
    ds_entry = _resolve_dataset_entry(bench_cfg, dataset_name)

    defaults = bench_cfg.get("defaults", {})
    do_spatial = defaults.get("spatial_smooth", False) if spatial_smooth is None else spatial_smooth
    k_spatial = defaults.get("spatial_k_neighbors", spatial_k_neighbors)
    overrides = {
        "kfold_method": ds_entry.get("kfold_method", defaults.get("kfold_method", "StratifiedKFold")),
        "granularity": ds_entry.get("granularity", defaults.get("granularity", "level3")),
        "phenotype_column": ds_entry.get("phenotype_column", defaults.get("phenotype_column", "cell_type")),
        "batch_column": ds_entry.get("batch_column", defaults.get("batch_column", "batch_id")),
        "image_data_dir": ds_entry.get("image_data_dir"),
    }

    ds_config_path = root / ds_entry.get("config", f"configs/{dataset_name.lower()}.yaml")
    ctx = DatasetContext.from_config(ds_config_path, root=root, benchmark_overrides=overrides)

    # Merge dataset artifacts into config for executor lookups
    if "artifacts" in ds_entry:
        ctx.config.setdefault("benchmark", {})["artifacts"] = ds_entry["artifacts"]

    if recreate_kfolds:
        kdir = ctx.kfold_dir()
        if kdir.is_dir():
            shutil.rmtree(kdir)
        labels = ctx.labels_path()
        if labels.is_file():
            labels.unlink()

    if ensure_kfolds or recreate_kfolds:
        ctx.ensure_kfolds(strip_labels=defaults.get("strip_labels", True))

    results = {"succeeded": [], "skipped": [], "failed": []}

    for method_id in methods:
        if method_id not in METHOD_REGISTRY:
            logger.error("Unknown method '%s' — run --list_methods to see options.", method_id)
            results["failed"].append((method_id, "unknown method"))
            if fail_fast:
                break
            continue

        spec = METHOD_REGISTRY[method_id]
        logger.info("=" * 60)
        logger.info(
            "Method: %s [%s] — %s",
            spec.display_name,
            spec.category.value,
            spec.description,
        )

        try:
            out = execute_method(
                ctx, spec,
                iterations=iterations,
                skip_missing_deps=skip_missing_deps,
                spatial_smooth=do_spatial,
                spatial_k_neighbors=k_spatial,
            )
            if out is None:
                results["skipped"].append(method_id)
            else:
                results["succeeded"].append((method_id, str(out)))
        except MethodExecutionError as exc:
            logger.error("FAIL %s: %s", method_id, exc)
            results["failed"].append((method_id, str(exc)))
            if fail_fast:
                break

    return results


def _print_method_catalog() -> None:
    print("\nRegistered methods in spCellEval benchmark:\n")
    for cat in MethodCategory:
        specs = list_methods(category=cat)
        if not specs:
            continue
        print(f"  [{cat.value}]")
        for s in specs:
            flags = []
            if s.tabular_default:
                flags.append("tabular-default")
            if s.requires_r:
                flags.append("R")
            if s.requires_gpu:
                flags.append("GPU")
            if s.requires_raw_images:
                flags.append("images")
            flag_str = f" ({', '.join(flags)})" if flags else ""
            print(f"    {s.id:<22} {s.display_name}{flag_str}")
            print(f"    {'':22} {s.description}")
        print()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="spCellEval Stage 3 — unified benchmark runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", type=str, help="Dataset name (e.g. IMMUcan, CRC_TMA).")
    p.add_argument(
        "--methods", nargs="+", metavar="METHOD",
        help="Method IDs to run (default: tabular-default methods for dataset).",
    )
    p.add_argument(
        "--benchmark_config",
        type=Path,
        default=_REPO / "configs" / "benchmark.yaml",
        help="Benchmark configuration YAML.",
    )
    p.add_argument("--root_dir", type=Path, default=_REPO)
    p.add_argument(
        "--tabular_only",
        action="store_true",
        default=True,
        help="When no --methods given, run only tabular-default methods (default: true).",
    )
    p.add_argument(
        "--all_methods",
        action="store_true",
        help="Run every registered method (image methods skipped if no image_data_dir).",
    )
    p.add_argument(
        "--recreate_kfolds",
        action="store_true",
        help="Delete and recreate k-folds (with --dropna) before supervised methods.",
    )
    p.add_argument(
        "--ensure_kfolds",
        action="store_true",
        help="Create k-folds via run_kfold_creator before supervised methods.",
    )
    p.add_argument(
        "-n", "--iterations",
        type=int,
        default=1,
        help="Stability iterations for unsupervised / prior-knowledge methods.",
    )
    p.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop on first method failure.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Error instead of skip when R/images are missing.",
    )
    p.add_argument(
        "--no_spatial_smooth",
        action="store_true",
        help="Disable sliding-window spatial post-processing after each method.",
    )
    p.add_argument("--list_methods", action="store_true", help="Print method catalog and exit.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    if args.list_methods:
        _print_method_catalog()
        return

    if not args.dataset:
        parser.error("--dataset is required (or use --list_methods).")

    root = args.root_dir.resolve()
    bench_cfg = _load_benchmark_config(
        args.benchmark_config if args.benchmark_config.is_absolute()
        else root / args.benchmark_config
    )
    ds_entry = _resolve_dataset_entry(bench_cfg, args.dataset)

    if args.methods:
        methods = args.methods
    elif args.all_methods:
        methods = list(METHOD_REGISTRY.keys())
    else:
        methods = _default_methods(bench_cfg, ds_entry, tabular_only=True)

    logger.info("Dataset : %s", args.dataset)
    logger.info("Methods : %s", ", ".join(methods))

    summary = run_benchmark(
        dataset_name=args.dataset,
        methods=methods,
        bench_config_path=args.benchmark_config if args.benchmark_config.is_absolute() else root / args.benchmark_config,
        root=root,
        ensure_kfolds=args.ensure_kfolds,
        recreate_kfolds=args.recreate_kfolds,
        iterations=args.iterations,
        skip_missing_deps=not args.strict,
        fail_fast=args.fail_fast,
        spatial_smooth=False if args.no_spatial_smooth else None,
    )

    print("\n" + "=" * 60)
    print(f"Benchmark complete for {args.dataset}")
    print(f"  Succeeded : {len(summary['succeeded'])}")
    for mid, path in summary["succeeded"]:
        print(f"    OK {mid} -> {path}")
    if summary["skipped"]:
        print(f"  Skipped   : {len(summary['skipped'])}")
        for mid in summary["skipped"]:
            print(f"    - {mid}")
    if summary["failed"]:
        print(f"  Failed    : {len(summary['failed'])}")
        for mid, err in summary["failed"]:
            print(f"    FAIL {mid}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()

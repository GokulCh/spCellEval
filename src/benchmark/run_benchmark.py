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
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait
from pathlib import Path
from typing import Dict, List, Optional

import yaml

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_UTILS = _REPO / "src" / "utils"
if str(_UTILS) not in sys.path:
    sys.path.insert(0, str(_UTILS))

from dataset_context import DatasetContext  # noqa: E402
from executors import MethodExecutionError, execute_method  # noqa: E402
from method_registry import METHOD_REGISTRY, MethodCategory, list_methods  # noqa: E402
from method_registry import resolve_unlabeled_methods, filter_unlabeled_methods  # noqa: E402
from parallel import resolve_ml_n_jobs, resolve_worker_count  # noqa: E402
from parallel_runner import build_method_payloads, run_method_job  # noqa: E402
from pipeline_logging import PipelineLogSession  # noqa: E402

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
    unlabeled: bool = False,
    kfold_method: Optional[str] = None,
    kfold_methods: Optional[List[str]] = None,
    parallel_jobs: Optional[int] = None,
    method_timeout: int = 3600,
    *,
    baseline_split: bool = False,
    cross_validation: bool = False,
    subsample_experiment: bool = False,
    fractions: str = "0.01,0.05,0.10,0.20,0.40,0.60,0.80",
    holdout: float = 0.20,
) -> dict:
    bench_cfg = _load_benchmark_config(bench_config_path)
    ds_entry = _resolve_dataset_entry(bench_cfg, dataset_name)

    defaults = bench_cfg.get("defaults", {})
    do_spatial = defaults.get("spatial_smooth", False) if spatial_smooth is None else spatial_smooth
    k_spatial = defaults.get("spatial_k_neighbors", spatial_k_neighbors)
    if kfold_methods:
        resolved_kfold_methods = list(kfold_methods)
    elif kfold_method:
        resolved_kfold_methods = [kfold_method]
    else:
        resolved_kfold_methods = list(
            ds_entry.get(
                "kfold_methods",
                defaults.get("kfold_methods", ["StratifiedKFold", "ProgressiveKFold"]),
            )
        )

    overrides = {
        "kfold_method": resolved_kfold_methods[0],
        "kfold_methods": resolved_kfold_methods,
        "granularity": ds_entry.get("granularity", defaults.get("granularity", "level3")),
        "phenotype_column": ds_entry.get("phenotype_column", defaults.get("phenotype_column", "cell_type")),
        "batch_column": ds_entry.get("batch_column", defaults.get("batch_column", "batch_id")),
        "n_splits": ds_entry.get("n_splits", defaults.get("n_splits", 5)),
        "rare_fraction": ds_entry.get("rare_fraction", defaults.get("rare_fraction", 0.01)),
        "common_fraction": ds_entry.get("common_fraction", defaults.get("common_fraction", 0.05)),
        "image_data_dir": ds_entry.get("image_data_dir"),
    }

    ds_config_path = root / ds_entry.get("config", f"configs/{dataset_name.lower()}.yaml")
    ctx = DatasetContext.from_config(ds_config_path, root=root, benchmark_overrides=overrides)

    # Merge dataset artifacts into config for executor lookups
    if "artifacts" in ds_entry:
        ctx.config.setdefault("benchmark", {})["artifacts"] = ds_entry["artifacts"]

    if unlabeled:
        methods = filter_unlabeled_methods(methods)
        logger.info("Unlabeled mode — methods: %s", ", ".join(methods))
    else:
        if recreate_kfolds:
            ctx.remove_kfolds()
        if ensure_kfolds or recreate_kfolds:
            ctx.ensure_kfolds(strip_labels=defaults.get("strip_labels", True))

    results = {"succeeded": [], "skipped": [], "failed": []}

    # ── Objective experiments (share the same DatasetContext / split config) ──
    # Run before the method loop so the experiments are executed in BOTH the
    # sequential and the parallel branch (the sequential branch returns early).
    if baseline_split or cross_validation or subsample_experiment:
        from experiments import run_baseline_split as _run_baseline  # noqa: WPS433
        from experiments import run_five_fold_cv as _run_fivefold  # noqa: WPS433
        from experiments import run_subsampling_experiment as _run_subsample  # noqa: WPS433
        from experiments import SUPERVISED_METHODS  # noqa: WPS433

        exp_root = root / "experiments" / dataset_name
        exp_methods = [m for m in methods if m in SUPERVISED_METHODS]

        if baseline_split and exp_methods:
            logger.info("Objective 1 — baseline 80/20 split experiment")
            _run_baseline(ctx, exp_methods, exp_root, test_size=holdout)
        if cross_validation and exp_methods:
            logger.info("Objective 3 — 5-fold cross-validation summary")
            _run_fivefold(ctx, exp_methods, exp_root)
        if subsample_experiment and exp_methods:
            logger.info("Objective 4 — subsampling efficiency sweep")
            fraction_values = [float(x.strip()) for x in fractions.split(",")]
            _run_subsample(
                ctx, exp_methods, exp_root, fractions=fraction_values, holdout=holdout
            )
        results.setdefault("experiments_dir", str(exp_root))

    workers = resolve_worker_count(
        parallel_jobs if parallel_jobs is not None else defaults.get("parallel_jobs", 0)
    )
    ml_n_jobs = resolve_ml_n_jobs(workers)

    known_methods = [m for m in methods if m in METHOD_REGISTRY]
    unknown = [m for m in methods if m not in METHOD_REGISTRY]
    for method_id in unknown:
        logger.error("Unknown method '%s' — run --list_methods to see options.", method_id)
        results["failed"].append((method_id, "unknown method"))

    if fail_fast and unknown:
        return results

    def _record(method_id: str, status: str, path: Optional[str], error: Optional[str]) -> bool:
        if status == "succeeded":
            results["succeeded"].append((method_id, path or ""))
            return True
        if status == "skipped":
            results["skipped"].append(method_id)
            return True
        results["failed"].append((method_id, error or "failed"))
        return False

    if workers <= 1 or len(known_methods) <= 1:
        for method_id in known_methods:
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
                    ml_n_jobs=ml_n_jobs,
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
            except Exception as exc:  # isolate one method's crash from the suite
                logger.error("FAIL %s (unexpected): %s", method_id, exc)
                results["failed"].append((method_id, f"{type(exc).__name__}: {exc}"))
                if fail_fast:
                    break
        return results

    logger.info("Running %d methods with %d parallel workers", len(known_methods), workers)
    payloads = build_method_payloads(
        methods=known_methods,
        config_path=ds_config_path,
        root=root,
        benchmark_overrides=overrides,
        artifacts=ds_entry.get("artifacts"),
        iterations=iterations,
        skip_missing_deps=skip_missing_deps,
        spatial_smooth=do_spatial,
        spatial_k_neighbors=k_spatial,
        ml_n_jobs=ml_n_jobs,
    )

    # Bound per-process OpenMP/thread pools so `workers * threads <= cpus`
    # (prevents the XGBoost/libgomp oversubscription stall) and use the *spawn*
    # context: forking workers from a main process that already initialised
    # OpenMP/XGBoost thread pools is a documented deadlock source.
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, ml_n_jobs)))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max(1, ml_n_jobs)))
    os.environ.setdefault("MKL_NUM_THREADS", str(max(1, ml_n_jobs)))

    mp_ctx = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx)
    futures: Dict = {}
    submitted_at: Dict[str, float] = {}
    for payload in payloads:
        submitted_at[payload["method_id"]] = time.monotonic()
        futures[pool.submit(run_method_job, payload)] = payload["method_id"]

    pending = dict(futures)
    timed_out: List[str] = []

    try:
        while pending:
            done, _ = wait(list(pending), timeout=30)
            now = time.monotonic()
            for future in done:
                method_id = pending.pop(future)
                try:
                    m_id, status, path, error = future.result()
                except Exception as exc:  # worker crashed / pool broke — record, keep going
                    m_id, status, path, error = method_id, "failed", None, f"{type(exc).__name__}: {exc}"
                    logger.error("Worker for %s crashed: %s", method_id, error)
                logger.info("Method %s finished: %s", m_id, status)
                if not _record(m_id, status, path, error) and fail_fast:
                    for f in pending:
                        f.cancel()
                    pending.clear()
                    break

            # Wall-clock timeout guard: a wedged worker must not freeze the batch.
            for future in list(pending):
                method_id = pending[future]
                elapsed = now - submitted_at[method_id]
                if elapsed > method_timeout:
                    logger.error(
                        "Method %s timed out after %.0fs — marked failed (worker may be wedged).",
                        method_id, elapsed,
                    )
                    _record(method_id, "failed", None, f"timed out after {elapsed:.0f}s")
                    timed_out.append(method_id)
                    pending.pop(future)
                    future.cancel()
                    if fail_fast:
                        for f in pending:
                            f.cancel()
                        pending.clear()
                        break
    finally:
        # wait=False: never block on a wedged worker during teardown.
        pool.shutdown(wait=False, cancel_futures=True)

    if timed_out:
        logger.warning("Timed-out methods: %s", ", ".join(sorted(timed_out)))

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
    p.add_argument(
        "--unlabeled",
        action="store_true",
        help="Run without expert labels: skip k-folds, use prior-knowledge + clustering methods only.",
    )
    p.add_argument(
        "--kfold_method",
        type=str,
        choices=["StratifiedKFold", "ProgressiveKFold", "StratifiedGroupKFold", "GroupShuffleSplit"],
        default=None,
        help="Override k-fold strategy from benchmark config (default: StratifiedKFold).",
    )
    p.add_argument(
        "--parallel_jobs",
        type=int,
        default=None,
        help="Number of benchmark methods to run in parallel (0=auto, 1=sequential).",
    )
    p.add_argument(
        "--method_timeout",
        type=int,
        default=3600,
        help="Max wall-clock seconds per method in the parallel loop (0=never time out). "
             "A wedged worker is marked failed and the batch continues. Default: 3600.",
    )
    p.add_argument(
        "--baseline_split",
        action="store_true",
        help="Objective 1 — run the baseline 80/20 train/test split experiment.",
    )
    p.add_argument(
        "--cross_validation",
        action="store_true",
        help="Objective 3 — run the 5-fold cross-validation summary.",
    )
    p.add_argument(
        "--subsample_experiment",
        action="store_true",
        help="Objective 4 — run the training-size efficiency (subsampling) sweep.",
    )
    p.add_argument(
        "--fractions",
        type=str,
        default="0.01,0.05,0.10,0.20,0.40,0.60,0.80",
        help="Comma-separated train fractions for --subsample_experiment.",
    )
    p.add_argument(
        "--holdout",
        type=float,
        default=0.20,
        help="Fixed test fraction for --baseline_split / --subsample_experiment.",
    )
    p.add_argument("--list_methods", action="store_true", help="Print method catalog and exit.")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--log_dir",
        type=Path,
        default=None,
        help="Directory for execution logs (default: results/{dataset}/logs/).",
    )
    p.add_argument(
        "--no_log_file",
        action="store_true",
        help="Disable writing execution logs to disk.",
    )
    return p


def _run_benchmark_cli(args: argparse.Namespace) -> None:
    root = args.root_dir.resolve()
    bench_cfg = _load_benchmark_config(
        args.benchmark_config if args.benchmark_config.is_absolute()
        else root / args.benchmark_config
    )
    ds_entry = _resolve_dataset_entry(bench_cfg, args.dataset)

    if args.methods:
        methods = args.methods
    elif args.unlabeled:
        methods = resolve_unlabeled_methods(ds_entry, bench_cfg, None)
    elif args.all_methods:
        methods = list(METHOD_REGISTRY.keys())
    else:
        methods = _default_methods(bench_cfg, ds_entry, tabular_only=True)

    if args.unlabeled:
        methods = filter_unlabeled_methods(methods)

    logger.info("Dataset : %s", args.dataset)
    logger.info("Methods : %s", ", ".join(methods))
    if args.kfold_method:
        logger.info("K-fold strategies: %s (CLI override)", args.kfold_method)
    else:
        ds_defaults = bench_cfg.get("defaults", {})
        ds_entry = bench_cfg.get("datasets", {}).get(args.dataset, {})
        strategies = ds_entry.get(
            "kfold_methods",
            ds_defaults.get("kfold_methods", ["StratifiedKFold", "ProgressiveKFold"]),
        )
        logger.info("K-fold strategies: %s", ", ".join(strategies))

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
        unlabeled=args.unlabeled,
        kfold_method=args.kfold_method,
        parallel_jobs=args.parallel_jobs,
        method_timeout=args.method_timeout,
        baseline_split=args.baseline_split,
        cross_validation=args.cross_validation,
        subsample_experiment=args.subsample_experiment,
        fractions=args.fractions,
        holdout=args.holdout,
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

    if summary.get("experiments_dir"):
        print(f"\nExperiment outputs written to: {summary['experiments_dir']}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_methods:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            level=logging.DEBUG if args.verbose else logging.INFO,
        )
        _print_method_catalog()
        return

    if not args.dataset:
        parser.error("--dataset is required (or use --list_methods).")

    root = args.root_dir.resolve()
    if args.no_log_file:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            level=logging.DEBUG if args.verbose else logging.INFO,
        )
        _run_benchmark_cli(args)
        return

    with PipelineLogSession(
        root=root,
        dataset=args.dataset,
        run_name="benchmark",
        verbose=args.verbose,
        log_dir=args.log_dir.resolve() if args.log_dir else None,
    ) as log_session:
        log_session.configure_logging()
        _run_benchmark_cli(args)
        print(f"\nFull execution log: {log_session.main_log}")


if __name__ == "__main__":
    main()

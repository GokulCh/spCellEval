#!/usr/bin/env python
"""
run_evaluation.py
===================
Stage 4 — automated evaluation CLI.

Scans ``results/{dataset}/`` for prediction CSVs produced by Stage 3,
computes supervised metrics (accuracy, macro F1, ARI, NMI, …) and
unsupervised metrics (silhouette, spatial entropy, marker consistency)
when ground truth is absent.

Usage
-----
Evaluate all methods for IMMUcan::

    python src/evaluation/run_evaluation.py --dataset IMMUcan

Evaluate specific methods::

    python src/evaluation/run_evaluation.py --dataset IMMUcan \\
        --methods random_forest signature

Outputs (under ``results/{dataset}/summary/``)::

    per_fold_metrics.csv    — one row per method × level × fold
    final_results.csv       — mean ± std aggregated across folds
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from evaluator import aggregate_results, evaluate_dataset  # noqa: E402
from dataset_paths import resolve_dataset_quant  # noqa: E402

logger = logging.getLogger("run_evaluation")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="spCellEval Stage 4 — automated evaluation.")
    p.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. IMMUcan).")
    p.add_argument("--methods", nargs="+", help="Limit to specific method IDs.")
    p.add_argument(
        "--results_dir",
        type=Path,
        default=_REPO / "results",
        help="Root results directory.",
    )
    p.add_argument(
        "--levels",
        nargs="+",
        default=["level1", "level2", "level3"],
        choices=["level1", "level2", "level3"],
    )
    p.add_argument(
        "--hierarchy",
        type=Path,
        default=_HERE / "cell_type_hierarchy.txt",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=_REPO / "configs" / "evaluation.yaml",
        help="Optional evaluation YAML config.",
    )
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--plot", action="store_true", help="Generate summary plots after evaluation.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    results_root = args.results_dir.resolve()
    methods = args.methods

    if args.config and args.config.is_file():
        with args.config.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        ds_cfg = cfg.get("datasets", {}).get(args.dataset, {})
        if not methods and "methods" in ds_cfg:
            methods = ds_cfg["methods"]

    logger.info("Evaluating dataset: %s", args.dataset)
    quant_path, marker_cols = resolve_dataset_quant(args.dataset, _REPO)
    if quant_path:
        logger.info("Merging marker/spatial data from %s", quant_path)
    per_fold = evaluate_dataset(
        args.dataset,
        results_root,
        methods=methods,
        levels=args.levels,
        hierarchy_path=args.hierarchy,
        quant_path=quant_path,
        marker_cols=marker_cols or None,
    )

    if per_fold.empty:
        logger.error("No results to evaluate.")
        sys.exit(1)

    summary = aggregate_results(per_fold)

    out_dir = args.output_dir or (results_root / args.dataset / "summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_fold_path = out_dir / "per_fold_metrics.csv"
    summary_path = out_dir / "final_results.csv"
    per_fold.to_csv(per_fold_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print(f"Evaluation complete for {args.dataset}")
    print(f"  Per-fold metrics : {per_fold_path}")
    print(f"  Summary          : {summary_path}")
    print(f"  Methods evaluated: {per_fold['method'].nunique()}")
    print(f"  Total fold rows  : {len(per_fold)}")
    print("=" * 60)

    if not summary.empty:
        print("\nTop methods (level3, by overall_score):")
        l3 = summary[summary["level"] == "level3"].sort_values(
            "overall_score", ascending=False, na_position="last"
        )
        cols = ["method", "accuracy_mean", "macro_f1_mean", "silhouette_mean",
                "marker_purity_mean", "neighborhood_consistency_mean", "overall_score"]
        cols = [c for c in cols if c in l3.columns]
        print(l3[cols].head(10).to_string(index=False))

    if args.plot:
        from visualize import generate_report_plots

        plots = generate_report_plots(args.dataset, results_root)
        print(f"\n  Plots written: {len(plots)} file(s) under {results_root / args.dataset / 'summary' / 'plots'}")


if __name__ == "__main__":
    main()

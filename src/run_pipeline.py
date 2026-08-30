#!/usr/bin/env python
"""
run_pipeline.py
===============
End-to-end spCellEval pipeline (Modules 1–5).

Stages
------
1. Preprocess raw data -> quantification CSV (+ optional parquet/h5ad)
2. Strip labels / export features-only matrix (leakage prevention)
3. Multi-method benchmark
4. Spatial post-processing (sliding-window smoothing + graph summary)
5. Evaluation + automated visualizations

Usage
-----
Full pipeline for IMMUcan::

    python src/run_pipeline.py --dataset IMMUcan --recreate_kfolds

Skip preprocessing when quant CSV already exists::

    python src/run_pipeline.py --dataset IMMUcan --skip_preprocess

Unlabeled clinical sample (no expert ClusterName / cell_type)::

    python src/run_pipeline.py --dataset CRC_TMA --unlabeled
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import yaml

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"

for _p in [
    str(_SRC / "benchmark"),
    str(_SRC / "evaluation"),
    str(_SRC / "spatial"),
    str(_SRC / "preprocessing"),
    str(_SRC / "pseudo_labeling"),
    str(_SRC / "utils"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from method_registry import resolve_unlabeled_methods  # noqa: E402
from pipeline_logging import PipelineLogSession, log_command_banner  # noqa: E402

logger = logging.getLogger("run_pipeline")


def _run_script(cmd: List[str], cwd: Optional[Path] = None, *, stage: str = "subprocess") -> None:
    log_command_banner(stage, cmd)
    result = subprocess.run(cmd, cwd=str(cwd or _REPO))
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")


def _dataset_config_path(dataset: str, bench_cfg: dict) -> Path:
    ds_entry = bench_cfg.get("datasets", {}).get(dataset, {})
    rel = ds_entry.get("config", f"configs/{dataset.lower()}.yaml")
    return _REPO / rel


def run_pipeline(
    dataset: str,
    *,
    methods: Optional[List[str]] = None,
    skip_preprocess: bool = False,
    skip_stage2: bool = False,
    skip_benchmark: bool = False,
    skip_spatial: bool = False,
    skip_eval: bool = False,
    skip_viz: bool = False,
    recreate_kfolds: bool = False,
    ensure_kfolds: bool = True,
    spatial_smooth: bool = True,
    export_graphs: bool = True,
    pseudo_label: bool = False,
    pseudo_method: str = "signature",
    unlabeled: bool = False,
    benchmark_config: Optional[Path] = None,
    root: Path = _REPO,
    parent_log_active: bool = False,
) -> dict:
    """Execute the full benchmark pipeline for one dataset."""
    bench_cfg_path = (benchmark_config or (root / "configs" / "benchmark.yaml")).resolve()
    with bench_cfg_path.open("r", encoding="utf-8") as fh:
        bench_cfg = yaml.safe_load(fh) or {}

    ds_entry = bench_cfg.get("datasets", {}).get(dataset, {})
    pipeline_defaults = bench_cfg.get("defaults", {}).get("pipeline", {})
    ds_config = _dataset_config_path(dataset, bench_cfg)
    with ds_config.open("r", encoding="utf-8") as fh:
        ds_cfg = yaml.safe_load(fh) or {}

    if unlabeled:
        methods = methods or resolve_unlabeled_methods(ds_entry, bench_cfg, None)
        pseudo_label = True
        ensure_kfolds = False
        recreate_kfolds = False

    quant_dir = root / ds_cfg.get("output", {}).get("processed_dir", f"data/processed/{dataset}")
    quant_file = ds_cfg.get("output", {}).get("output_filename", f"{dataset}_quantification.csv")
    quant_path = quant_dir / quant_file
    features_path = quant_dir / f"{dataset}_features_only.csv"

    stages = {}

    def _append_no_log(cmd: List[str]) -> List[str]:
        return [*cmd, "--no_log_file"]

    # Stage 1 — ETL
    if not skip_preprocess:
        preprocess_cmd = [
            sys.executable,
            str(_SRC / "preprocessing" / "run_preprocess.py"),
            "--config", str(ds_config),
            "--root_dir", str(root),
        ]
        if unlabeled:
            preprocess_cmd.append("--unlabeled")
        _run_script(preprocess_cmd, stage="preprocess")
        stages["preprocess"] = str(quant_path)
    elif not quant_path.is_file():
        raise FileNotFoundError(f"--skip_preprocess set but quant file missing: {quant_path}")
    else:
        stages["preprocess"] = f"skipped (exists: {quant_path})"

    # Stage 2 — Feature separation / optional pseudo-labeling
    if not skip_stage2:
        do_strip = pipeline_defaults.get("strip_labels", True) and not unlabeled
        if do_strip:
            _run_script([
                sys.executable,
                str(_SRC / "pseudo_labeling" / "run_pseudo_labeler.py"),
                "--config", str(ds_config),
                "--root_dir", str(root),
                "--strip_labels",
                "--output", str(features_path),
            ], stage="feature_separation")
            stages["feature_separation"] = str(features_path)
        elif unlabeled:
            stages["feature_separation"] = "skipped (unlabeled — no expert labels to strip)"

        if pseudo_label or pipeline_defaults.get("pseudo_label", False) or unlabeled:
            method = pseudo_method or pipeline_defaults.get("pseudo_label_method", "signature")
            pl_cmd = [
                sys.executable,
                str(_SRC / "pseudo_labeling" / "run_pseudo_labeler.py"),
                "--config", str(ds_config),
                "--root_dir", str(root),
                "--method", method,
            ]
            if unlabeled and method == "signature":
                pl_cmd.append("--annotate_quant")
            _run_script(pl_cmd, stage="pseudo_labeling")
            stages["pseudo_labeling"] = method
            if unlabeled:
                stages["annotated_quant"] = str(quant_dir / f"{dataset}_annotated.csv")
    else:
        stages["feature_separation"] = "skipped"

    # Stage 3 — Benchmark (spatial smoothing disabled here; done once in stage 4)
    if not skip_benchmark:
        cmd = [
            sys.executable,
            str(_SRC / "benchmark" / "run_benchmark.py"),
            "--dataset", dataset,
            "--root_dir", str(root),
            "--benchmark_config", str(bench_cfg_path),
            "--no_spatial_smooth",
        ]
        if methods:
            cmd.extend(["--methods", *methods])
        if unlabeled:
            cmd.append("--unlabeled")
        elif recreate_kfolds:
            cmd.append("--recreate_kfolds")
        elif ensure_kfolds:
            cmd.append("--ensure_kfolds")
        if parent_log_active:
            cmd = _append_no_log(cmd)
        _run_script(cmd, stage="benchmark")
        stages["benchmark"] = "done"
    else:
        stages["benchmark"] = "skipped"

    # Stage 4 — Spatial post-processing
    if spatial_smooth and not skip_spatial:
        from postprocess import export_spatial_graph_summary, postprocess_predictions_dir

        graph_method = pipeline_defaults.get("graph_method", "both")
        results_root = root / "results" / dataset
        n_files = 0
        if results_root.is_dir():
            for method_dir in results_root.iterdir():
                if method_dir.is_dir() and method_dir.name not in {"summary", "logs"}:
                    n_files += postprocess_predictions_dir(method_dir, quant_path)

        graph_path = None
        if export_graphs:
            from integration import attach_spatial_artifacts

            graph_path = export_spatial_graph_summary(
                quant_path,
                results_root / "summary" / "spatial",
                graph_method=graph_method,
            )
            if results_root.is_dir():
                for method_dir in results_root.iterdir():
                    if method_dir.is_dir() and method_dir.name not in {"summary", "logs"}:
                        attach_spatial_artifacts(
                            quant_path, method_dir, graph_method=graph_method, export_edges=False,
                        )
        stages["spatial"] = {
            "smoothed_files": n_files,
            "graph_summary": str(graph_path) if graph_path else None,
        }
    else:
        stages["spatial"] = "skipped"

    # Stage 5 — Evaluation (+ plots unless --skip_viz)
    if not skip_eval:
        eval_cmd = [
            sys.executable,
            str(_SRC / "evaluation" / "run_evaluation.py"),
            "--dataset", dataset,
            "--results_dir", str(root / "results"),
        ]
        if not skip_viz:
            eval_cmd.append("--plot")
        if unlabeled:
            eval_cmd.append("--unlabeled")
        if parent_log_active:
            eval_cmd = _append_no_log(eval_cmd)
        _run_script(eval_cmd, stage="evaluation")
        stages["evaluation"] = str(root / "results" / dataset / "summary" / "final_results.csv")
    else:
        stages["evaluation"] = "skipped"

    if skip_viz:
        stages["visualization"] = "skipped"
    elif skip_eval:
        stages["visualization"] = "skipped (evaluation skipped)"
    else:
        stages["visualization"] = "included via evaluation --plot"

    return stages


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="spCellEval end-to-end pipeline.")
    p.add_argument("--dataset", required=True, help="Dataset name (IMMUcan, CRC_TMA, …).")
    p.add_argument("--methods", nargs="+", help="Limit benchmark to specific method IDs.")
    p.add_argument("--root_dir", type=Path, default=_REPO)
    p.add_argument("--skip_preprocess", action="store_true")
    p.add_argument("--skip_stage2", action="store_true", help="Skip feature separation / pseudo-labeling.")
    p.add_argument("--skip_benchmark", action="store_true")
    p.add_argument("--skip_spatial", action="store_true")
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--skip_viz", action="store_true")
    p.add_argument("--no_spatial_smooth", action="store_true")
    p.add_argument("--no_graph_export", action="store_true")
    p.add_argument("--pseudo_label", action="store_true", help="Run Stage 2 pseudo-labeling.")
    p.add_argument("--pseudo_method", default="signature", choices=["signature", "tacit"])
    p.add_argument(
        "--unlabeled",
        action="store_true",
        help="Clinical/unannotated mode: skip GT ingest & k-folds; run signature/scyan/leiden.",
    )
    p.add_argument("--recreate_kfolds", action="store_true")
    p.add_argument(
        "--benchmark_config",
        type=Path,
        default=None,
        help="Benchmark YAML (default: configs/benchmark.yaml).",
    )
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


def main() -> None:
    args = build_parser().parse_args()
    root = args.root_dir.resolve()

    def _execute() -> dict:
        stages = run_pipeline(
            args.dataset,
            methods=args.methods,
            skip_preprocess=args.skip_preprocess,
            skip_stage2=args.skip_stage2,
            skip_benchmark=args.skip_benchmark,
            skip_spatial=args.skip_spatial,
            skip_eval=args.skip_eval,
            skip_viz=args.skip_viz,
            recreate_kfolds=args.recreate_kfolds,
            spatial_smooth=not args.no_spatial_smooth,
            export_graphs=not args.no_graph_export,
            pseudo_label=args.pseudo_label,
            pseudo_method=args.pseudo_method,
            unlabeled=args.unlabeled,
            benchmark_config=args.benchmark_config,
            root=root,
            parent_log_active=not args.no_log_file,
        )

        print("\n" + "=" * 60)
        print(f"Pipeline complete for {args.dataset}")
        for stage, info in stages.items():
            print(f"  {stage}: {info}")
        print("=" * 60)
        return stages

    if args.no_log_file:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            level=logging.DEBUG if args.verbose else logging.INFO,
        )
        _execute()
        return

    with PipelineLogSession(
        root=root,
        dataset=args.dataset,
        run_name="pipeline",
        verbose=args.verbose,
        log_dir=args.log_dir.resolve() if args.log_dir else None,
    ) as log_session:
        log_session.configure_logging()
        stages = _execute()
        log_session.add_manifest(stages=stages)
        print(f"\nFull execution log: {log_session.main_log}")


if __name__ == "__main__":
    main()

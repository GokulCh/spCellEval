"""
executors.py
============
Category-specific method executors for the Stage 3 benchmark runner.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from dataset_context import DatasetContext
from method_registry import MethodCategory, MethodSpec, resolve_artifact

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[2]
_METHODS_UTILS = _REPO / "src" / "methods" / "utils"
_PSEUDO_DIR = _REPO / "src" / "pseudo_labeling"
_EVAL_DIR = _REPO / "src" / "evaluation"
_SPATIAL_DIR = _REPO / "src" / "spatial"

for _p in [str(_METHODS_UTILS), str(_PSEUDO_DIR), str(_EVAL_DIR), str(_SPATIAL_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ground_truth import DEFAULT_EVAL_DROP_COLUMNS, resolve_markers_in_quant  # noqa: E402
from performance import write_run_manifest  # noqa: E402
from postprocess import postprocess_predictions_dir  # noqa: E402
from integration import attach_spatial_artifacts  # noqa: E402


def _peak_memory_mb() -> Optional[float]:
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _artifact_path(ctx: DatasetContext, key: str) -> Optional[Path]:
    """Resolve artifact from benchmark config or method_registry defaults."""
    bench = ctx.config.get("benchmark", {}).get("artifacts", {})
    rel = bench.get(key) or resolve_artifact(ctx.dataset_name, key)
    return (ctx.root / rel).resolve() if rel else None


class MethodExecutionError(Exception):
    pass


def _check_tool(name: str) -> bool:
    return shutil.which(name) is not None


def _image_dataset_slug(ctx: DatasetContext) -> str:
    """Return the dataset slug expected by image-method scripts."""
    if "immucan" in ctx.dataset_name.lower():
        return "immucan"
    if "crc" in ctx.dataset_name.lower():
        return "crc_tma"
    raise MethodExecutionError(
        f"{ctx.dataset_name} is not configured for image-based methods "
        f"({ctx.dataset_name} ≠ IMMUcan/CRC_TMA). Remove those methods from "
        f"benchmark.yaml for this dataset."
    )


def _image_pipeline_cfg(ctx: DatasetContext, method_id: str) -> dict:
    """Per-dataset options for image pipelines, from the `image_pipelines:` block."""
    return ctx.config.get("image_pipelines", {}).get(method_id, {})


def _resolve_cfg_path(ctx: DatasetContext, rel) -> Optional[Path]:
    if not rel:
        return None
    p = Path(str(rel))
    return p if p.is_absolute() else ctx.root / p


def _require_cfg_path(path: Optional[Path], ctx: DatasetContext, method_id: str, key: str) -> None:
    if path is None or not path.exists():
        raise MethodExecutionError(
            f"Missing prerequisite for image pipeline {method_id}: '{key}' = "
            f"{path or '<not set>'}. Set image_pipelines.{method_id}.{key} in "
            f"{ctx.config_path.name} (relative to the repo root)."
        )


def _run_cmd(cmd: List[str], cwd: Optional[Path] = None, timeout: Optional[int] = None) -> None:
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(cwd or _REPO),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise MethodExecutionError(
            f"Command failed (exit {result.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stdout: {result.stdout[-2000:]}\n"
            f"  stderr: {result.stderr[-2000:]}"
        )
    if result.stdout.strip():
        logger.debug(result.stdout[-500:])


def _dumb_columns_for_ml(ctx: DatasetContext) -> List[str]:
    """Columns to drop from k-fold feature matrices (Stage 2 ground-truth list).

    Includes standard metadata, label columns, AND spatial coordinate variants
    (``SPATIAL_COLUMNS``) so no ``X``/``Y``/``Pos_X`` obs columns ever leak into
    model features even when k-fold CSVs contain them.
    """
    from ground_truth import SPATIAL_COLUMNS  # noqa: WPS433

    extra = [c for c in SPATIAL_COLUMNS if c not in DEFAULT_EVAL_DROP_COLUMNS]
    return [c for c in [*DEFAULT_EVAL_DROP_COLUMNS, *extra] if c not in ctx.markers]


# ── Supervised k-fold ─────────────────────────────────────────────────────

def run_supervised_kfold(ctx: DatasetContext, spec: MethodSpec, *, ml_n_jobs: int = -1) -> Path:
    from kfold_strategies import result_method_id  # noqa: WPS433

    primary_out: Optional[Path] = None
    strategies: List[Tuple[str, Path]] = []

    def _run_one(kfold_method: str) -> Tuple[str, Path]:
        start = time.time()
        out = _run_supervised_kfold_strategy(ctx, spec, kfold_method, ml_n_jobs=ml_n_jobs)
        write_run_manifest(
            out / "benchmark_manifest.json",
            result_method_id(spec.id, kfold_method),
            elapsed_sec=time.time() - start,
        )
        return kfold_method, out

    if len(ctx.kfold_methods) > 1 and ml_n_jobs == -1:
        with ThreadPoolExecutor(max_workers=len(ctx.kfold_methods)) as pool:
            futures = [pool.submit(_run_one, m) for m in ctx.kfold_methods]
            for future in as_completed(futures):
                kfold_method, out = future.result()
                strategies.append((kfold_method, out))
    else:
        for kfold_method in ctx.kfold_methods:
            strategies.append(_run_one(kfold_method))

    for kfold_method, out in strategies:
        if kfold_method == "StratifiedKFold" or primary_out is None:
            primary_out = out
    if primary_out is None:
        raise MethodExecutionError(f"No k-fold results produced for '{spec.id}'")
    return primary_out


def _run_supervised_kfold_strategy(
    ctx: DatasetContext,
    spec: MethodSpec,
    kfold_method: str,
    *,
    ml_n_jobs: int = -1,
) -> Path:
    out = ctx.results_dir(spec.id, kfold_method)
    out.mkdir(parents=True, exist_ok=True)

    if spec.id in ("random_forest", "logistic_regression", "xgboost", "svm"):
        return _run_classic_ml(ctx, spec.id, out, kfold_method, n_jobs=ml_n_jobs)

    if spec.id == "maps":
        return _run_maps(ctx, out, kfold_method)

    if spec.id in ("singler", "scarches", "ribca_adapted"):
        return _run_reference_mapping(ctx, spec.id, out, kfold_method)

    raise MethodExecutionError(f"No supervised executor for '{spec.id}'")


def _run_classic_ml(ctx: DatasetContext, model: str, out: Path, kfold_method: str, *, n_jobs: int = -1) -> Path:
    from default_classic_ml_models_kfolds import ClassicMLDefault  # noqa: WPS433

    kdir = ctx.kfold_dir(kfold_method)
    labels = ctx.labels_path(kfold_method)
    if not kdir.is_dir():
        raise MethodExecutionError(f"K-fold directory not found: {kdir}")

    dumb = _dumb_columns_for_ml(ctx)
    clf = ClassicMLDefault(random_state=42, model=model, n_jobs=n_jobs)
    clf.train_tune_evaluate(str(kdir), str(labels), verbose=0, scaling=True, dumb_columns=dumb)
    clf.save_results(str(out), str(labels), str(kdir), save_model=False)
    try:
        clf.save_feature_importances(str(out), str(labels), str(kdir))
    except Exception as exc:  # feature export must never block the run
        logger.warning("Feature importance export skipped: %s", exc)
    logger.info("Classic ML (%s, %s) → %s", model, kfold_method, out)
    return out


def _run_maps(ctx: DatasetContext, out: Path, kfold_method: str) -> Path:
    cmd = [
        sys.executable,
        str(_REPO / "src" / "methods" / "MAPS" / "run_maps.py"),
        str(ctx.kfold_dir(kfold_method)),
        str(out),
        str(ctx.labels_path(kfold_method)),
    ]
    _run_cmd(cmd)
    return out


def _run_reference_mapping(
    ctx: DatasetContext,
    method_id: str,
    out: Path,
    kfold_method: str,
) -> Path:
    from reference_mapping import (  # noqa: WPS433
        ribca_predict,
        run_kfold_label_transfer,
        scarches_predict,
        singler_predict,
    )

    kdir = ctx.kfold_dir(kfold_method)
    labels = ctx.labels_path(kfold_method)
    if not kdir.is_dir():
        raise MethodExecutionError(f"K-fold directory not found: {kdir}")
    if not labels.is_file():
        raise MethodExecutionError(f"Labels file not found: {labels}")

    if method_id == "ribca_adapted":
        predict_fn = ribca_predict
    else:
        predict_fn = singler_predict if method_id == "singler" else scarches_predict
    run_kfold_label_transfer(
        kdir,
        labels,
        out,
        ctx.markers,
        predict_fn,
        dumb_columns=_dumb_columns_for_ml(ctx),
    )
    logger.info("Reference mapping (%s, %s) → %s", method_id, kfold_method, out)
    return out


# ── Unsupervised (quant CSV) ──────────────────────────────────────────────

def run_unsupervised(ctx: DatasetContext, spec: MethodSpec, iterations: int = 1) -> Path:
    if not ctx.quant_path.exists():
        raise MethodExecutionError(f"Quantification CSV not found: {ctx.quant_path}")

    out = ctx.results_dir(spec.id)
    out.mkdir(parents=True, exist_ok=True)
    markers = resolve_markers_in_quant(ctx.quant_path, ctx.markers)

    if spec.id == "leiden":
        cmd = [
            sys.executable,
            str(spec.script),
            "-i", str(ctx.quant_path),
            "-o", str(out),
            "-m", *markers,
            "-it", str(iterations),
            "-r", "1.0",
            "-l", "off",
        ]
        _run_cmd(cmd)
        return out

    if spec.id == "louvain":
        cmd = [
            sys.executable,
            str(spec.script),
            "-i", str(ctx.quant_path),
            "-o", str(out),
            "-m", *markers,
            "-it", str(iterations),
            "-r", "1.0",
            "-l", "off",
        ]
        _run_cmd(cmd)
        return out

    if spec.id == "spade":
        n_clusters = min(20, max(5, len(markers)))
        cmd = [
            sys.executable,
            str(spec.script),
            "-i", str(ctx.quant_path),
            "-o", str(out),
            "-m", *markers,
            "-it", str(iterations),
            "-k", str(n_clusters),
            "-l", "off",
        ]
        _run_cmd(cmd)
        return out

    if spec.id == "starling":
        cmd = [
            sys.executable,
            str(spec.script),
            "--dataset_path", str(ctx.quant_path),
            "--split_col_name", ctx.split_col,
            "--output_path", str(out),
            "--n_runs", str(iterations),
            "--transform", "none",
        ]
        _run_cmd(cmd)
        return out

    if spec.id == "flowsom":
        if not _check_tool("Rscript"):
            raise MethodExecutionError("Rscript not found — install R for FlowSOM.")
        n_clusters = min(20, max(5, len(markers)))
        cmd = [
            "Rscript", str(spec.script),
            "-i", str(ctx.quant_path),
            "-m", *markers,
            "-o", str(out),
            "-flow", str(n_clusters),
            "-it", str(iterations),
        ]
        _run_cmd(cmd)
        return out

    if spec.id == "phenograph":
        if not _check_tool("Rscript"):
            raise MethodExecutionError("Rscript not found — install R for Phenograph.")
        cmd = [
            "Rscript", str(spec.script),
            "-i", str(ctx.quant_path),
            "-m", *markers,
            "-o", str(out),
            "-k", "30",
            "-it", str(iterations),
        ]
        _run_cmd(cmd)
        return out

    if spec.id == "fusesom":
        if not _check_tool("Rscript"):
            raise MethodExecutionError("Rscript not found — install R for FuseSOM.")
        cmd = [
            "Rscript", str(spec.script),
            "-i", str(ctx.quant_path),
            "-m", *markers,
            "-o", str(out),
            "-fuse", "15",
            "-it", str(iterations),
        ]
        _run_cmd(cmd)
        return out

    raise MethodExecutionError(f"No unsupervised executor for '{spec.id}'")


# ── Prior-knowledge ───────────────────────────────────────────────────────

def run_prior_knowledge(ctx: DatasetContext, spec: MethodSpec, iterations: int = 1) -> Path:
    if not ctx.quant_path.exists():
        raise MethodExecutionError(f"Quantification CSV not found: {ctx.quant_path}")

    out = ctx.results_dir(spec.id)
    out.mkdir(parents=True, exist_ok=True)

    if spec.id == "signature":
        cmd = [
            sys.executable,
            str(_PSEUDO_DIR / "run_pseudo_labeler.py"),
            "--config", str(ctx.config_path),
            "--root_dir", str(ctx.root),
            "--method", "signature",
            "--output", str(out),
        ]
        _run_cmd(cmd)
        return out

    if spec.id == "tacit":
        dm = _artifact_path(ctx, "decision_matrix_tacit")
        if not dm or not dm.exists():
            raise MethodExecutionError(
                f"No TACIT decision matrix configured for {ctx.dataset_name}."
            )
        if not _check_tool("Rscript"):
            raise MethodExecutionError("Rscript not found — install R + TACIT package.")
        from run_tacit import run_tacit  # noqa: WPS433

        run_tacit(
            input_path=ctx.quant_path,
            decision_matrix_path=dm,
            output_path=out,
            config=ctx.config,
            iterations=iterations,
        )
        return out

    if spec.id == "scyan":
        dm = _artifact_path(ctx, "decision_matrix_scyan")
        if not dm or not dm.exists():
            raise MethodExecutionError(f"No Scyan matrix for {ctx.dataset_name}.")
        cmd = [
            sys.executable,
            str(spec.script),
            "--dataset_path", str(ctx.quant_path),
            "--decision_matrix_path", str(dm),
            "--split_col", ctx.split_col,
            "--output_path", str(out),
            "--n_runs", str(iterations),
            "--granularity_level", ctx.granularity,
            "--accelerator", "cpu",
        ]
        _run_cmd(cmd, timeout=3600)
        return out

    if spec.id == "astir":
        dm = _artifact_path(ctx, "decision_matrix_astir")
        if not dm or not dm.exists():
            raise MethodExecutionError(f"No Astir YAML for {ctx.dataset_name}.")
        cmd = [
            sys.executable,
            str(spec.script),
            "--quant_path", str(ctx.quant_path),
            "--decision_matrix_path", str(dm),
            "--separate_col", ctx.split_col,
            "--output_path", str(out),
            "--n_runs", str(iterations),
            "--device", "cpu",
        ]
        _run_cmd(cmd, timeout=3600)
        return out

    if spec.id == "tribus":
        dm = _artifact_path(ctx, "decision_matrix_tribus")
        if not dm or not dm.exists():
            raise MethodExecutionError(
                f"No Tribus decision matrix for {ctx.dataset_name}. "
                "Add benchmark.artifacts.decision_matrix_tribus to the dataset config."
            )
        cmd = [
            sys.executable,
            str(spec.script),
            "--dataset_path", str(ctx.quant_path),
            "--decision_matrix_path", str(dm),
            "--columns_to_use", ",".join(ctx.markers),
            "--output_path", str(out),
            "-n", str(iterations),
        ]
        _run_cmd(cmd, timeout=3600)
        return out

    raise MethodExecutionError(f"No prior-knowledge executor for '{spec.id}'")


# ── Image pipelines ───────────────────────────────────────────────────────

def run_image_pipeline(ctx: DatasetContext, spec: MethodSpec) -> Path:
    if ctx.image_data_dir is None or not ctx.image_data_dir.exists():
        raise MethodExecutionError(
            f"{spec.display_name} requires raw images. "
            f"Set benchmark.image_data_dir in configs/benchmark.yaml for {ctx.dataset_name}."
        )

    out = ctx.results_dir(spec.id)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = ctx.image_data_dir

    if spec.id == "stellar":
        ds_flag = _image_dataset_slug(ctx)
        cmd = [
            sys.executable, str(spec.script),
            "--dataset", ds_flag,
            "--data-base-dir", str(data_dir),
            "--output-dir", str(out.parent),
            "--device", "cpu",
        ]
        kfold_json = ctx.kfold_dir() / "fold_indices.json"
        if kfold_json.is_file():
            cmd.extend(["--folds-json", str(kfold_json)])
        if ctx.quant_path.is_file():
            cmd.extend(["--quant-csv", str(ctx.quant_path)])
        _run_cmd(cmd, timeout=7200)
        return out

    if spec.id == "cellsighter":
        cmd = [
            sys.executable, str(spec.script),
            "--dataset", _image_dataset_slug(ctx),
            "--config", str(_REPO / "src" / "methods" / "CellSighter" / "cellsighter.json"),
            "--results_dir", str(out),
            "--output_root", str(data_dir),
        ]
        _run_cmd(cmd)
        return out

    if spec.id.startswith("eva_"):
        mode = "supervised" if "supervised" in spec.id else "leiden"
        cmd = [
            sys.executable, str(spec.script),
            mode,
            "--dataset", _image_dataset_slug(ctx),
            "--data-dir", str(data_dir),
            "--output-dir", str(out.parent),
            "--spceleval-dir", str(ctx.root),
            "--device", "cpu",
        ]
        _run_cmd(cmd, timeout=7200)
        return out

    if spec.id.startswith("kronos_"):
        mode = "supervised" if "supervised" in spec.id else "leiden"
        cmd = [
            sys.executable, str(spec.script),
            mode,
            "--data-dir", str(data_dir),
            "--output-dir", str(out.parent),
            "--spceleval-dir", str(ctx.root),
            "--device", "cpu",
        ]
        _run_cmd(cmd, timeout=7200)
        return out

    if spec.id.startswith("virtues_"):
        mode = "supervised" if "supervised" in spec.id else "leiden"
        virtues_dir = _REPO / "src" / "methods" / "VirTues"
        cmd = [
            sys.executable, str(spec.script),
            mode,
            "--dataset", _image_dataset_slug(ctx),
            "--data-dir", str(data_dir),
            "--output-dir", str(out.parent),
            "--spceleval-dir", str(ctx.root),
            "--virtues-dir", str(virtues_dir),
            "--device", "cpu",
        ]
        _run_cmd(cmd, timeout=7200)
        return out

    if spec.id == "celllens":
        script_path = _artifact_path(ctx, "celllens_script")
        if not script_path or not script_path.exists():
            raise MethodExecutionError(f"No CellLENS script for {ctx.dataset_name}.")
        cmd = [
            sys.executable, str(script_path),
            "-i", str(ctx.quant_path),
            "-o", str(out),
            "-m", *ctx.markers,
            "-it", "1",
        ]
        _run_cmd(cmd, timeout=7200)
        return out

    if spec.id == "nimbus":
        opts = _image_pipeline_cfg(ctx, "nimbus")
        images_dir = _resolve_cfg_path(ctx, opts.get("images_dir"))
        seg_dir = _resolve_cfg_path(ctx, opts.get("seg_mask_dir"))
        mpp = opts.get("mpp")
        _require_cfg_path(images_dir, ctx, "nimbus", "images_dir")
        _require_cfg_path(seg_dir, ctx, "nimbus", "seg_mask_dir")
        if not mpp:
            raise MethodExecutionError(
                f"nimbus requires 'mpp' under image_pipelines.nimbus in {ctx.config_path.name}."
            )
        cmd = [
            sys.executable, str(spec.script),
            "-i", str(images_dir),
            "-s", str(seg_dir),
            "-o", str(out.parent),
            "-m", *ctx.markers,
            "--input_type", str(opts.get("input_type", "single")),
            "--image_suffix", str(opts.get("image_suffix", "tiff")),
            "--mask_type", str(opts.get("mask_type", "tiff")),
            "-it", str(opts.get("iterations", 1)),
            "--mpp", str(mpp),
        ]
        if opts.get("log") in ("short", "long"):
            cmd.extend(["-l", str(opts["log"])])
        _run_cmd(cmd, timeout=7200)
        return out

    if spec.id == "deepcelltypes":
        opts = _image_pipeline_cfg(ctx, "deepcelltypes")
        images_dir = _resolve_cfg_path(ctx, opts.get("images_dir"))
        masks_dir = _resolve_cfg_path(ctx, opts.get("masks_dir"))
        marker_path = _resolve_cfg_path(ctx, opts.get("marker_path"))
        mpp = opts.get("mpp")
        model_name = opts.get("model_name")
        _require_cfg_path(images_dir, ctx, "deepcelltypes", "images_dir")
        _require_cfg_path(masks_dir, ctx, "deepcelltypes", "masks_dir")
        _require_cfg_path(marker_path, ctx, "deepcelltypes", "marker_path")
        if not mpp:
            raise MethodExecutionError(
                f"deepcelltypes requires 'mpp' under image_pipelines.deepcelltypes in {ctx.config_path.name}."
            )
        if not model_name:
            raise MethodExecutionError(
                f"deepcelltypes requires 'model_name' under image_pipelines.deepcelltypes in {ctx.config_path.name}."
            )
        cmd = [
            sys.executable, str(spec.script),
            "--input_dirs", str(images_dir), str(masks_dir),
            "--marker_path", str(marker_path),
            "--quant_path", str(ctx.quant_path),
            "--mpp", str(mpp),
            "--model_name", str(model_name),
            "--output_dir", str(out),
            "--device", str(opts.get("device", "cuda")),
            "--num_data_loader_threads", str(opts.get("num_data_loader_threads", 4)),
            "--n_runs", str(opts.get("n_runs", 1)),
        ]
        if opts.get("strip_extensions"):
            cmd.append("--strip_extensions")
        if opts.get("rename_rules"):
            rr = _resolve_cfg_path(ctx, opts["rename_rules"])
            _require_cfg_path(rr, ctx, "deepcelltypes", "rename_rules")
            cmd.extend(["--rename_rules", str(rr)])
        _run_cmd(cmd, timeout=7200)
        return out

    raise MethodExecutionError(
        f"Image pipeline '{spec.id}' is registered but not yet wired in the benchmark runner. "
        "Run it manually via its script in src/methods/."
    )


def execute_method(
    ctx: DatasetContext,
    spec: MethodSpec,
    *,
    iterations: int = 1,
    skip_missing_deps: bool = True,
    spatial_smooth: bool = False,
    spatial_k_neighbors: int = 15,
    ml_n_jobs: int = -1,
) -> Optional[Path]:
    """Run one method and return its output directory (or None if skipped)."""
    if spec.requires_r and not _check_tool("Rscript"):
        msg = f"Skipping {spec.id}: Rscript not found."
        if skip_missing_deps:
            logger.warning(msg)
            return None
        raise MethodExecutionError(msg)

    if spec.requires_raw_images and (ctx.image_data_dir is None or not ctx.image_data_dir.exists()):
        msg = (
            f"Skipping {spec.id}: image_data_dir not configured for {ctx.dataset_name}."
        )
        if skip_missing_deps:
            logger.warning(msg)
            return None
        raise MethodExecutionError(msg)

    start = time.time()
    mem_before = _peak_memory_mb()
    try:
        if spec.category == MethodCategory.SUPERVISED_KFOLD:
            result = run_supervised_kfold(ctx, spec, ml_n_jobs=ml_n_jobs)
        elif spec.category == MethodCategory.UNSUPERVISED_QUANT:
            result = run_unsupervised(ctx, spec, iterations=iterations)
        elif spec.category == MethodCategory.PRIOR_KNOWLEDGE:
            result = run_prior_knowledge(ctx, spec, iterations=iterations)
        elif spec.category == MethodCategory.IMAGE_PIPELINE:
            result = run_image_pipeline(ctx, spec)
        else:
            raise MethodExecutionError(f"Unknown category: {spec.category}")

        elapsed = time.time() - start
        mem_after = _peak_memory_mb()
        peak = max(v for v in [mem_before, mem_after] if v is not None) if mem_before or mem_after else None
        if spec.category != MethodCategory.SUPERVISED_KFOLD:
            write_run_manifest(
                result / "benchmark_manifest.json",
                spec.id,
                elapsed_sec=elapsed,
                peak_memory_mb=peak,
            )

        if spatial_smooth and result is not None:
            n = postprocess_predictions_dir(
                result,
                ctx.quant_path,
                k_neighbors=spatial_k_neighbors,
            )
            if n:
                logger.info("Spatial smoothing applied to %d prediction file(s).", n)

        if result is not None and ctx.quant_path.is_file():
            try:
                attach_spatial_artifacts(ctx.quant_path, result)
            except Exception as exc:
                logger.warning("Spatial artifact export skipped: %s", exc)

        logger.info(
            "OK %s finished in %.1f s -> %s",
            spec.display_name,
            elapsed,
            result,
        )
        return result
    except MethodExecutionError:
        raise
    except Exception as exc:
        raise MethodExecutionError(f"{spec.id} failed: {exc}") from exc

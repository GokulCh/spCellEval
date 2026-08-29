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
from typing import List, Optional

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

from ground_truth import DEFAULT_EVAL_DROP_COLUMNS  # noqa: E402
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
    """Columns to drop from k-fold feature matrices (Stage 2 ground-truth list)."""
    return [c for c in DEFAULT_EVAL_DROP_COLUMNS if c not in ctx.markers]


# ── Supervised k-fold ─────────────────────────────────────────────────────

def run_supervised_kfold(ctx: DatasetContext, spec: MethodSpec) -> Path:
    ctx.ensure_kfolds(strip_labels=True)
    out = ctx.results_dir(spec.id)
    out.mkdir(parents=True, exist_ok=True)

    if spec.id in ("random_forest", "logistic_regression", "xgboost"):
        return _run_classic_ml(ctx, spec.id, out)

    if spec.id == "maps":
        return _run_maps(ctx, out)

    raise MethodExecutionError(f"No supervised executor for '{spec.id}'")


def _run_classic_ml(ctx: DatasetContext, model: str, out: Path) -> Path:
    from default_classic_ml_models_kfolds import ClassicMLDefault  # noqa: WPS433

    kdir = ctx.kfold_dir()
    labels = ctx.labels_path()
    if not kdir.is_dir():
        raise MethodExecutionError(f"K-fold directory not found: {kdir}")

    dumb = _dumb_columns_for_ml(ctx)
    clf = ClassicMLDefault(random_state=42, model=model, n_jobs=-1)
    clf.train_tune_evaluate(str(kdir), str(labels), verbose=0, scaling=True, dumb_columns=dumb)
    clf.save_results(str(out), str(labels), str(kdir), save_model=False)
    logger.info("Classic ML (%s) → %s", model, out)
    return out


def _run_maps(ctx: DatasetContext, out: Path) -> Path:
    cmd = [
        sys.executable,
        str(_REPO / "src" / "methods" / "MAPS" / "run_maps.py"),
        str(ctx.kfold_dir()),
        str(out),
        str(ctx.labels_path()),
    ]
    _run_cmd(cmd)
    return out


# ── Unsupervised (quant CSV) ──────────────────────────────────────────────

def run_unsupervised(ctx: DatasetContext, spec: MethodSpec, iterations: int = 1) -> Path:
    if not ctx.quant_path.exists():
        raise MethodExecutionError(f"Quantification CSV not found: {ctx.quant_path}")

    out = ctx.results_dir(spec.id)
    out.mkdir(parents=True, exist_ok=True)
    markers = ctx.markers

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
        ds_flag = "immucan" if "immucan" in ctx.dataset_name.lower() else "chl"
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
            "--dataset", "immucan" if "immucan" in ctx.dataset_name.lower() else "chl",
            "--config", str(_REPO / "src" / "methods" / "CellSighter" / "cellsighter.json"),
            "--results_dir", str(out),
            "--output_root", str(data_dir),
        ]
        _run_cmd(cmd)
        return out

    if spec.id.startswith("eva_"):
        mode = "supervised" if "supervised" in spec.id else "leiden"
        ds = "immucan" if "immucan" in ctx.dataset_name.lower() else "chl"
        cmd = [
            sys.executable, str(spec.script),
            mode,
            "--dataset", ds,
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
        ds = "immucan" if "immucan" in ctx.dataset_name.lower() else "chl"
        virtues_dir = _REPO / "src" / "methods" / "VirTues"
        cmd = [
            sys.executable, str(spec.script),
            mode,
            "--dataset", ds,
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
            result = run_supervised_kfold(ctx, spec)
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

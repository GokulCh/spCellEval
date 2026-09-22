"""
parallel_runner.py
==================
Process-pool worker for independent benchmark method runs.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parents[2]
_BENCH = Path(__file__).resolve().parent
for _p in (str(_BENCH), str(_REPO / "src" / "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset_context import DatasetContext  # noqa: E402
from executors import MethodExecutionError, execute_method  # noqa: E402
from method_registry import METHOD_REGISTRY  # noqa: E402

logger = logging.getLogger(__name__)


def _init_worker_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            level=logging.INFO,
        )


def run_method_job(payload: Dict[str, Any]) -> Tuple[str, str, Optional[str], Optional[str]]:
    """Execute one benchmark method in a worker process.

    Returns ``(method_id, status, path, error)`` where status is
    ``succeeded``, ``skipped``, or ``failed``.
    """
    _init_worker_logging()
    method_id = payload["method_id"]
    spec = METHOD_REGISTRY.get(method_id)
    if spec is None:
        return method_id, "failed", None, "unknown method"

    ctx = DatasetContext.from_config(
        Path(payload["config_path"]),
        root=Path(payload["root"]),
        benchmark_overrides=payload.get("benchmark_overrides", {}),
    )
    if payload.get("artifacts"):
        ctx.config.setdefault("benchmark", {})["artifacts"] = payload["artifacts"]

    try:
        out = execute_method(
            ctx,
            spec,
            iterations=int(payload.get("iterations", 1)),
            skip_missing_deps=bool(payload.get("skip_missing_deps", True)),
            spatial_smooth=bool(payload.get("spatial_smooth", False)),
            spatial_k_neighbors=int(payload.get("spatial_k_neighbors", 15)),
            ml_n_jobs=int(payload.get("ml_n_jobs", -1)),
            method_timeout=payload.get("method_timeout"),
        )
        if out is None:
            return method_id, "skipped", None, None
        return method_id, "succeeded", str(out), None
    except MethodExecutionError as exc:
        return method_id, "failed", None, str(exc)
    except Exception as exc:
        return method_id, "failed", None, str(exc)


def build_method_payloads(
    *,
    methods: List[str],
    config_path: Path,
    root: Path,
    benchmark_overrides: Dict[str, Any],
    artifacts: Optional[Dict[str, str]],
    iterations: int,
    skip_missing_deps: bool,
    spatial_smooth: bool,
    spatial_k_neighbors: int,
    ml_n_jobs: int,
    method_timeout: Optional[int] = None,
) -> List[Dict[str, Any]]:
    return [
        {
            "method_id": method_id,
            "config_path": str(config_path),
            "root": str(root),
            "benchmark_overrides": benchmark_overrides,
            "artifacts": artifacts,
            "iterations": iterations,
            "skip_missing_deps": skip_missing_deps,
            "spatial_smooth": spatial_smooth,
            "spatial_k_neighbors": spatial_k_neighbors,
            "ml_n_jobs": ml_n_jobs,
            "method_timeout": method_timeout,
        }
        for method_id in methods
    ]

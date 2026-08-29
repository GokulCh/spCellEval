"""
run_tacit.py
============
Python wrapper that prepares a TACIT-compatible input CSV and invokes the
existing R runner in ``src/methods/TACIT/run_TACIT.R``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# Path bootstrap
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_TACIT_R_SCRIPT = _REPO_ROOT / "src" / "methods" / "TACIT" / "run_TACIT.R"

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ground_truth import (  # noqa: E402
    infer_separate_col,
    load_dataset_config,
    prepare_tacit_input,
)

logger = logging.getLogger(__name__)


def _find_rscript() -> str:
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError(
            "Rscript not found on PATH. Install R and the TACIT package, "
            "or use --method signature for a pure-Python fallback."
        )
    return rscript


def run_tacit(
    input_path: str | Path,
    decision_matrix_path: str | Path,
    output_path: str | Path,
    config: Optional[Dict[str, Any]] = None,
    separate_col: Optional[str] = None,
    scaling: Optional[float] = None,
    log1p: bool = False,
    iterations: int = 5,
    r: int = 10,
    p: int = 10,
    pseudo_label_column: Optional[str] = None,
) -> Path:
    """Run TACIT via Rscript and return the output directory."""
    input_path = Path(input_path)
    decision_matrix_path = Path(decision_matrix_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if config is None and separate_col is None:
        raise ValueError("Provide either a dataset config dict or separate_col.")

    if separate_col is None:
        assert config is not None
        separate_col = infer_separate_col(config)

    df = pd.read_csv(input_path)
    tacit_df = prepare_tacit_input(df, config or {}, pseudo_label_column=pseudo_label_column)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = Path(tmp.name)
        tacit_df.to_csv(tmp_path, index=False)

    rscript = _find_rscript()
    cmd = [
        rscript,
        str(_TACIT_R_SCRIPT),
        "--input_path", str(tmp_path),
        "--decision_matrix_path", str(decision_matrix_path),
        "--separate_col", separate_col,
        "--output_path", str(output_path),
        "-n", str(iterations),
        "-r", str(r),
        "-p", str(p),
    ]
    if scaling is not None:
        cmd.extend(["--scaling", str(scaling)])
    if log1p:
        cmd.append("--log1p")

    logger.info("Running TACIT: %s", " ".join(cmd))
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    finally:
        tmp_path.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"TACIT failed (exit {result.returncode}).\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    elapsed = time.time() - start
    logger.info("TACIT completed in %.1f s. Output → %s", elapsed, output_path)
    if result.stdout.strip():
        logger.debug(result.stdout)
    return output_path


def main() -> None:
    import argparse

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(description="Run TACIT pseudo-labeler (R wrapper).")
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--decision_matrix_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None, help="Dataset YAML config.")
    parser.add_argument("--separate_col", type=str, default=None)
    parser.add_argument("--scaling", type=float, default=None)
    parser.add_argument("--log1p", action="store_true")
    parser.add_argument("-n", "--iterations", type=int, default=5)
    parser.add_argument("-r", type=int, default=10)
    parser.add_argument("-p", type=int, default=10)
    args = parser.parse_args()

    config = load_dataset_config(args.config) if args.config else None
    run_tacit(
        input_path=args.input_path,
        decision_matrix_path=args.decision_matrix_path,
        output_path=args.output_path,
        config=config,
        separate_col=args.separate_col,
        scaling=args.scaling,
        log1p=args.log1p,
        iterations=args.iterations,
        r=args.r,
        p=args.p,
    )


if __name__ == "__main__":
    main()

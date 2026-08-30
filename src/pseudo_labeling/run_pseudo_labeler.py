#!/usr/bin/env python
"""
run_pseudo_labeler.py
=====================
Unified CLI for Stage 2: automated pseudo-labeling and ground-truth management.

Usage
-----
Signature rules (pure Python, no R required)::

    python src/pseudo_labeling/run_pseudo_labeler.py \\
        --config configs/immucan.yaml \\
        --method signature

TACIT (requires R + TACIT package)::

    python src/pseudo_labeling/run_pseudo_labeler.py \\
        --config configs/immucan.yaml \\
        --method tacit

Strip labels from a quantification CSV (leakage prevention)::

    python src/pseudo_labeling/run_pseudo_labeler.py \\
        --config configs/immucan.yaml \\
        --strip_labels \\
        --output data/processed/IMMUcan/IMMUcan_features_only.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Path bootstrap
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in [str(_HERE), str(_REPO_ROOT / "src" / "preprocessing" / "utils")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ground_truth import (  # noqa: E402
    get_marker_columns,
    load_dataset_config,
    strip_ground_truth,
)
from signature_rules import apply_signature_rules, add_hierarchy_from_predictions  # noqa: E402

logger = logging.getLogger("run_pseudo_labeler")


def _resolve_processed_path(config: dict, root: Path) -> Path:
    out = config.get("output", {})
    processed_dir = root / out.get("processed_dir", "data/processed")
    filename = out.get("output_filename", f"{config['dataset_name']}_quantification.csv")
    return processed_dir / filename


def _resolve_output_dir(config: dict, root: Path, method: str) -> Path:
    pl = config.get("pseudo_labeling", {})
    if "output_dir" in pl:
        return root / pl["output_dir"]
    dataset = config.get("dataset_name", "dataset")
    return root / "results" / dataset / method / "level3"


def run_signature(
    df: pd.DataFrame,
    config: dict,
    root: Path,
    threshold: float,
    min_score: float,
) -> pd.DataFrame:
    pl = config.get("pseudo_labeling", {})
    dm_path = root / pl.get(
        "decision_matrix",
        "src/methods/TACIT/IMMUcan_decision_matrix_level3.csv",
    )
    marker_cols = get_marker_columns(config)
    predictions = apply_signature_rules(
        df,
        decision_matrix=dm_path,
        marker_cols=marker_cols,
        threshold=threshold,
        min_score=min_score,
    )
    out = df.copy()
    out["predicted_phenotype"] = predictions.values

    # Preserve ground truth as true_phenotype when available (evaluation contract).
    if "cell_type" in out.columns:
        out["true_phenotype"] = out["cell_type"]
    else:
        out = add_hierarchy_from_predictions(out, prediction_col="predicted_phenotype")
    return out


def write_annotated_quant(
    config: dict,
    root: Path,
    method: str,
    threshold: float,
    min_score: float,
) -> Path:
    """Pseudo-label an unlabeled quant CSV and save ``*_annotated.csv``."""
    input_path = _resolve_processed_path(config, root)
    if not input_path.is_file():
        raise FileNotFoundError(f"Quantification CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if method == "signature":
        annotated = run_signature(df, config, root, threshold, min_score)
    elif method == "tacit":
        raise ValueError(
            "write_annotated_quant supports signature only; run TACIT via --method tacit "
            "to write benchmark predictions."
        )
    else:
        raise ValueError(f"Unknown annotation method: {method}")

    annotated["cell_type"] = annotated["predicted_phenotype"]
    dataset = config.get("dataset_name", "dataset")
    out_cfg = config.get("output", {})
    out_dir = root / out_cfg.get("processed_dir", f"data/processed/{dataset}")
    out_path = out_dir / f"{dataset}_annotated.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(out_path, index=False)
    logger.info("Annotated quant (pseudo-labels) → %s", out_path)
    return out_path


def run_pseudo_labeling(
    config_path: Path,
    root: Path,
    method: str,
    threshold: float,
    min_score: float,
    output_dir: Path | None,
    n_iterations: int,
) -> Path:
    config = load_dataset_config(config_path)
    input_path = _resolve_processed_path(config, root)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Processed quantification not found: {input_path}\n"
            "Run preprocessing first: python src/preprocessing/run_preprocess.py --config ..."
        )

    logger.info("Loading %s …", input_path)
    df = pd.read_csv(input_path)
    out_dir = output_dir or _resolve_output_dir(config, root, method)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    if method == "signature":
        result = run_signature(df, config, root, threshold, min_score)
        out_file = out_dir / "predictions_1.csv"
        result.to_csv(out_file, index=False)
        with (out_dir / "fold_times.txt").open("w", encoding="utf-8") as fh:
            fh.write(f"Fold 1 inference_time: {time.time() - start:.2f} seconds\n")
        logger.info("Signature predictions → %s", out_file)

    elif method == "tacit":
        from run_tacit import run_tacit  # noqa: WPS433

        pl = config.get("pseudo_labeling", {})
        dm_path = root / pl.get(
            "decision_matrix",
            "src/methods/TACIT/IMMUcan_decision_matrix_level3.csv",
        )
        tacit_cfg = pl.get("tacit", {})
        run_tacit(
            input_path=input_path,
            decision_matrix_path=dm_path,
            output_path=out_dir,
            config=config,
            scaling=pl.get("scaling"),
            log1p=pl.get("log1p", False),
            iterations=n_iterations,
            r=tacit_cfg.get("r", 10),
            p=tacit_cfg.get("p", 10),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    elapsed = time.time() - start
    logger.info("Pseudo-labeling (%s) finished in %.1f s → %s", method, elapsed, out_dir)
    return out_dir


def run_strip_labels(
    config_path: Path,
    root: Path,
    output_path: Path | None,
    drop_metadata: bool,
) -> Path:
    config = load_dataset_config(config_path)
    input_path = _resolve_processed_path(config, root)
    df = pd.read_csv(input_path)

    features, labels = strip_ground_truth(df, drop_metadata=drop_metadata)

    if output_path is None:
        out_cfg = config.get("output", {})
        processed_dir = root / out_cfg.get("processed_dir", "data/processed")
        dataset = config.get("dataset_name", "dataset")
        output_path = processed_dir / f"{dataset}_features_only.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)

    labels_path = output_path.with_name(output_path.stem + "_labels_sidecar.csv")
    labels.to_csv(labels_path, index=False)

    logger.info(
        "Stripped %d label column(s). Features → %s  |  Labels → %s",
        len(df.columns) - len(features.columns),
        output_path,
        labels_path,
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="spCellEval Stage 2 — pseudo-labeling and ground-truth management.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Dataset YAML config (e.g. configs/immucan.yaml).",
    )
    parser.add_argument(
        "--root_dir",
        type=Path,
        default=_REPO_ROOT,
        help="Repository root for resolving relative paths.",
    )
    parser.add_argument(
        "--strip_labels",
        action="store_true",
        help="Strip ground-truth columns and write a features-only CSV.",
    )
    parser.add_argument(
        "--drop_metadata",
        action="store_true",
        help="With --strip_labels, also remove spatial/identity metadata.",
    )
    parser.add_argument(
        "--method",
        choices=["signature", "tacit"],
        default="signature",
        help="Pseudo-labeling method (default: signature).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Marker expression threshold for signature rules (default: from config).",
    )
    parser.add_argument(
        "--min_score",
        type=float,
        default=None,
        help="Minimum rule satisfaction fraction (default: from config).",
    )
    parser.add_argument(
        "--annotate_quant",
        action="store_true",
        help="Write pseudo-labels + hierarchy to data/processed/{dataset}_annotated.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file (strip_labels) or directory (pseudo-labeling).",
    )
    parser.add_argument(
        "-n", "--iterations",
        type=int,
        default=5,
        help="Number of TACIT iterations (default: 5).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    root = args.root_dir.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config

    if args.strip_labels:
        run_strip_labels(config_path, root, args.output, args.drop_metadata)
        return

    config = load_dataset_config(config_path)
    pl = config.get("pseudo_labeling", {})
    threshold = args.threshold if args.threshold is not None else pl.get("marker_threshold", 0.0)
    min_score = args.min_score if args.min_score is not None else pl.get("min_score", 1.0)

    if args.annotate_quant:
        write_annotated_quant(config, root, args.method, threshold, min_score)

    run_pseudo_labeling(
        config_path=config_path,
        root=root,
        method=args.method,
        threshold=threshold,
        min_score=min_score,
        output_dir=args.output,
        n_iterations=args.iterations,
    )


if __name__ == "__main__":
    main()

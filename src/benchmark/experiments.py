"""
experiments.py
==============
Objective experiment drivers for the Stage 3 / CRC_TMA benchmark suite.

This module is a STATIC driver: it never runs anything at import time and only
materialises artifacts when invoked via ``run_benchmark.py`` or its CLI.

Objectives covered
------------------
* Objective 1 — baseline 80/20 train/test split architecture
  (``run_baseline_split``).
* Objective 3 — 5-fold cross-validation framework with per-fold metric
  aggregation (``run_five_fold_cv``).
* Objective 4 — subsampling / training-size efficiency sweep, 1%–80% of the
  training pool against a fixed 20% holdout (``run_subsampling_experiment``).

Every experiment reuses the existing supervised pipelines
(``default_classic_ml_models_kfolds`` / ``reference_mapping``) so training
configs stay in ONE place, and enforces non-spatial masking everywhere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
_METHODS_UTILS = _REPO / "src" / "methods" / "utils"
_PSEUDO_DIR = _REPO / "src" / "pseudo_labeling"
_EVAL_DIR = _REPO / "src" / "evaluation"
_UTILS_DIR = _REPO / "src" / "utils"
for _p in (_HERE, _METHODS_UTILS, _PSEUDO_DIR, _EVAL_DIR, _UTILS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CLASSIC_SUPERVISED: Tuple[str, ...] = (
    "logistic_regression",
    "random_forest",
    "svm",
    "xgboost",
)
REFERENCE_SUPERVISED: Tuple[str, ...] = ("singler", "scarches", "ribca_adapted")
SUPERVISED_METHODS = CLASSIC_SUPERVISED + REFERENCE_SUPERVISED


def _materialize_split_as_kfold(
    split_data: dict,
    out_dir: Path,
    random_state: int,
    *,
    markers: Optional[Sequence[str]] = None,
) -> Path:
    """Write an 80/20 (or subsampled) split into the ``kfolds_*`` layout.

    ClassicMLDefault expects ``fold_1_train.csv`` / ``fold_1_validation.csv`` /
    ``fold_1_test.csv`` with an ``encoded_phenotype`` label column. The
    validation partition is a stratified 10% slice of the training pool.

    Only marker columns survive (plus the label) so that no non-protein
    metadata (Cell_ID, batch, coordinates) ever leaks into the features.
    """
    import pandas as pd
    from kfold_strategies import stratified_80_20_split  # noqa: WPS433

    out_dir.mkdir(parents=True, exist_ok=True)

    X_train = pd.DataFrame(split_data["X_train"]).reset_index(drop=True)
    X_test = pd.DataFrame(split_data["X_test"]).reset_index(drop=True)
    Y_train = split_data["Y_train"]
    Y_test = split_data["Y_test"]

    keep_cols = None
    if markers is not None:
        keep_cols = [c for c in markers if c in X_train.columns]

    # Stratified validation slice of the training pool.
    val_idx, train_idx_filtered = stratified_80_20_split(
        Y_train, random_state=random_state, test_size=0.10
    )

    def _write(name: str, X: pd.DataFrame, Y: Sequence) -> None:
        df = X.reset_index(drop=True).copy()
        if keep_cols is not None:
            df = df[keep_cols].copy()
        df["encoded_phenotype"] = list(Y)
        df.to_csv(out_dir / f"fold_1_{name}.csv", index=False)

    _write("train", X_train.iloc[train_idx_filtered], Y_train[train_idx_filtered])
    _write("validation", X_train.iloc[val_idx], Y_train[val_idx])
    _write("test", X_test, Y_test)
    return out_dir


def _run_supervised_method(
    method_id: str,
    kdir: Path,
    labels: Path,
    out: Path,
    *,
    ml_n_jobs: int = -1,
    dumb_columns: Optional[List[str]] = None,
    markers: Optional[Sequence[str]] = None,
) -> Path:
    """Train + predict one supervised method on an existing k-fold-shaped dir."""
    from default_classic_ml_models_kfolds import ClassicMLDefault  # noqa: WPS433

    out.mkdir(parents=True, exist_ok=True)
    if method_id in CLASSIC_SUPERVISED:
        clf = ClassicMLDefault(random_state=42, model=method_id, n_jobs=ml_n_jobs)
        clf.train_tune_evaluate(
            str(kdir), str(labels), verbose=0, scaling=True, dumb_columns=dumb_columns
        )
        clf.save_results(str(out), str(labels), str(kdir), save_model=False)
        try:
            clf.save_feature_importances(str(out), str(labels), str(kdir))
        except Exception:
            pass
        return out

    if method_id in REFERENCE_SUPERVISED:
        from reference_mapping import (  # noqa: WPS433
            ribca_predict,
            run_kfold_label_transfer,
            scarches_predict,
            singler_predict,
        )

        predict_fn = {
            "singler": singler_predict,
            "scarches": scarches_predict,
            "ribca_adapted": ribca_predict,
        }[method_id]
        markers = markers or _markers_from_dir(kdir)
        run_kfold_label_transfer(kdir, labels, out, markers, predict_fn)
        return out

    raise ValueError(f"Unsupported supervised method for experiments: {method_id}")


def _csv_header(csv_path: Path) -> List[str]:
    import pandas as pd  # noqa: WPS433

    return pd.read_csv(csv_path, nrows=1).columns.tolist()


def _markers_from_dir(kdir: Path) -> List[str]:
    return [c for c in _csv_header(kdir / "fold_1_train.csv") if c != "encoded_phenotype"]


def _prediction_files(result_dir: Path) -> List[Path]:
    """Predictions CSVs, both naming conventions.

    Classic ML + MAPS write ``predictions_fold_*.csv`` while the reference
    mapping pipeline (``run_kfold_label_transfer``) writes ``predictions_*.csv``.
    """
    return sorted(result_dir.glob("predictions_*.csv"))


def _evaluate_prediction_file(
    pred_file: Path,
    level: str = "level3",
) -> dict:
    """Full supervised metric panel for one predictions CSV."""
    import pandas as pd  # noqa: WPS433
    from metrics import compute_supervised_metrics  # noqa: WPS433

    df = pd.read_csv(pred_file)
    if "predicted_phenotype" not in df or "true_phenotype" not in df:
        return {}
    m = compute_supervised_metrics(
        df["true_phenotype"], df["predicted_phenotype"], level=level
    )
    d = m.to_dict()
    d["n_cells"] = int(len(df))
    d["prediction_file"] = pred_file.name
    return d


def _collect_kfold_metrics(result_dir: Path, level: str) -> dict:
    """Mean ± std across all per-fold predictions CSVs in *result_dir*."""
    files = _prediction_files(result_dir)
    if not files:
        return {}
    per_fold = [_evaluate_prediction_file(f, level=level) for f in files]
    per_fold = [d for d in per_fold if d]
    if not per_fold:
        return {}
    keys = [k for k in per_fold[0].keys() if k not in ("prediction_file", "n_cells")]
    agg = {}
    for key in keys:
        values = [d.get(key) for d in per_fold]
        values = [v for v in values if isinstance(v, (int, float))]
        if values:
            mean = float(np.mean(values))
            std = float(np.std(values))
            agg[f"{key}_mean"] = mean
            agg[f"{key}_std"] = std
    agg["n_folds"] = len(per_fold)
    agg["n_cells_total"] = int(sum(int(d.get("n_cells", 0)) for d in per_fold))
    return agg


def _prepare_dataset(ctx) -> object:
    """Load quant CSV, enforce non-spatial masking, return a DataSetHandler.

    The feature matrix ``X`` is reduced to the ground-truth marker columns only
    (protein expression), so no metadata or coordinates participate in the
    supervised experiments.
    """
    from data_handler import DataSetHandler  # noqa: WPS433

    if not ctx.quant_path.exists():
        raise FileNotFoundError(f"Quantification CSV not found: {ctx.quant_path}")
    dsh = DataSetHandler(str(ctx.quant_path), random_state=42)
    dsh.preprocess(
        dropna=False,
        phenotype_column=ctx.phenotype_column,
        drop_non_numerical=True,
        enforce_non_spatial=True,
    )
    keep = [c for c in ctx.markers if c in dsh.X.columns]
    if not keep:
        raise ValueError(
            f"No marker columns matched in {ctx.quant_path.name}. "
            f"Configured markers: {ctx.markers}"
        )
    dsh.X = dsh.X[keep]
    return dsh


def run_baseline_split(
    ctx,
    methods: Sequence[str],
    out_root: Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    ml_n_jobs: int = -1,
) -> dict:
    """Objective 1 — baseline 80/20 train/test split for every method."""
    out = out_root / "baseline_80_20"
    out.mkdir(parents=True, exist_ok=True)
    labels_path = out / "labels.csv"

    dsh = _prepare_dataset(ctx)
    dsh.create_stratified_split(test_size=test_size, random_state=random_state)
    dsh.labels.to_csv(labels_path, index=False)

    split_kdir = out / "kfold_materialized"
    _materialize_split_as_kfold(
        dsh.split_data, split_kdir, random_state, markers=ctx.markers
    )

    summary = {"split": "80_20", "test_size": test_size, "random_state": random_state, "methods": {}}
    for method in methods:
        if method not in SUPERVISED_METHODS:
            continue
        result_dir = out / method
        try:
            _run_supervised_method(
                method, split_kdir, labels_path, result_dir, ml_n_jobs=ml_n_jobs
            )
        except Exception as exc:
            summary["methods"][method] = {"error": str(exc)}
            continue
        pred_files = _prediction_files(result_dir)
        if pred_files:
            m = _evaluate_prediction_file(pred_files[0], level=ctx.granularity)
            summary["methods"][method] = m

    _write_summary(out, summary, "baseline_80_20")
    return summary


def run_five_fold_cv(
    ctx,
    methods: Sequence[str],
    out_root: Path,
    *,
    kfold_method: str = "StratifiedKFold",
    ml_n_jobs: int = -1,
) -> dict:
    """Objective 3 — 5-fold CV: train/test on the real k-fold dirs, aggregate mean±std."""
    kdir = ctx.kfold_dir(kfold_method)
    labels = ctx.labels_path(kfold_method)
    if not kdir.is_dir():
        raise FileNotFoundError(
            f"K-folds missing: {kdir} — run with --ensure_kfolds first."
        )

    out = out_root / "five_fold_cv"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"kfold_method": kfold_method, "n_splits": ctx.n_splits, "methods": {}}

    # Strict non-spatial masking on the real k-fold CSVs: drop metadata columns
    # (incl. x/y that run_kfold_creator deliberately keeps) from the features.
    from ground_truth import DEFAULT_EVAL_DROP_COLUMNS, SPATIAL_COLUMNS  # noqa: WPS433

    fold_header = _csv_header(kdir / "fold_1_train.csv")
    dumb = [c for c in DEFAULT_EVAL_DROP_COLUMNS if c in fold_header]
    dumb += [c for c in SPATIAL_COLUMNS if c in fold_header and c not in dumb]
    # Defensive: only configured protein markers (plus batch col, excluded above)
    # may act as features; any residual obs/morphology column is removed.
    configured = [c for c in ctx.markers if c in fold_header]
    non_marker = [
        c for c in fold_header
        if c != "encoded_phenotype"
        and c not in configured
        and c not in dumb
        and c != ctx.batch_column
    ]
    if non_marker:
        print(
            f"Non-spatial masking: dropping {len(non_marker)} non-marker column(s) "
            f"from k-fold features: {non_marker}"
        )
        dumb += non_marker
    markers = [c for c in configured if c not in dumb] or [
        c for c in fold_header if c not in dumb and c != "encoded_phenotype"
    ]

    for method in methods:
        if method not in SUPERVISED_METHODS:
            continue
        result_dir = out / kfold_method / method
        try:
            _run_supervised_method(
                method, kdir, labels, result_dir, ml_n_jobs=ml_n_jobs,
                dumb_columns=dumb, markers=markers,
            )
        except Exception as exc:
            summary["methods"][method] = {"error": str(exc)}
            continue
        try:
            agg = _collect_kfold_metrics(result_dir, level=ctx.granularity)
        except Exception as exc:
            summary["methods"][method] = {"error": f"metric aggregation failed: {exc}"}
            continue
        summary["methods"][method] = agg or {"error": "no predictions_fold_*.csv produced"}

    _write_summary(out, summary, "five_fold_cv")
    return summary


def run_subsampling_experiment(
    ctx,
    methods: Sequence[str],
    out_root: Path,
    *,
    fractions: Sequence[float] = (0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80),
    holdout: float = 0.20,
    random_state: int = 42,
    ml_n_jobs: int = -1,
) -> dict:
    """Objective 4 — training-size efficiency sweep on a fixed 20% holdout.

    For every fraction of the 80% training pool, methods are retrained and
    evaluated on the SAME test partition, isolating the effect of training
    data amount (1%–80%).
    """
    from kfold_strategies import SUBSAMPLING_FRACTIONS, subsample_stratified_indices  # noqa: WPS433

    if fractions is None:
        fractions = SUBSAMPLING_FRACTIONS
    out = out_root / "subsampling_experiment"
    out.mkdir(parents=True, exist_ok=True)
    labels_path = out / "labels.csv"

    dsh = _prepare_dataset(ctx)
    dsh.create_stratified_split(test_size=holdout, random_state=random_state)
    dsh.labels.to_csv(labels_path, index=False)

    summary = {"holdout": holdout, "fractions": list(fractions), "methods": {}}

    for fraction in fractions:
        train_idx = subsample_stratified_indices(
            dsh.Y, dsh.split_indices[0], fraction, random_state=random_state
        )
        split_data = {
            "X_train": dsh.X.iloc[train_idx].reset_index(drop=True),
            "X_test": dsh.split_data["X_test"],
            "Y_train": dsh.Y[train_idx],
            "Y_test": dsh.split_data["Y_test"],
        }
        frac_out = out / f"fraction_{fraction:g}"
        split_kdir = frac_out / "kfold_materialized"
        _materialize_split_as_kfold(
            split_data, split_kdir, random_state, markers=ctx.markers
        )

        summary["methods"][f"{fraction:g}"] = {}
        for method in methods:
            if method not in SUPERVISED_METHODS:
                continue
            result_dir = frac_out / method
            try:
                _run_supervised_method(
                    method, split_kdir, labels_path, result_dir, ml_n_jobs=ml_n_jobs
                )
            except Exception as exc:
                summary["methods"][f"{fraction:g}"][method] = {"error": str(exc)}
                continue
            pred_files = _prediction_files(result_dir)
            if pred_files:
                m = _evaluate_prediction_file(pred_files[0], level=ctx.granularity)
                m["train_fraction"] = float(fraction)
                m["train_cells"] = int(len(split_data["Y_train"]))
                summary["methods"][f"{fraction:g}"][method] = m

    _write_summary(out, summary, "subsampling_experiment")
    return summary


def _write_summary(out_dir: Path, summary: dict, name: str) -> None:
    import pandas as pd  # noqa: WPS433

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    rows = []
    for method, result in summary.get("methods", {}).items():
        if not isinstance(result, dict):
            continue
        row = {"method": method, "fraction": None, **result}
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / f"{name}_summary.csv", index=False)


# ── CLI ───────────────────────────────────────────────────────────────────

def build_parser():
    import argparse  # noqa: WPS433

    p = argparse.ArgumentParser(description="spCellEval experiment runner.")
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--experiment",
        choices=["baseline", "five_fold_cv", "subsample"],
        required=True,
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=list(SUPERVISED_METHODS),
        help="Supervised method ids. Default: %(default)s",
    )
    p.add_argument("--benchmark_config", type=Path, default=_REPO / "configs" / "benchmark.yaml")
    p.add_argument("--root_dir", type=Path, default=_REPO)
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument(
        "--fractions",
        type=str,
        default="0.01,0.05,0.10,0.20,0.40,0.60,0.80",
        help="Comma-separated train fractions for the subsampling experiment.",
    )
    p.add_argument("--holdout", type=float, default=0.20)
    return p


def main(argv=None) -> int:
    import yaml  # noqa: WPS433
    from dataset_context import DatasetContext  # noqa: WPS433

    args = build_parser().parse_args(argv)

    bench_cfg = yaml.safe_load(args.benchmark_config.read_text(encoding="utf-8"))
    ds_entry = bench_cfg["datasets"][args.dataset]
    ds_config_path = args.root_dir / ds_entry.get("config", f"configs/{args.dataset.lower()}.yaml")
    ctx = DatasetContext.from_config(ds_config_path, root=args.root_dir)
    out_root = args.out_dir or ctx.root / "experiments" / ctx.dataset_name
    out_root.mkdir(parents=True, exist_ok=True)

    methods = [m for m in args.methods if m in SUPERVISED_METHODS]
    if args.experiment == "baseline":
        run_baseline_split(ctx, methods, out_root)
    elif args.experiment == "five_fold_cv":
        run_five_fold_cv(ctx, methods, out_root)
    elif args.experiment == "subsample":
        fractions = [float(x) for x in args.fractions.split(",")]
        run_subsampling_experiment(ctx, methods, out_root, fractions=fractions, holdout=args.holdout)
    print(f"Experiment '{args.experiment}' written to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python
"""Validate CRC_TMA on-disk results against raw prediction CSVs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "evaluation"))

from dataset_paths import resolve_dataset_quant
from evaluator import aggregate_results, evaluate_dataset
from prediction_io import discover_prediction_files

DATASET = "CRC_TMA"
TOL = 1e-4


def _accuracy_from_csv(path: Path) -> float:
    df = pd.read_csv(path, usecols=["true_phenotype", "predicted_phenotype"])
    return float((df["true_phenotype"].astype(str) == df["predicted_phenotype"].astype(str)).mean())


def main() -> int:
    results = REPO / "results" / DATASET
    summary = results / "summary"
    final_path = summary / "final_results.csv"
    per_fold_path = summary / "per_fold_metrics.csv"

    if not final_path.is_file() or not per_fold_path.is_file():
        print("FAIL: summary CSVs missing — run evaluation first.")
        return 1

    final = pd.read_csv(final_path)
    per_fold = pd.read_csv(per_fold_path)
    quant_path, marker_cols = resolve_dataset_quant(DATASET, REPO)

    issues: list[str] = []
    ok_count = 0

    print("=" * 60)
    print("CRC_TMA RESULT VALIDATION")
    print("=" * 60)

    # 1. Structure
    spatial_n = int(per_fold["file"].str.contains("_spatial").sum())
    raw_files = discover_prediction_files(results)
    print(f"per_fold rows: {len(per_fold)} (expect 84 with leiden granularity fix)")
    print(f"spatial rows in per_fold: {spatial_n} (expect 0)")
    print(f"raw prediction files: {len(raw_files)} (expect 30)")
    print(f"plots: {len(list((summary / 'plots').rglob('*.png')))}")

    if spatial_n:
        issues.append("per_fold_metrics still includes _spatial files")
    if len(per_fold) != 84:
        issues.append(f"per_fold row count {len(per_fold)} != 84")

    # 2. ML methods vs benchmark JSON
    print("\n--- ML methods (level3) ---")
    ml_methods = [
        ("random_forest", "average_rfc_results.json", "predictions_fold_*.csv"),
        ("logistic_regression", "average_logreg_results.json", "predictions_fold_*.csv"),
        ("xgboost", "average_xgboost_results.json", "predictions_fold_*.csv"),
    ]
    for method, json_name, pattern in ml_methods:
        jpath = results / method / "level3" / json_name
        json_acc = json.loads(jpath.read_text())["average_accuracy"]
        paths = [p for p in sorted((results / method / "level3").glob(pattern)) if "_spatial" not in p.name]
        csv_mean = float(np.mean([_accuracy_from_csv(p) for p in paths]))
        pf_sub = per_fold[(per_fold["method"] == method) & (per_fold["level"] == "level3")]
        pf_mean = float(pf_sub["accuracy"].mean())
        final_acc = float(final[(final["method"] == method) & (final["level"] == "level3")]["accuracy_mean"].iloc[0])
        n_folds = int(final[(final["method"] == method) & (final["level"] == "level3")]["n_folds"].iloc[0])
        match = max(abs(json_acc - csv_mean), abs(csv_mean - final_acc), abs(pf_mean - final_acc)) < TOL
        status = "OK" if match and n_folds == 5 else "FAIL"
        if status == "OK":
            ok_count += 1
        else:
            issues.append(f"{method}: json/csv/final mismatch or n_folds={n_folds}")
        print(f"  {method}: json={json_acc:.6f} csv={csv_mean:.6f} final={final_acc:.6f} folds={n_folds} [{status}]")

    # 3. Ground truth
    print("\n--- Ground truth consistency ---")
    quant = pd.read_csv(quant_path, usecols=["Cell_ID", "cell_type"])
    for rel in [
        "random_forest/level3/predictions_fold_1.csv",
        "singler/level3/predictions_1.csv",
        "signature/level3/predictions_1.csv",
    ]:
        pred = pd.read_csv(results / rel, usecols=["Cell_ID", "true_phenotype"])
        merged = pred.merge(quant, on="Cell_ID", how="inner")
        rate = float((merged["true_phenotype"].astype(str) == merged["cell_type"].astype(str)).mean())
        status = "OK" if rate == 1.0 and len(merged) == len(pred) else "FAIL"
        if status != "OK":
            issues.append(f"ground truth mismatch in {rel}")
        print(f"  {rel}: {len(merged)}/{len(pred)} cells, match={rate:.4f} [{status}]")

    # 4. Recompute with current code (sample: ML only for speed)
    print("\n--- Recompute check (ML methods, current code) ---")
    per_fold_new = evaluate_dataset(
        DATASET, REPO / "results",
        methods=["random_forest", "xgboost", "logistic_regression"],
        quant_path=quant_path,
        marker_cols=marker_cols,
    )
    summary_new = aggregate_results(per_fold_new)
    for method in ["random_forest", "xgboost", "logistic_regression"]:
        old = float(final[(final["method"] == method) & (final["level"] == "level3")]["accuracy_mean"].iloc[0])
        new = float(summary_new[(summary_new["method"] == method) & (summary_new["level"] == "level3")]["accuracy_mean"].iloc[0])
        status = "OK" if abs(old - new) < TOL else "FAIL"
        if status != "OK":
            issues.append(f"disk vs current code mismatch for {method}")
        print(f"  {method}: disk={old:.6f} code={new:.6f} [{status}]")

    # 5. Leiden granularity (current code should give n_folds=1 at level3)
    print("\n--- Leiden (level3, current code) ---")
    per_fold_leiden = evaluate_dataset(
        DATASET, REPO / "results", methods=["leiden"],
        quant_path=quant_path, marker_cols=marker_cols,
    )
    leiden_l3 = per_fold_leiden[per_fold_leiden["level"] == "level3"]
    leiden_summary = aggregate_results(per_fold_leiden)
    leiden_n = int(leiden_summary[(leiden_summary["method"] == "leiden") & (leiden_summary["level"] == "level3")]["n_folds"].iloc[0])
    disk_leiden_n = int(final[(final["method"] == "leiden") & (final["level"] == "level3")]["n_folds"].iloc[0])
    print(f"  disk n_folds={disk_leiden_n}, current code n_folds={leiden_n}, level3 rows={len(leiden_l3)}")
    if disk_leiden_n != leiden_n:
        issues.append(f"leiden n_folds stale on disk ({disk_leiden_n}) vs current code ({leiden_n}) — re-run evaluation")

    print("\n" + "=" * 60)
    if issues:
        print("ISSUES:")
        for i in issues:
            print(f"  - {i}")
        print("=" * 60)
        return 1

    print("ALL CHECKS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

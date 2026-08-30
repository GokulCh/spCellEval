"""Shared helpers for unsupervised clustering benchmark outputs."""

from __future__ import annotations

import os
import time
from typing import Iterable, Sequence

import pandas as pd

from greedy_f1_utils import greedy_f1_score

GRANULARITY = {
    "level3": "cell_type",
    "level1": "level_1_cell_type",
    "level2": "level_2_cell_type",
}


def write_greedy_predictions(
    df: pd.DataFrame,
    cluster_col: str,
    output_path: str,
    iteration: int,
    *,
    log: str = "off",
    logger=None,
    tie_strategy: str = "random",
    timing_path: str | None = None,
    elapsed_sec: float | None = None,
) -> None:
    """Map clusters to phenotypes via greedy F1 and write per-level prediction CSVs."""
    if timing_path and elapsed_sec is not None:
        os.makedirs(os.path.dirname(timing_path), exist_ok=True)
        mode = "w" if iteration == 1 else "a"
        with open(timing_path, mode, encoding="utf-8") as fh:
            fh.write(f"Fold {iteration} train_time, {elapsed_sec:.2f}\n")

    for level, true_col in GRANULARITY.items():
        if true_col not in df.columns:
            continue
        results = greedy_f1_score(df, true_col, cluster_col, tie_strategy=tie_strategy)
        output = df.copy()
        output["predicted_phenotype"] = results["mapped_predictions"]
        output = output.rename(columns={true_col: "true_phenotype"})
        path = f"{output_path}/{level}/predictions_{iteration}.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if log == "long" and logger is not None:
            logger.info("Saving predictions to %s", path)
        output.to_csv(path, index=False)


def timed_cluster_run(
    cluster_fn,
    *,
    log: str = "off",
    logger=None,
    message: str = "clustering",
) -> float:
    """Run *cluster_fn* and return elapsed seconds."""
    start = time.time()
    cluster_fn()
    elapsed = time.time() - start
    if log == "long" and logger is not None:
        logger.info("%s took %.2f minutes", message, elapsed / 60)
    elif log == "short" and logger is not None:
        logger.info("%.2f min", elapsed / 60)
    return elapsed

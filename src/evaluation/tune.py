"""Automatic parameter search: for each dataset x method, try candidate parameter sets (``models/params.py``: SPACE),
score each with the benchmark's own hold-out run, and write the best set into the params file under the dataset's name.

Trials are scored on the *validation* split of private folds (the test split is never looked at, and the folds use
``tune_seed`` instead of the run's seed), so a following ``benchmark`` gives an honest number. Methods that cluster all
cells without training (leiden, flowsom, ...) are scored against the labels on every cell, as in the benchmark itself.

Speed-ups: the table is loaded once and reused (``cache_data``); ``jobs`` runs trials in parallel; with ``stage1_cells`` every
candidate is first scored on a subsample of that many cells and only the best ``finalists`` (plus the current setting) are
re-scored on the full data.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..models import params as mparams
from .runner import BenchConfig, run_dataset

log = logging.getLogger("spcelleval")


def dataset_name(path) -> str:
    return Path(path).stem.replace("_quantification", "")     # the name load_dataset gives it


def trial_score(res: pd.DataFrame, metric: str) -> float:
    """Mean ``metric`` over a trial's ok rows; for methods that write several variants (leiden resolutions) the best variant."""
    ok = res[res.status == "ok"] if "status" in res and metric in res else res.iloc[:0]
    return float(ok.groupby("method")[metric].mean().max()) if len(ok) else float("nan")


def tune(cfg: BenchConfig, methods: list[str], trials: int = 12, metric: str = "f1_macro", tune_seed: int = 1000,
         jobs: int = 1, stage1_cells: int = 0, finalists: int = 3) -> pd.DataFrame:
    """Search each of ``methods`` on every dataset in ``cfg.data``; returns (and saves) the log of all trials."""
    out, rows, summary = Path(cfg.out), [], []
    for path in cfg.data:
        name = dataset_name(path)
        for base in dict.fromkeys(m.removesuffix("+vote") for m in methods):
            cands = mparams.trials(base, trials, cfg.seed, mparams.resolve(name, base, cfg.overrides.get(base), cfg.params_file))
            if not cands:
                log.info("[%s] %s: no tunable parameters, skipped", name, base)
                continue

            def run(i, cells, stage):
                tc = replace(cfg, data=[path], methods=[base], out=str(out / "_tune" / name / base / f"{stage}{i}"), split="holdout",
                             seed=cfg.seed + tune_seed, splits_dir="out", max_cells=cells,
                             overrides={**cfg.overrides, base: {**cfg.overrides.get(base, {}), **cands[i]}},
                             report=False, resume=False, save_predictions=False, jobs=1, tune_on_validation=True, cache_data=True)
                try:
                    s = trial_score(run_dataset(path, tc)["results"], metric)
                except Exception as e:                               # a bad trial must not stop the search
                    log.warning("[%s] %s trial %d failed: %s", name, base, i, e)
                    s = float("nan")
                rows.append(dict(dataset=name, method=base, trial=i, stage=stage, cells=cells, **{metric: s}, params=json.dumps(cands[i])))
                log.info("[%s] %-20s %s trial %d/%d %s=%.4f %s", name, base, stage, i + 1, len(cands), metric, s, cands[i])
                return i, s

            two_stage = bool(stage1_cells) and len(cands) > finalists and (cfg.max_cells == 0 or cfg.max_cells > stage1_cells)
            with ThreadPoolExecutor(max(jobs, 1)) as ex:
                todo = list(range(len(cands)))
                if two_stage:
                    s1 = dict(ex.map(lambda i: run(i, stage1_cells, "s"), todo))
                    ranked = sorted((i for i in todo if i and not np.isnan(s1[i])), key=lambda i: -s1[i])
                    todo = [0, *ranked[:finalists - 1]]              # the current setting always competes in the final round
                scores = dict(ex.map(lambda i: run(i, cfg.max_cells, "f" if two_stage else "t"), todo))
            best = None
            for i in sorted(scores):                                  # strict: a tie keeps the earlier (current) setting
                if not np.isnan(scores[i]) and (best is None or scores[i] > scores[best]):
                    best = i
            if best is None:
                log.warning("[%s] %s: every trial failed; the params file is unchanged", name, base)
            else:
                mparams.save(name, base, cands[best], cfg.params_file)
                log.info("[%s] %-20s best %s=%.4f -> %s", name, base, metric, scores[best], cands[best])
                summary.append(dict(dataset=name, method=base, current=scores.get(0, np.nan), tuned=scores[best],
                                    gain=scores[best] - scores.get(0, np.nan), params=json.dumps(cands[best])))
            out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).sort_values(["dataset", "method", "stage", "trial"]).to_csv(out / "tuning_log.csv", index=False)
            if summary:
                write_summary(pd.DataFrame(summary), out, metric)
    return pd.DataFrame(rows)


def write_summary(s: pd.DataFrame, out: Path, metric: str) -> None:
    """tuning_summary.csv (current vs tuned score per dataset x method) and a bar figure of the two."""
    s.to_csv(out / "tuning_summary.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    lab = s.dataset + " / " + s.method
    fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(s) + 3), 4))
    x = np.arange(len(s))
    ax.bar(x - 0.2, s.current, 0.4, label="current setting")
    ax.bar(x + 0.2, s.tuned, 0.4, label="tuned")
    ax.set_xticks(x, lab, rotation=60, ha="right")
    ax.set_ylabel(f"{metric} (validation split)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "tuning_gain.png", dpi=150)
    plt.close(fig)

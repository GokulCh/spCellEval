"""Automatic parameter search: for each dataset x method, try candidate parameter sets (``models/params.py``: SPACE),
score each with the benchmark's own hold-out run, and write the best set into the params file under the dataset's name.

Every trial runs the normal ``run_dataset`` on a hold-out split. The split uses ``tune_seed`` instead of the run's seed, so
it is not the split the final benchmark reports on - but cells overlap, so the tuned score is still optimistic. For an
unbiased number, re-run ``benchmark`` after tuning and report that.
"""
from __future__ import annotations

import json
import logging
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


def tune(cfg: BenchConfig, methods: list[str], trials: int = 12, metric: str = "f1_macro", tune_seed: int = 1000) -> pd.DataFrame:
    """Search each of ``methods`` on every dataset in ``cfg.data``; returns (and saves) the log of all trials."""
    out, rows = Path(cfg.out), []
    for path in cfg.data:
        name = dataset_name(path)
        for base in dict.fromkeys(m.removesuffix("+vote") for m in methods):
            cands = mparams.trials(base, trials, cfg.seed, mparams.resolve(name, base, cfg.overrides.get(base), cfg.params_file))
            if not cands:
                log.info("[%s] %s: no tunable parameters, skipped", name, base)
                continue
            best, scores = None, []
            for i, c in enumerate(cands):
                tc = replace(cfg, data=[path], methods=[base], out=str(out / "_tune" / name / base / f"t{i}"), split="holdout",
                             seed=cfg.seed + tune_seed, splits_dir="out", overrides={**cfg.overrides, base: {**cfg.overrides.get(base, {}), **c}},
                             report=False, resume=False, save_predictions=False, jobs=1)
                try:
                    s = trial_score(run_dataset(path, tc)["results"], metric)
                except Exception as e:                               # a bad trial must not stop the search
                    log.warning("[%s] %s trial %d failed: %s", name, base, i, e)
                    s = float("nan")
                scores.append(s)
                rows.append(dict(dataset=name, method=base, trial=i, **{metric: s}, params=json.dumps(c)))
                log.info("[%s] %-20s trial %d/%d %s=%.4f %s", name, base, i + 1, len(cands), metric, s, c)
                if not np.isnan(s) and (best is None or s > scores[best]):      # strict: a tie keeps the earlier (current) setting
                    best = i
            if best is None:
                log.warning("[%s] %s: every trial failed; the params file is unchanged", name, base)
            else:
                mparams.save(name, base, cands[best], cfg.params_file)
                log.info("[%s] %-20s best %s=%.4f -> %s", name, base, metric, scores[best], cands[best])
            out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out / "tuning_log.csv", index=False)
    return pd.DataFrame(rows)

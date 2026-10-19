"""Automatic, progressive parameter search: for each dataset x method, find the best parameter set and write it into the
params file under the dataset's name (``models/params.py``: SPACE holds the candidates).

Phases (each only runs when it can pay off):
  1. explore  - ``trials`` parameter sets (the current setting + random draws from the grid). With ``budgets`` /
                ``stage1_cells`` this is successive halving: every candidate is scored on a small subsample, the best
                1/``eta`` (at least ``finalists``, and always the current setting) advance to the next, larger subsample,
                and the survivors are scored on all the data.
  2. refine   - coordinate descent from the winner: each parameter in turn is moved through its other candidate values on
                the full data; a move is kept only if it beats the incumbent by more than ``min_gain`` (so noise is not
                chased). ``refine`` passes, at most ``refine_evals`` evaluations.
Scores come from the *validation* split of private folds (the test split is never looked at, and the folds use
``tune_seed``), so a following ``benchmark`` gives an honest number. Methods that cluster all cells without training
(leiden, flowsom, ...) are scored against the labels on every cell, as in the benchmark itself.

Outputs: ``tuning_log.csv`` (every evaluation: score, runtime, extra metrics, parameters), ``tuning_summary.csv`` +
``tuning_gain.png`` (current vs tuned), ``tuning_param_effects.csv`` and ``tuning_report.md`` (what changed and which
parameters mattered).
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
EXTRA_METRICS = ("f1_weighted", "mcc", "accuracy")
FULL = ("t", "f", "r")                                     # log stages scored on the full data (s = subsample)


def dataset_name(path) -> str:
    return Path(path).stem.replace("_quantification", "")     # the name load_dataset gives it


def trial_stats(res: pd.DataFrame, metric: str) -> dict:
    """``metric`` (mean over a trial's ok rows; for methods that write several variants, e.g. leiden resolutions, the best
    variant) plus that variant's total runtime and a few other metrics. NaN score if nothing ran."""
    ok = res[res.status == "ok"] if "status" in res and metric in res else res.iloc[:0]
    means = ok.groupby("method")[metric].mean().dropna() if len(ok) else pd.Series(dtype=float)
    if means.empty:
        return {metric: float("nan")}
    b = ok[ok.method == means.idxmax()]
    return {metric: float(means.max()), "runtime_s": float(b.runtime_s.sum()) if "runtime_s" in b else float("nan"),
            **{k: float(b[k].mean()) for k in EXTRA_METRICS if k in b and k != metric}}


def trial_score(res: pd.DataFrame, metric: str) -> float:
    return trial_stats(res, metric)[metric]


def _key(p: dict) -> str:
    return json.dumps(p, sort_keys=True, default=str)


def pick(scores: dict[int, float], min_gain: float = 0.0) -> int | None:
    """Index of the best score; the current setting (index 0) wins unless beaten by more than ``min_gain``. None if all NaN."""
    ok = {i: s for i, s in sorted(scores.items()) if not np.isnan(s)}
    if not ok:
        return None
    best = max(ok, key=ok.get)                                 # first of equal scores = earliest (current) setting
    return 0 if 0 in ok and ok[best] <= ok[0] + min_gain else best


def tune(cfg: BenchConfig, methods: list[str], trials: int = 12, metric: str = "f1_macro", tune_seed: int = 1000,
         jobs: int = 1, stage1_cells: int = 0, finalists: int = 3, budgets: tuple = (), eta: int = 3, refine: int = 1,
         min_gain: float = 0.002, refine_evals: int = 0) -> pd.DataFrame:
    """Search each of ``methods`` on every dataset in ``cfg.data``; returns (and saves) the log of all evaluations."""
    out, rows, summary = Path(cfg.out), [], []
    rungs = sorted({int(b) for b in (*budgets, stage1_cells) if b})
    rungs = [r for r in rungs if cfg.max_cells == 0 or r < cfg.max_cells]
    for path in cfg.data:
        name = dataset_name(path)
        for base in dict.fromkeys(m.removesuffix("+vote") for m in methods):
            start = mparams.resolve(name, base, cfg.overrides.get(base), cfg.params_file)
            cands = mparams.trials(base, trials, cfg.seed, start)
            if not cands:
                log.info("[%s] %s: no tunable parameters, skipped", name, base)
                continue

            def run(i, params, cells, stage):
                tc = replace(cfg, data=[path], methods=[base], out=str(out / "_tune" / name / base / f"{stage}{i}"), split="holdout",
                             seed=cfg.seed + tune_seed, max_cells=cells,
                             splits_dir=str(out / "_tune" / name / "_folds" / f"c{cells}"),      # one private folds store per subsample size: every trial reuses it
                             overrides={**cfg.overrides, base: {**cfg.overrides.get(base, {}), **params}},
                             report=False, resume=False, save_predictions=False, jobs=1, tune_on_validation=True, cache_data=True)
                try:
                    st = trial_stats(run_dataset(path, tc)["results"], metric)
                except Exception as e:                               # a bad trial must not stop the search
                    log.warning("[%s] %s trial %d failed: %s", name, base, i, e)
                    st = {metric: float("nan")}
                rows.append(dict(dataset=name, method=base, trial=i, stage=stage, cells=cells, **st, params=json.dumps(params, default=str)))
                log.info("[%s] %-20s %s trial %d %s=%.4f %s", name, base, stage, i, metric, st[metric], params)
                return i, st[metric]

            with ThreadPoolExecutor(max(jobs, 1)) as ex:
                # ---- 1. explore (successive halving over growing subsamples)
                alive, halved = list(range(len(cands))), False
                for cells in rungs:
                    if len(alive) <= finalists:
                        break
                    s = dict(ex.map(lambda i: run(i, cands[i], cells, "s"), alive))
                    halved = True
                    keep = max(finalists, -(-len(alive) // max(eta, 2)))
                    ranked = sorted((i for i in alive if i and not np.isnan(s[i])), key=lambda i: -s[i])
                    alive = [0, *ranked[:keep - 1]]                  # the current setting always advances
                scores = dict(ex.map(lambda i: run(i, cands[i], cfg.max_cells, "f" if halved else "t"), alive))
                best = pick(scores, min_gain)
                # ---- 2. refine (one parameter at a time, full data)
                if best is not None and refine > 0:
                    cur, cur_s, seen = cands[best], scores[best], {_key(cands[i]): s for i, s in scores.items()}
                    budget, used = refine_evals or trials, 0
                    for _ in range(refine):
                        moved = False
                        for k, vals in mparams.alternatives(base, cur):
                            todo = [{**cur, k: v} for v in vals if _key({**cur, k: v}) not in seen][:max(budget - used, 0)]
                            if not todo:
                                continue
                            ids = list(range(len(cands), len(cands) + len(todo)))
                            cands += todo
                            got = dict(ex.map(lambda a: run(a[0], a[1], cfg.max_cells, "r"), zip(ids, todo)))
                            used += len(todo)
                            seen.update({_key(cands[i]): s for i, s in got.items()})
                            top = pick({0: cur_s, **{i: s for i, s in got.items()}}, min_gain)
                            if top:
                                cur, cur_s, moved = cands[top], got[top], True
                        if not moved or used >= budget:
                            break
                    scores[len(cands)] = cur_s                      # the refined winner
                    cands.append(cur)
                    best = len(cands) - 1
            if best is None:
                log.warning("[%s] %s: every trial failed; the params file is unchanged", name, base)
            else:
                mparams.save(name, base, cands[best], cfg.params_file)
                changed = {k: v for k, v in cands[best].items() if v != start.get(k)}
                log.info("[%s] %-20s best %s=%.4f -> %s", name, base, metric, scores[best], cands[best])
                summary.append(dict(dataset=name, method=base, current=scores.get(0, np.nan), tuned=scores[best],
                                    gain=scores[best] - scores.get(0, np.nan), evaluations=len(rows_of(rows, name, base)),
                                    changed=json.dumps(changed, default=str), params=json.dumps(cands[best], default=str)))
            out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).sort_values(["dataset", "method", "stage", "trial"]).to_csv(out / "tuning_log.csv", index=False)
            if summary:
                write_summary(pd.DataFrame(summary), out, metric)
                write_report(pd.DataFrame(rows), pd.DataFrame(summary), out, metric)
    return pd.DataFrame(rows)


def rows_of(rows: list[dict], dataset: str, method: str) -> list[dict]:
    return [r for r in rows if r["dataset"] == dataset and r["method"] == method]


def param_effects(log_: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per dataset x method x parameter: the best full-data score reached with each value tried. ``spread`` (best value's score
    minus the worst value's) ranks which parameters mattered; it is a rough guide, since trials change several parameters at once."""
    f = log_[log_.stage.isin(FULL)].dropna(subset=[metric]) if len(log_) else log_
    out = []
    for (ds, m), g in f.groupby(["dataset", "method"]):
        P = pd.DataFrame([json.loads(p) for p in g.params], index=g.index)
        for k in P:
            per = g[metric].groupby(P[k].astype(str)).max()
            if len(per) > 1:
                out.append(dict(dataset=ds, method=m, param=k, best_value=per.idxmax(), best_score=per.max(), worst_value=per.idxmin(),
                                worst_score=per.min(), spread=per.max() - per.min(), values_tried=len(per)))
    cols = ["dataset", "method", "param", "best_value", "best_score", "worst_value", "worst_score", "spread", "values_tried"]
    return pd.DataFrame(out, columns=cols).sort_values(["dataset", "method", "spread"], ascending=[True, True, False])


def write_report(log_: pd.DataFrame, s: pd.DataFrame, out: Path, metric: str) -> None:
    """tuning_param_effects.csv and tuning_report.md: for each method what changed, the gain, the cost, which parameters mattered."""
    eff = param_effects(log_, metric)
    eff.to_csv(out / "tuning_param_effects.csv", index=False)
    L = [f"# Tuning report ({metric}, validation split)", "",
         "Scores come from private validation folds, so the test split is untouched; run `benchmark` for the reported numbers. "
         "A change is only accepted if it beats the incumbent by more than the noise margin (`--min-gain`).", ""]
    for r in s.itertuples():
        g = log_[(log_.dataset == r.dataset) & (log_.method == r.method)]
        full = g[g.stage.isin(FULL)]
        L += [f"## {r.dataset} / {r.method}: {r.current:.4f} -> {r.tuned:.4f} ({r.gain:+.4f})", "",
              f"- evaluations: {len(g)} ({len(full)} on the full data); search time {g.get('runtime_s', pd.Series(dtype=float)).sum():.0f}s of model time",
              f"- changed from the current setting: {r.changed if json.loads(r.changed) else 'nothing (the current setting was not beaten)'}"]
        e = eff[(eff.dataset == r.dataset) & (eff.method == r.method)].head(3)
        for x in e.itertuples():
            L.append(f"- `{x.param}`: best {x.best_value} ({x.best_score:.4f}) vs worst {x.worst_value} ({x.worst_score:.4f}), spread {x.spread:.4f}")
        L.append("")
    (out / "tuning_report.md").write_text("\n".join(L), encoding="utf-8")


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

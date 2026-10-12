"""Additional result tables built from the final results: long metric table, fold-to-fold variability, per-metric ranks,
paired tests against the best method, and the parameters every method ran with. Used by ``analysis.analyze_results``."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

LOWER_IS_BETTER = {"jsd", "unassigned_fraction", "n_classes_missed", "spatial_homophily_gap_abs"}


def extra_tables(fr: pd.DataFrame, lp: pd.DataFrame, mets: list[str], prim: dict, root: Path) -> dict[str, pd.DataFrame]:
    t: dict[str, pd.DataFrame] = {}
    cols = [f"{m}_mean" for m in mets]
    # every metric as (dataset, method, level, metric, mean, std): easy to filter and to paste as text
    m_ = fr.melt(id_vars=["dataset", "method", "eval_level"], value_vars=cols, var_name="metric", value_name="mean")
    s_ = fr.melt(id_vars=["dataset", "method", "eval_level"], value_vars=[f"{m}_std" for m in mets], var_name="metric", value_name="std")
    m_["metric"], s_["metric"] = m_.metric.str.removesuffix("_mean"), s_.metric.str.removesuffix("_std")
    t["metrics_long"] = m_.merge(s_, on=["dataset", "method", "eval_level", "metric"]).dropna(subset=["mean"])

    lv = "level3" if "level3" in set(lp.eval_level) else lp.eval_level.max()
    lpl = lp[lp.eval_level == lv]
    # variability across folds, with a normal-approximation 95% confidence interval of the mean
    keep = [m for m in ("f1_macro", "f1_weighted", "mcc", "accuracy", "balanced_accuracy") if m in lpl]
    if keep:
        g = lpl.groupby(["dataset", "method"])
        fv = pd.concat({f"{m}_{s}": g[m].agg(s) for m in keep for s in ("mean", "std", "min", "max")}, axis=1).reset_index()
        fv["n_folds"] = g.size().values
        for m in keep:
            fv[f"{m}_ci95"] = 1.96 * fv[f"{m}_std"].fillna(0) / np.sqrt(fv.n_folds.clip(lower=1))
        t["fold_variability"] = fv

    # rank of every method on every metric (1 = best; metrics where lower is better are flipped)
    rk = m_[m_.eval_level == lv].dropna(subset=["mean"]).copy()
    rk["rank"] = np.nan
    low = rk.metric.isin(LOWER_IS_BETTER)
    rk.loc[low, "rank"] = rk[low].groupby(["dataset", "metric"])["mean"].rank(ascending=True, method="min")
    rk.loc[~low, "rank"] = rk[~low].groupby(["dataset", "metric"])["mean"].rank(ascending=False, method="min")
    t["metric_ranks"] = rk[["dataset", "method", "metric", "mean", "rank"]]

    # paired comparison of every method with the best one over the same folds (few folds -> low power; read as a guide)
    from scipy.stats import ttest_rel, wilcoxon
    rows = []
    for ds, d in lpl.groupby("dataset"):
        piv = d.pivot_table(index="fold", columns="method", values="f1_macro")
        if piv.shape[0] < 3:
            continue
        best = piv.mean().idxmax()
        for m in piv.columns.drop(best):
            pr = piv[[best, m]].dropna()
            if len(pr) < 3:
                continue
            same = bool((pr[best] == pr[m]).all())
            try:
                wp = float("nan") if same else float(wilcoxon(pr[best], pr[m]).pvalue)
            except ValueError:
                wp = float("nan")
            tp = float("nan") if same else float(ttest_rel(pr[best], pr[m]).pvalue)
            rows.append(dict(dataset=ds, best_method=best, method=m, n_folds=len(pr), mean_f1_best=pr[best].mean(),
                             mean_f1_method=pr[m].mean(), mean_difference=(pr[best] - pr[m]).mean(), wilcoxon_p=wp, paired_t_p=tp))
    if rows:
        t["significance_vs_best"] = pd.DataFrame(rows)

    pr_rows = []                                              # the exact parameters each method ran with
    for ds in prim:
        f = Path(root) / ds / "params_used.json"
        if f.exists():
            for meth, ps in json.loads(f.read_text(encoding="utf-8")).items():
                pr_rows += [dict(dataset=ds, method=meth, parameter=k, value=json.dumps(v)) for k, v in ps.items()]
    if pr_rows:
        t["params_used"] = pd.DataFrame(pr_rows)
    return t

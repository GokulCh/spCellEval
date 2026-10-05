"""Dataset analysis and results analysis.

``dataset_report``  describes the data itself (composition, rare/common types, marker profiles, spatial
                    neighbourhoods) and runs at benchmark time.
``analyze_results`` reads only files the runner saved, so it can be re-run any time (``cli.py visualize``).
                    It writes ``<out>/analysis/*.csv``, ``insights.md/json`` and a repo-format
                    ``final_results.csv`` per dataset (``;``-separated, as the existing notebooks expect).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..models.spatial import spatial_graph
from ..preprocessing.data import LEVEL_COLUMN, Dataset
from .metrics import composition_table, top_confusions
from .repo_assets import (RUNTIME_THRESH_S, STABILITY_THRESH, category, overall_score, scaling_score, to_level)

RARE = 0.01
BASELINES = {"most_frequent", "stratified"}
# internal metric name -> name used by the repo notebooks / final_results.csv
EXPORT = {"f1_macro": "macro_f1", "pearson": "pearson_corr"}
CLASS_METRICS = ["f1_weighted", "accuracy", "f1_macro", "hierarchical_f1", "mcc", "kappa", "ari", "nmi", "g_mean",
                 "r2", "pearson", "jsd", "jsd_scaled", "sensitivity_macro", "specificity_macro", "rare_accuracy",
                 "abundant_accuracy"]


# ----------------------------------------------------------------------------- dataset report
def level_labels(ds: Dataset, train_level: str) -> dict[str, np.ndarray]:
    """Labels at every granularity: dataset columns when present, else the repo's hierarchy mapping."""
    out = {train_level: ds.y}
    for lv, col in LEVEL_COLUMN.items():
        if lv != train_level and col in ds.meta:
            out[lv] = ds.meta[col].astype(str).to_numpy()
    for lv in ("level2", "level1")[("level3", "level2").index(train_level) if train_level != "level1" else 2:]:
        out.setdefault(lv, to_level(ds.y, lv))
    return out


def _composition(y: np.ndarray) -> pd.DataFrame:
    c = pd.Series(y).value_counts()
    d = pd.DataFrame({"cell_type": c.index, "n_cells": c.values})
    d["percent"] = d.n_cells / d.n_cells.sum() * 100
    d["rank_most_common"] = np.arange(1, len(d) + 1)
    d["cumulative_percent"] = d.percent.cumsum()
    d["is_rare"] = d.percent < RARE * 100
    return d


def dataset_report(ds: Dataset, out: str | Path, train_level: str = "level3", k: int = 10) -> dict:
    """Write dataset_report/ tables for one dataset and return its summary dict."""
    d = Path(out) / ds.name / "dataset_report"
    d.mkdir(parents=True, exist_ok=True)
    summ: dict = dict(dataset=ds.name, modality=ds.modality, n_cells=len(ds), n_markers=len(ds.markers),
                      markers=ds.markers, n_images=int(len(np.unique(ds.groups))) if ds.groups is not None else None,
                      has_spatial=ds.xy is not None, labelled=ds.y is not None)
    X = ds.X.to_numpy()
    summ["mean_fraction_zero"] = float((X == 0).mean())
    pd.DataFrame(dict(marker=ds.markers, mean=X.mean(0), std=X.std(0), median=np.median(X, 0),
                      p99=np.percentile(X, 99, axis=0), fraction_zero=(X == 0).mean(0))
                 ).to_csv(d / "marker_stats.csv", index=False)
    if ds.groups is not None:
        cpi = pd.Series(ds.groups).value_counts()
        cpi.rename_axis("image").rename("n_cells").to_csv(d / "cells_per_image.csv")
        summ["median_cells_per_image"] = float(cpi.median())
    if ds.xy is not None:
        s = np.random.default_rng(0).choice(len(ds), min(20000, len(ds)), replace=False)
        from sklearn.neighbors import NearestNeighbors
        dist = NearestNeighbors(n_neighbors=2).fit(ds.xy[s]).kneighbors(ds.xy[s])[0][:, 1]
        summ["median_nn_distance"] = float(np.median(dist))   # on a <=20k subsample

    if ds.y is None:
        (d / "summary.json").write_text(json.dumps(summ, indent=2))
        return summ

    levels = level_labels(ds, train_level)
    summ["levels"] = {}
    for lv, y in levels.items():
        comp = _composition(y)
        comp.to_csv(d / f"composition_{lv}.csv", index=False)
        p = comp.percent.to_numpy() / 100
        ent = float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0
        summ["levels"][lv] = dict(
            n_types=len(comp), most_common=dict(cell_type=comp.cell_type.iloc[0], n_cells=int(comp.n_cells.iloc[0]),
                                                percent=float(comp.percent.iloc[0])),
            least_common=dict(cell_type=comp.cell_type.iloc[-1], n_cells=int(comp.n_cells.iloc[-1]),
                              percent=float(comp.percent.iloc[-1])),
            top3=comp.cell_type.head(3).tolist(), bottom3=comp.cell_type.tail(3).tolist(),
            imbalance_ratio=float(comp.n_cells.max() / comp.n_cells.min()),
            evenness_shannon=ent, simpson_diversity=float(1 - (p ** 2).sum()),
            n_rare_types=int(comp.is_rare.sum()), rare_types=comp.cell_type[comp.is_rare].tolist(),
            n_types_under_50_cells=int((comp.n_cells < 50).sum()))
        if ds.groups is not None:
            pi = pd.crosstab(ds.groups, y, normalize="index") * 100
            pi.to_csv(d / f"composition_per_image_{lv}.csv")

    y = ds.y                                           # training level: marker profiles + neighbourhoods
    Z = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    means = pd.DataFrame(X, columns=ds.markers).groupby(y).mean()
    zmeans = pd.DataFrame(Z, columns=ds.markers).groupby(y).mean()
    means.to_csv(d / "marker_means_by_celltype.csv")
    zmeans.to_csv(d / "marker_zscores_by_celltype.csv")
    spec = []
    for t, r in zmeans.iterrows():
        top = r.sort_values(ascending=False).head(3)
        spec.append(dict(cell_type=t, top_markers=";".join(top.index), top_z=float(top.iloc[0]),
                         second_z=float(top.iloc[1]) if len(top) > 1 else np.nan))
    pd.DataFrame(spec).sort_values("top_z").to_csv(d / "marker_specificity.csv", index=False)

    if ds.xy is not None:
        classes, inv = np.unique(y, return_inverse=True)
        from scipy.sparse import csr_matrix
        oh = csr_matrix((np.ones(len(inv)), (np.arange(len(inv)), inv)), shape=(len(inv), len(classes)))
        C = np.zeros((len(classes), len(classes)))
        nb = np.asarray((spatial_graph(ds.xy, ds.groups, k) @ oh).todense())
        for c in range(len(classes)):
            C[c] = nb[inv == c].sum(0)
        comp_n = C / np.maximum(C.sum(1, keepdims=True), 1)
        glob = np.bincount(inv) / len(inv)
        with np.errstate(divide="ignore"):
            enr = np.log2(np.maximum(comp_n, 1e-6) / glob[None, :])
        pd.DataFrame(comp_n, index=classes, columns=classes).to_csv(d / "neighborhood_composition.csv")
        pd.DataFrame(enr, index=classes, columns=classes).to_csv(d / "neighborhood_enrichment_log2.csv")
        summ["neighbour_homophily"] = float(np.average(np.diag(comp_n), weights=np.bincount(inv)))
    (d / "summary.json").write_text(json.dumps(summ, indent=2))
    return summ


# ----------------------------------------------------------------------------- results analysis
def _primary_split(g: pd.DataFrame) -> str:
    for s in ("cv", "holdout", "progressive", "all"):
        if (g.split == s).any():
            return s
    return g.split.iloc[0]


def _primary(df: pd.DataFrame, prim: dict) -> pd.DataFrame:
    """Rows of each dataset's primary split plus its split='all' rows (clusterers / priors / repo scripts run once over
    every cell, as in the repo). Progressive: the largest fraction only."""
    out = []
    for ds, s in prim.items():
        g = df[(df.dataset == ds) & (df.split.isin([s, "all"]))]
        if s == "progressive" and len(g):
            g = g[(g.split != "progressive") | (g.fraction == g[g.split == "progressive"].fraction.max())]
        out.append(g)
    return pd.concat(out) if out else df.iloc[:0]


def _read(p: Path) -> pd.DataFrame:
    return pd.read_csv(p) if p.exists() and p.stat().st_size > 2 else pd.DataFrame()


def analyze_results(out: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(out)
    a = root / "analysis"
    a.mkdir(exist_ok=True)
    res = pd.read_csv(root / "benchmark_results.csv")
    lm, pc, fi = (_read(root / f) for f in ("level_metrics.csv", "per_class_results.csv", "feature_importance.csv"))
    ok = res[res.status == "ok"]
    tabs: dict[str, pd.DataFrame] = {}
    prim = {ds: _primary_split(g) for ds, g in res.groupby("dataset")}
    tabs["primary_split"] = pd.DataFrame(dict(dataset=list(prim), primary_split=list(prim.values())))

    # every split side by side: 5-fold CV, 80/20 hold-out, progressive fractions, one-pass ('all') ----------------
    sc = [c for c in ("f1_macro", "f1_weighted", "accuracy", "mcc", "sensitivity_macro", "specificity_macro", "rare_accuracy",
                      "runtime_s") if c in ok]
    if sc:
        g = ok.assign(fraction=ok.fraction.fillna(-1)).groupby(["dataset", "method", "split", "fraction"])
        bs = g[sc].mean().reset_index()
        bs["n_runs"] = g.size().values
        bs["fraction"] = bs.fraction.replace(-1, float("nan"))
        tabs["by_split"] = bs.sort_values(["dataset", "method", "split", "fraction"])

    # failures ---------------------------------------------------------------------------------
    tabs["failures"] = (res[res.status != "ok"].groupby(["dataset", "method", "status", "error"], dropna=False)
                        .size().reset_index(name="n_runs"))
    tabs["run_status"] = res.groupby(["dataset", "method", "status"]).size().reset_index(name="n_runs")

    # final results (repo format) ------------------------------------------------------------
    if len(lm):
        key = ["dataset", "method", "split", "fold", "fraction"]
        lm = lm.merge(res[key + ["runtime_s", "mem_mb", "status"]], on=key, how="left")
        lp = _primary(lm[lm.status == "ok"], prim)
        mets = [m for m in CLASS_METRICS if m in lp]
        g = lp.groupby(["dataset", "method", "eval_level"])
        fr = g[mets].mean().add_suffix("_mean").join(g[mets].std().add_suffix("_std")).reset_index()
        fr["n_runs"] = g.size().values
        fr["run_time"] = g["runtime_s"].mean().values
        fr["mem_mb"] = g["mem_mb"].mean().values
        fr["stability"] = np.where(fr.n_runs > 1, 1 - fr["f1_weighted_std"] / STABILITY_THRESH, np.nan)
        fr["runtime_scaled"] = np.clip(1 - fr.run_time / RUNTIME_THRESH_S, 0, 1)
        fr["scaling_score"] = fr.method.map(scaling_score)
        fr["scalability"] = (fr.scaling_score + fr.runtime_scaled) / 2
        fr["category"] = fr.method.map(category)
        fr["overall_performance"] = fr.apply(
            lambda r: overall_score({m: r[f"{m}_mean"] for m in mets}, r.eval_level), axis=1)
        tabs["final_results"] = fr
        # per-dataset repo-format file (';'-separated, level_3 style names)
        for ds, g2 in fr.groupby("dataset"):
            ex = g2.rename(columns={f"{k}_{s}": f"{v}_{s}" for k, v in EXPORT.items() for s in ("mean", "std")})
            ex = ex.assign(level=ex.eval_level.str.replace("level", "level_"))
            (root / ds).mkdir(exist_ok=True)
            ex.to_csv(root / ds / "final_results.csv", index=False, sep=";")
        # ranking ------------------------------------------------------------------------------
        l3 = fr[fr.eval_level == fr.eval_level.max()] if "level3" not in set(fr.eval_level) else fr[fr.eval_level == "level3"]
        rk = l3.copy()
        rk["rank"] = rk.groupby("dataset").overall_performance.rank(ascending=False, method="min")
        tabs["method_ranking"] = rk[["dataset", "method", "category", "overall_performance", "rank", "f1_weighted_mean",
                                     "f1_macro_mean", "mcc_mean", "run_time", "stability", "scalability"]
                                    ].sort_values(["dataset", "rank"])
        avg = rk.groupby(["method", "category"]).agg(mean_rank=("rank", "mean"), mean_overall=("overall_performance", "mean"),
                                                      n_datasets=("dataset", "nunique")).reset_index().sort_values("mean_rank")
        tabs["method_ranking_overall"] = avg
        # funky-heatmap input for src/plotting/funky_heatmap.R
        fh = l3.groupby("method").mean(numeric_only=True).reset_index()
        tabs["funky_heatmap_input"] = fh.rename(columns={
            "method": "method", "f1_weighted_mean": "Weighted F1", "hierarchical_f1_mean": "Hierarchical F1",
            "f1_macro_mean": "Macro F1", "mcc_mean": "MCC", "ari_mean": "ARI", "jsd_scaled_mean": "JSD Scaled",
            "overall_performance": "Overall Performance", "stability": "Stability", "scalability": "Scalability"}
        )[["method", "Weighted F1", "Hierarchical F1", "Macro F1", "MCC", "ARI", "JSD Scaled", "Overall Performance",
           "Stability", "Scalability"]]
        tabs["level_comparison"] = fr.pivot_table(index=["dataset", "method"], columns="eval_level",
                                                  values="f1_macro_mean").reset_index()
        pm = lp[mets].dropna(axis=1, how="all")
        tabs["metric_correlation"] = pm.corr().rename_axis("metric").reset_index()
        # efficiency / Pareto
        ef = l3[["dataset", "method", "category", "overall_performance", "run_time", "mem_mb", "scalability"]].copy()
        ef["pareto_optimal"] = False
        for ds, g3 in ef.groupby("dataset"):
            for i, r in g3.iterrows():
                dom = ((g3.overall_performance >= r.overall_performance) & (g3.run_time <= r.run_time)
                       & ((g3.overall_performance > r.overall_performance) | (g3.run_time < r.run_time))).any()
                ef.loc[i, "pareto_optimal"] = not dom
        tabs["efficiency"] = ef

    # unsupervised QC ----------------------------------------------------------------------------
    qc = [c for c in ("silhouette", "davies_bouldin", "marker_enrichment", "pseudo_consistency", "knn_consistency",
                      "n_predicted_types") if c in ok]
    if qc and ok[qc].notna().any().any():
        tabs["unsupervised_qc"] = ok.groupby(["dataset", "method"])[qc].mean().reset_index()

    # data scarcity --------------------------------------------------------------------------------
    prog = ok[ok.split == "progressive"]
    if len(prog):
        sc = prog.groupby(["dataset", "method", "fraction"])[["f1_macro", "f1_weighted", "accuracy"]].mean().reset_index()
        def to95(g):
            m = g.f1_macro.max()
            hit = g[g.f1_macro >= 0.95 * m]
            return hit.fraction.min() if m > 0 else np.nan
        f95 = sc.groupby(["dataset", "method"]).apply(to95, include_groups=False).rename("fraction_to_95pct_of_best").reset_index()
        tabs["data_scarcity"] = sc.merge(f95, on=["dataset", "method"])

    # per-class analysis -----------------------------------------------------------------------------
    if len(pc):
        pcp = _primary(pc.merge(res[["dataset", "method", "split", "fold", "fraction", "status"]],
                                on=["dataset", "method", "split", "fold", "fraction"]).query("status == 'ok'"), prim)
        byc = pcp.groupby(["dataset", "method", "cell_type"]).agg(
            support=("support", "mean"), prevalence=("prevalence", "mean"), precision=("precision", "mean"),
            recall=("recall", "mean"), specificity=("specificity", "mean"), f1=("f1", "mean"),
            f1_std=("f1", "std")).reset_index()
        tabs["per_class_by_method"] = byc
        real = byc[~byc.method.isin(BASELINES) & (byc.support > 0)]
        cs = real.groupby(["dataset", "cell_type"]).agg(
            prevalence=("prevalence", "mean"), mean_f1=("f1", "mean"), min_f1=("f1", "min"), max_f1=("f1", "max"),
            mean_recall=("recall", "mean"), mean_precision=("precision", "mean"), n_methods=("method", "nunique")
        ).reset_index()
        best = real.loc[real.groupby(["dataset", "cell_type"]).f1.idxmax()][["dataset", "cell_type", "method"]].rename(columns={"method": "best_method"})
        worst = real.loc[real.groupby(["dataset", "cell_type"]).f1.idxmin()][["dataset", "cell_type", "method"]].rename(columns={"method": "worst_method"})
        cs = cs.merge(best).merge(worst)
        cs["difficulty_rank"] = cs.groupby("dataset").mean_f1.rank(method="min")      # 1 = hardest
        tabs["class_difficulty"] = cs.sort_values(["dataset", "difficulty_rank"])
        cor = []
        for ds, g4 in cs.groupby("dataset"):
            if len(g4) > 2:
                r = spearmanr(np.log10(g4.prevalence.clip(lower=1e-6)), g4.mean_f1)
                cor.append(dict(dataset=ds, spearman_logprev_vs_f1=r.statistic, p_value=r.pvalue, n_classes=len(g4)))
        tabs["prevalence_vs_f1"] = pd.DataFrame(cor)

    # predictions-based: confusions, composition bias -------------------------------------------------------
    conf_rows, comp_rows = [], []
    for ds, p in prim.items():
        sel = _primary(ok[ok.dataset == ds], {ds: p})
        by_method: dict[str, list] = {}
        for r in sel[sel.pred_file.fillna("").astype(str).str.len() > 0].itertuples():
            if not Path(r.pred_file).exists():
                continue
            d0 = pd.read_csv(r.pred_file)
            # repo scripts (TACIT, Scyan, Leiden ...) write the truth as 'cell_type'; the runner renames it only in memory
            tcol = next((c for c in ("gt_cell_type", "true_phenotype", "cell_type") if c in d0), None)
            if tcol and "predicted_phenotype" in d0:          # no ground truth / predictions -> nothing to compare
                by_method.setdefault(r.method, []).append(d0[[tcol, "predicted_phenotype"]].set_axis(["true_phenotype", "predicted_phenotype"], axis=1))
        for m, parts in by_method.items():
            d = pd.concat(parts, ignore_index=True)
            cm = pd.crosstab(d.true_phenotype.astype(str), d.predicted_phenotype.astype(str))
            cdir = a / "confusion_matrices" / ds
            cdir.mkdir(parents=True, exist_ok=True)
            cm.to_csv(cdir / f"{m}.csv")
            if m not in BASELINES:
                t = top_confusions(d.true_phenotype, d.predicted_phenotype, 50)
                t.insert(0, "method", m)
                t.insert(0, "dataset", ds)
                conf_rows.append(t)
            c = composition_table(d.true_phenotype, d.predicted_phenotype)
            c.insert(0, "method", m)
            c.insert(0, "dataset", ds)
            comp_rows.append(c)
    if conf_rows:
        cf = pd.concat(conf_rows)
        tabs["top_confusions_by_method"] = cf
        tabs["top_confusions"] = (cf.groupby(["dataset", "true", "predicted"]).agg(
            total_errors=("count", "sum"), n_methods=("method", "nunique"),
            mean_fraction_of_true_class=("fraction_of_true_class", "mean")).reset_index()
            .sort_values(["dataset", "total_errors"], ascending=[True, False]))
    if comp_rows:
        cr = pd.concat(comp_rows)
        cr["bias_pct_points"] = cr.predicted_percentage - cr.true_percentage
        cr["pred_to_true_ratio"] = cr.predicted_percentage / cr.true_percentage.replace(0, np.nan)
        tabs["composition_recovery"] = cr

    # feature importance ---------------------------------------------------------------------------
    if len(fi):
        fp = fi[fi.split == fi.groupby("dataset").split.transform(lambda s: _primary_split(pd.DataFrame({"split": s})))]
        fp = fp.groupby(["dataset", "method", "marker"]).importance.mean().reset_index()
        fp["importance_norm"] = fp.importance / fp.groupby(["dataset", "method"]).importance.transform("sum").replace(0, np.nan)
        fp["rank_in_method"] = fp.groupby(["dataset", "method"]).importance_norm.rank(ascending=False, method="min")
        tabs["feature_importance_by_method"] = fp
        cons = fp.groupby(["dataset", "marker"]).agg(mean_importance=("importance_norm", "mean"),
                                                     mean_rank=("rank_in_method", "mean"),
                                                     n_methods=("method", "nunique")).reset_index()
        cons["consensus_rank"] = cons.groupby("dataset").mean_importance.rank(ascending=False, method="min")
        tabs["feature_importance_consensus"] = cons.sort_values(["dataset", "consensus_rank"])

    # spatial gain --------------------------------------------------------------------------------------
    if len(lm) and "final_results" in tabs:
        f3 = tabs["final_results"]
        f3 = f3[f3.eval_level == "level3"] if "level3" in set(f3.eval_level) else f3
        sg = []
        for ds, g5 in f3.groupby("dataset"):
            sc_ = g5.set_index("method").overall_performance
            pairs = [(m, m.removesuffix("+vote")) for m in sc_.index if m.endswith("+vote")]
            pairs += [("spatial_gnn", "logistic_regression"), ("knn_smooth", "random_forest")]
            for m, b in pairs:
                if m in sc_.index and b in sc_.index:
                    sg.append(dict(dataset=ds, spatial_method=m, non_spatial_reference=b,
                                   overall_spatial=sc_[m], overall_reference=sc_[b], gain=sc_[m] - sc_[b]))
        if sg:
            tabs["spatial_gain"] = pd.DataFrame(sg)

    for n, t in tabs.items():
        t.to_csv(a / f"{n}.csv", index=False)
    write_insights(root, tabs, res)
    return tabs


# ----------------------------------------------------------------------------- insights
def _pct(x) -> str:
    return f"{x:.2f}%"


def write_insights(root: Path, tabs: dict, res: pd.DataFrame) -> None:
    md, js = ["# Automated insights\n"], {}
    for ds in sorted(res.dataset.unique()):
        s, j = [f"\n## {ds}\n"], {}
        if "prior_source" in res and res[(res.dataset == ds)].prior_source.astype(str).str.startswith("auto-draft").any():
            s.append("- **Caveat**: the marker methods (" + ", ".join(sorted(res[(res.dataset == ds) & res.prior_source.astype(str).str.startswith("auto-draft")].method.unique()))
                     + ") used a decision matrix drafted from this dataset's own labels, so their scores are circular, not independent prior knowledge.")
        rep = root / ds / "dataset_report" / "summary.json"
        if rep.exists():
            r = json.loads(rep.read_text())
            s.append(f"- **Data**: {r['n_cells']:,} cells, {r['n_markers']} markers"
                     + (f", {r['n_images']} images" if r.get("n_images") else "") + ".")
            for lv, v in r.get("levels", {}).items():
                mc, lc = v["most_common"], v["least_common"]
                s.append(f"- **{lv}** ({v['n_types']} types): most common **{mc['cell_type']}** ({mc['n_cells']:,} cells, {_pct(mc['percent'])}); "
                         f"least common **{lc['cell_type']}** ({lc['n_cells']:,} cells, {_pct(lc['percent'])}); "
                         f"imbalance {v['imbalance_ratio']:.0f}:1; {v['n_rare_types']} rare type(s) (<1%)"
                         + (f": {', '.join(v['rare_types'][:8])}" if v["rare_types"] else "") + ".")
                j[f"{lv}_composition"] = v
            if "neighbour_homophily" in r:
                s.append(f"- **Spatial**: {r['neighbour_homophily']*100:.1f}% of a cell's spatial neighbours share its type on average.")
            ms = root / ds / "dataset_report" / "marker_specificity.csv"
            if ms.exists():
                m = pd.read_csv(ms).head(3)
                s.append("- **Least marker-distinct types** (lowest top-marker z): "
                         + ", ".join(f"{t} ({z:.2f})" for t, z in zip(m.cell_type, m.top_z)) + ".")
        mr = tabs.get("method_ranking")
        if mr is not None and len(mr[mr.dataset == ds]):
            g = mr[mr.dataset == ds].dropna(subset=["overall_performance"]).sort_values("rank")
            b, w = g.iloc[0], g.iloc[-1]
            s.append(f"- **Best method**: {b.method} (overall {b.overall_performance:.3f}, macro-F1 {b.f1_macro_mean:.3f}); "
                     f"worst: {w.method} ({w.overall_performance:.3f}).")
            top_by_cat = g.loc[g.groupby("category").overall_performance.idxmax()]
            s.append("- **Best per category**: " + "; ".join(f"{r.category}: {r.method} ({r.overall_performance:.2f})"
                                                           for r in top_by_cat.itertuples()) + ".")
            base = g[g.method.isin(BASELINES)].overall_performance.max()
            if base == base:
                lose = g[(~g.method.isin(BASELINES)) & (g.overall_performance <= base)].method.tolist()
                if lose:
                    s.append(f"- **Not better than the best baseline**: {', '.join(lose)}.")
            j["best_method"], j["worst_method"] = b.method, w.method
        ef = tabs.get("efficiency")
        if ef is not None and len(ef[ef.dataset == ds]):
            e = ef[ef.dataset == ds]
            s.append(f"- **Pareto-optimal (performance vs runtime)**: {', '.join(e[e.pareto_optimal].method)}; "
                     f"fastest: {e.loc[e.run_time.idxmin()].method} ({e.run_time.min():.1f}s).")
        cd = tabs.get("class_difficulty")
        if cd is not None and len(cd[cd.dataset == ds]):
            g = cd[cd.dataset == ds].sort_values("mean_f1")
            s.append("- **Hardest cell types** (mean F1 over methods): "
                     + ", ".join(f"{r.cell_type} ({r.mean_f1:.2f}, {r.prevalence*100:.2f}% of cells)" for r in g.head(3).itertuples()) + ".")
            s.append("- **Easiest cell types**: "
                     + ", ".join(f"{r.cell_type} ({r.mean_f1:.2f})" for r in g.tail(3).iloc[::-1].itertuples()) + ".")
            j["hardest"], j["easiest"] = g.cell_type.head(3).tolist(), g.cell_type.tail(3).tolist()
        pv = tabs.get("prevalence_vs_f1")
        if pv is not None and len(pv[pv.dataset == ds]):
            r = pv[pv.dataset == ds].iloc[0]
            s.append(f"- **Abundance vs accuracy**: Spearman rho = {r.spearman_logprev_vs_f1:.2f} between log class prevalence and mean F1 "
                     f"({'rarer types are harder' if r.spearman_logprev_vs_f1 > 0.3 else 'abundance is not a strong predictor' if abs(r.spearman_logprev_vs_f1) <= 0.3 else 'rarer types are easier'}).")
        tc = tabs.get("top_confusions")
        if tc is not None and len(tc[tc.dataset == ds]):
            r = tc[tc.dataset == ds].iloc[0]
            s.append(f"- **Most confused pair**: {r.true} -> {r.predicted} ({int(r.total_errors):,} errors across {int(r.n_methods)} methods).")
        cr = tabs.get("composition_recovery")
        if cr is not None and len(cr[cr.dataset == ds]):
            g = cr[(cr.dataset == ds) & ~cr.method.isin(BASELINES)].groupby("cell_type").bias_pct_points.mean().sort_values()
            s.append(f"- **Composition bias**: most under-predicted **{g.index[0]}** ({g.iloc[0]:+.2f} pp), most over-predicted "
                     f"**{g.index[-1]}** ({g.iloc[-1]:+.2f} pp).")
        fc = tabs.get("feature_importance_consensus")
        if fc is not None and len(fc[fc.dataset == ds]):
            s.append("- **Most informative markers (consensus)**: " + ", ".join(fc[fc.dataset == ds].sort_values("consensus_rank").marker.head(5)) + ".")
        dsc = tabs.get("data_scarcity")
        if dsc is not None and len(dsc[dsc.dataset == ds]):
            f = dsc[dsc.dataset == ds].drop_duplicates("method")
            s.append(f"- **Data scarcity**: median training fraction needed for 95% of best macro-F1 = {f.fraction_to_95pct_of_best.median()*100:.0f}% of the data.")
        sg = tabs.get("spatial_gain")
        if sg is not None and len(sg[sg.dataset == ds]):
            g = sg[sg.dataset == ds]
            s.append(f"- **Spatial information**: mean overall-score gain {g.gain.mean():+.3f} "
                     f"({', '.join(f'{r.spatial_method} vs {r.non_spatial_reference}: {r.gain:+.3f}' for r in g.itertuples())}).")
        uq = tabs.get("unsupervised_qc")
        if uq is not None and len(uq[uq.dataset == ds]):
            g = uq[uq.dataset == ds].dropna(subset=["silhouette"])
            if len(g):
                s.append(f"- **Best cluster separation (silhouette)**: {g.loc[g.silhouette.idxmax()].method} ({g.silhouette.max():.3f}).")
        fl = tabs.get("failures")
        if fl is not None and len(fl[fl.dataset == ds]):
            g = fl[fl.dataset == ds]
            s.append("- **Did not run**: " + "; ".join(f"{m} ({st})" for m, st in sorted(set(zip(g.method, g.status)))) + ".")
        md += s
        js[ds] = j
    (root / "analysis" / "insights.md").write_text("\n".join(md), encoding="utf-8")
    (root / "analysis" / "insights.json").write_text(json.dumps(js, indent=2, default=str))

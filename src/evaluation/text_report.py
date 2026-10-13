"""Plain-text results for reading or pasting into an AI model: every table that matters in one markdown file, no images needed.

    <out>/summary/AI_REPORT.md        the full report (run overview, parameters, leaderboards with all metrics, per-class results,
                                      confusions, composition bias, data scarcity, tests, marker importance, text confusion matrices,
                                      index of the figures and their text twins)
    <out>/summary/AI_REPORT_SHORT.md  leaderboards and insights only (a few screens)
    <out>/summary/AI_REPORT.json      the same leaderboards as records, for programs

Every figure (PNG / SVG) has a ``.txt`` twin next to it with the numbers it shows (see ``figure_text``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .reports import to_md

GLOSSARY = """| metric | meaning |
|---|---|
| f1_macro | F1 averaged over cell types, each type counts equally (punishes failing on rare types) |
| f1_weighted | F1 weighted by type size (dominated by abundant types) |
| accuracy | share of cells with the right label |
| balanced_accuracy / recall_macro | recall averaged over types |
| precision_macro / precision_weighted | share of predicted labels that are right (macro / size-weighted) |
| mcc, kappa | chance-corrected agreement (-1..1) |
| ari, nmi, v_measure, homogeneity, completeness, fowlkes_mallows | agreement of predictions and labels seen as two clusterings |
| g_mean | geometric mean of per-type recall (0 if a type is missed completely) |
| hierarchical_f1 | F1 that gives partial credit when the predicted type is a parent / child of the true one |
| jsd_scaled, r2, pearson | how well the predicted cell-type composition matches the true one (1 = identical) |
| rare_accuracy / abundant_accuracy / f1_rare_macro | recall / F1 on types below / above 1% of the cells |
| unassigned_fraction | share of cells left as undefined / unknown / others |
| n_classes_missed | cell types present in the truth that are never predicted |
| spatial_homophily_pred / _true / _gap | share of spatial neighbours sharing a cell's label, predicted vs truth (gap = pred - true) |
| overall_performance | weighted combination of the above used by the benchmark (higher is better) |
| run_time, mem_mb, stability, scalability | seconds, memory, 1 - F1 spread across folds, scaling score |
"""
LEADER = ["method", "category", "overall_performance", "f1_macro_mean", "f1_macro_std", "f1_weighted_mean", "accuracy_mean",
          "balanced_accuracy_mean", "precision_macro_mean", "recall_macro_mean", "mcc_mean", "kappa_mean", "ari_mean", "nmi_mean",
          "g_mean_mean", "jsd_scaled_mean", "rare_accuracy_mean", "unassigned_fraction_mean", "run_time", "mem_mb"]
SHORT = ["method", "category", "overall_performance", "f1_macro_mean", "f1_weighted_mean", "accuracy_mean", "mcc_mean", "run_time"]


def _read(p: Path) -> pd.DataFrame:
    return pd.read_csv(p) if p.exists() and p.stat().st_size > 2 else pd.DataFrame()


def _tbl(df: pd.DataFrame, cols: list[str] | None = None, n: int = 40, note: str = "") -> str:
    if df is None or df.empty:
        return "_(none)_\n"
    d = df[[c for c in cols if c in df]] if cols else df
    d = d.dropna(axis=1, how="all")
    s = to_md(d.head(n).copy())
    return s + (f"\n_(showing {n} of {len(d)} rows{note})_\n" if len(d) > n else "")


def write_text_report(out: str | Path, max_rows: int = 40) -> Path:
    root = Path(out)
    a, s = root / "analysis", root / "summary"
    s.mkdir(exist_ok=True)
    A = {p.stem: _read(p) for p in a.glob("*.csv")}
    res = _read(root / "benchmark_results.csv")
    fr = A.get("final_results", pd.DataFrame())
    L = ["# spCellEval results (text report)\n",
         "Everything below is plain text so it can be pasted into an AI model. Figures are in `figures/` and `<dataset>/figures/`; "
         "each has a `.txt` twin with its numbers (index at the end).\n", "## Metric glossary\n", GLOSSARY]
    short = ["# spCellEval results (short)\n"]
    js: dict = {}

    # ----------------------------------------------------------------- run overview
    if len(res):
        st = res.groupby(["dataset", "status"]).size().unstack(fill_value=0).reset_index()
        L += ["## 1. Run overview\n", f"datasets: {', '.join(sorted(res.dataset.unique()))}; methods: {res.method.nunique()}; "
              f"rows: {len(res)}\n", "Rows by status:\n", _tbl(st), ""]
        fl = A.get("failures", pd.DataFrame())
        if len(fl):
            fl = fl.assign(error=fl.error.astype(str).str.slice(0, 160))
            L += ["Runs that did not complete:\n", _tbl(fl, n=max_rows)]
        short += ["## Run overview\n",f"datasets: {', '.join(sorted(res.dataset.unique()))}; methods: {res.method.nunique()}\n", _tbl(st)]
    # ----------------------------------------------------------------- sanity checks on the predictions
    dg = A.get("prediction_diagnostics", pd.DataFrame())
    if len(dg):
        warn = []
        for ds, g in dg.groupby("dataset"):
            for grp in sorted({s for s in g.identical_to.fillna("") if s}):
                warn.append(f"- **{ds}**: these methods produced EXACTLY the same predictions: {grp} (a shared input or a bug, not a coincidence)")
            for r in g.itertuples():
                if r.n_true_classes > 2 and r.n_predicted_classes <= 2:
                    warn.append(f"- **{ds}/{r.method}**: predicts only {r.n_predicted_classes} class(es) ({r.top_predicted_class} = {r.top_predicted_share:.0%}) while the truth has {r.n_true_classes}")
                if r.unassigned_share > 0.5:
                    warn.append(f"- **{ds}/{r.method}**: {r.unassigned_share:.0%} of the cells are left unassigned (undefined / unknown / others)")
        L += ["## Sanity checks on the predictions\n", "\n".join(warn) if warn else "No identical, single-class or mostly-unassigned predictions found.", "",
              _tbl(dg, ["dataset", "method", "n_true_classes", "n_predicted_classes", "top_predicted_class", "top_predicted_share", "unassigned_share", "correct_share"], 80)]
        short += ["## Sanity checks\n", "\n".join(warn) if warn else "No identical, single-class or mostly-unassigned predictions found.", ""]
    # ----------------------------------------------------------------- parameters
    pu = A.get("params_used", pd.DataFrame())
    if len(pu):
        L.append("## 2. Parameters every method ran with\n")
        for ds, g in pu.groupby("dataset"):
            line = g.groupby("method").apply(lambda d: ", ".join(f"{k}={v}" for k, v in zip(d.parameter, d.value) if v not in ("null", "[]")), include_groups=False)
            L += [f"### {ds}\n", "\n".join(f"- **{m}**: {t or '(no parameters)'}" for m, t in line.items()) + "\n"]
    for name, title in (("tuning_summary", "Tuning: current vs tuned score (validation split)"), ("seed_variability", "Macro-F1 over several seeds")):
        for p in (root / f"{name}.csv", s / f"{name}.csv"):
            if p.exists():
                L += [f"## {title}\n", _tbl(_read(p), n=max_rows)]
                break
    ins = a / "insights.md"
    if ins.exists():
        L += ["## 3. Automatic insights\n", ins.read_text(encoding="utf-8"), ""]
        short += ["## Insights\n", ins.read_text(encoding="utf-8")]

    # ----------------------------------------------------------------- per dataset
    L.append("## 4. Results per dataset\n")
    short.append("## Leaderboards (level3)\n")
    for ds in sorted(fr.dataset.unique()) if len(fr) else []:
        f = fr[fr.dataset == ds]
        lv = "level3" if "level3" in set(f.eval_level) else f.eval_level.max()
        lb = f[f.eval_level == lv].sort_values("overall_performance", ascending=False)
        L += [f"### {ds}\n", f"**Leaderboard ({lv}, mean over folds of the primary split; std for macro-F1)**\n", _tbl(lb, LEADER, max_rows)]
        short += [f"### {ds}\n", _tbl(lb, SHORT, 15)]
        num = lb.drop(columns=[c for c in lb if lb[c].dtype == object and c not in ("method", "category")]).round(4)
        js[ds] = num.astype(object).where(num.notna(), None).to_dict("records")
        lc = A.get("level_comparison", pd.DataFrame())
        if len(lc[lc.dataset == ds]) and lc.shape[1] > 3:
            L += ["**Macro-F1 at each hierarchy level**\n", _tbl(lc[lc.dataset == ds].drop(columns="dataset"), n=max_rows)]
        for key, title, cols in (
                ("fold_variability", "Fold-to-fold variability of macro-F1", ["method", "f1_macro_mean", "f1_macro_std", "f1_macro_min", "f1_macro_max", "f1_macro_ci95", "n_folds"]),
                ("significance_vs_best", "Paired comparison with the best method over folds (few folds = low power)",
                 ["method", "best_method", "mean_f1_best", "mean_f1_method", "mean_difference", "n_folds", "wilcoxon_p", "paired_t_p"]),
                ("class_difficulty", "Hardest cell types (mean F1 over methods)", ["cell_type", "prevalence", "mean_f1", "min_f1", "max_f1", "best_method", "worst_method"]),
                ("top_confusions", "Most frequent mistakes (true -> predicted, summed over methods)", ["true", "predicted", "total_errors", "n_methods", "mean_fraction_of_true_class"]),
                ("data_scarcity", "Macro-F1 vs training fraction", None),
                ("spatial_gain", "Gain from spatial information", ["spatial_method", "non_spatial_reference", "overall_spatial", "overall_reference", "gain"]),
                ("unsupervised_qc", "Ground-truth-free quality of the clusterings", None),
                ("feature_importance_consensus", "Most informative markers (consensus over methods)", ["marker", "mean_importance", "mean_rank", "n_methods", "consensus_rank"])):
            t = A.get(key, pd.DataFrame())
            t = t[t.dataset == ds].drop(columns="dataset") if len(t) and "dataset" in t else pd.DataFrame()
            if key == "data_scarcity" and len(t):
                t = t.pivot_table(index="method", columns="fraction", values="f1_macro").reset_index()
                t.columns = [str(c) if c == "method" else f"f1@{float(c):.0%}" for c in t.columns]
            if key == "feature_importance_consensus" and len(t):
                t = t.sort_values("consensus_rank")
            if key == "fold_variability" and len(t):
                t = t.sort_values("f1_macro_mean", ascending=False)
            if len(t):
                L += [f"**{title}**\n", _tbl(t, cols, 15 if key in ("class_difficulty", "top_confusions", "feature_importance_consensus") else max_rows)]
        pc = A.get("per_class_by_method", pd.DataFrame())
        pc = pc[pc.dataset == ds] if len(pc) else pc
        if len(pc):
            top = lb.method.head(6).tolist()
            p = pc[pc.method.isin(top)].pivot_table(index="cell_type", columns="method", values="f1")[[m for m in top if m in set(pc.method)]]
            p.insert(0, "prevalence", pc.groupby("cell_type").prevalence.mean().reindex(p.index))
            L += ["**Per-cell-type F1 of the 6 best methods (rows by abundance)**\n", _tbl(p.sort_values("prevalence", ascending=False).reset_index(), n=60)]
        cr = A.get("composition_recovery", pd.DataFrame())
        cr = cr[(cr.dataset == ds) & ~cr.method.isin(["most_frequent", "stratified"])] if len(cr) else cr
        if len(cr):
            g = cr.groupby("cell_type").agg(true_pct=("true_percentage", "mean"), predicted_pct=("predicted_percentage", "mean"),
                                            bias_pp=("bias_pct_points", "mean")).reset_index()
            L += ["**Composition recovery (mean over methods; bias in percentage points)**\n",
                  _tbl(g.reindex(g.bias_pp.abs().sort_values(ascending=False).index), n=15)]
        for m in lb.method.head(3):                                         # confusion matrices as text, row-normalised %
            f_ = a / "confusion_matrices" / ds / f"{m}.csv"
            if f_.exists():
                cm = pd.read_csv(f_, index_col=0)
                if cm.shape[0] <= 25 and cm.shape[1] <= 25:
                    pct = (cm.div(cm.sum(1).replace(0, 1), axis=0) * 100).round(0).astype(int)
                    L += [f"**Confusion matrix of {m}** (rows = true, columns = predicted, % of the true type)\n", to_md(pct.reset_index().rename(columns={pct.index.name or "index": "true"}))]
    # ----------------------------------------------------------------- cross-dataset
    ov = A.get("method_ranking_overall", pd.DataFrame())
    if len(ov):
        L += ["## 5. Across datasets\n", "**Mean rank and overall performance per method**\n", _tbl(ov, n=max_rows)]
        short += ["## Across datasets\n", _tbl(ov, n=20)]
    mr = A.get("metric_ranks", pd.DataFrame())
    if len(mr):
        p = mr.groupby(["method", "metric"])["rank"].mean().unstack()
        L += ["**Mean rank of each method on each metric (1 = best, averaged over datasets)**\n", _tbl(p.reset_index(), n=max_rows)]
    # ----------------------------------------------------------------- figures
    figs = sorted(p for p in root.rglob("figures/**/*.txt"))
    L.append("## 6. Figure index (each figure has a text twin with its numbers)\n")
    for p in figs:
        first = next((ln for ln in p.read_text(encoding="utf-8").splitlines()[1:] if "title=" in ln), "")
        L.append(f"- `{p.relative_to(root).as_posix()}`: {first.split('title=')[-1].split(' x_label')[0].strip(chr(39))[:110]}")
    path = s / "AI_REPORT.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    (s / "AI_REPORT_SHORT.md").write_text("\n".join(short) + "\n", encoding="utf-8")
    (s / "AI_REPORT.json").write_text(json.dumps(js, indent=1, default=str), encoding="utf-8")
    return path

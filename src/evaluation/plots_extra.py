"""More result figures: per-metric ranks, fold variability, precision vs recall, grouped metric bars, rare vs abundant accuracy,
unassigned / missed classes, spatial smoothness, cluster agreement, paired tests, cross-dataset ranks and score spread.
Every figure also gets a text twin (see ``figure_text``) through ``plots._save``."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .plots import _save, color_for_method, heatmap, plt

BASE = ("most_frequent", "stratified")


def _l3(A: dict, ds: str) -> pd.DataFrame:
    """Mean metrics of every method on one dataset (level 3 when present), columns without the ``_mean`` suffix."""
    fr = A["final_results"]
    fr = fr[fr.dataset == ds]
    fr = fr[fr.eval_level == ("level3" if "level3" in set(fr.eval_level) else fr.eval_level.max())] if len(fr) else fr
    t = fr.set_index("method")[[c for c in fr.columns if c.endswith("_mean")]]
    t.columns = [c.removesuffix("_mean") for c in t.columns]
    return t


def _bars(t: pd.DataFrame, cols: list[str], title: str, ylabel: str, figs: Path, name: str, sort: str | None = None) -> list[Path]:
    cols = [c for c in cols if c in t and t[c].notna().any()]
    if t.empty or not cols:
        return []
    t = t.sort_values(sort if sort in cols else cols[0], ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(t) + 2), 4.2))
    w = 0.8 / len(cols)
    x = np.arange(len(t))
    for i, c in enumerate(cols):
        ax.bar(x + (i - (len(cols) - 1) / 2) * w, t[c], w, label=c.replace("_", " "))
    ax.set_xticks(x, t.index, rotation=60, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7, ncol=min(len(cols), 4))
    return _save(fig, figs / name)


def extra_plots(A: dict, figs: Path, ds: str) -> list[Path]:
    done: list[Path] = []
    t = _l3(A, ds)
    if t.empty:
        return done
    done += _bars(t, ["f1_macro", "f1_weighted", "balanced_accuracy", "accuracy", "mcc", "kappa"], f"{ds} - main classification metrics",
                  "score", figs, "metric_bars", "f1_macro")
    done += _bars(t, ["precision_macro", "recall_macro", "precision_weighted", "recall_weighted"], f"{ds} - precision and recall",
                  "score", figs, "precision_recall_bars", "precision_macro")
    done += _bars(t, ["rare_accuracy", "abundant_accuracy", "f1_rare_macro", "f1_abundant_macro"],
                  f"{ds} - rare vs abundant cell types (rare = < 1% of cells)", "score", figs, "rare_vs_abundant", "abundant_accuracy")
    done += _bars(t, ["unassigned_fraction"], f"{ds} - share of cells left unassigned (undefined / unknown / others)", "fraction",
                  figs, "unassigned_fraction")
    done += _bars(t, ["n_classes_missed"], f"{ds} - cell types present in the truth but never predicted", "classes", figs, "classes_missed")
    ca = t[[c for c in ("homogeneity", "completeness", "v_measure", "fowlkes_mallows", "ari", "nmi") if c in t]].dropna(axis=1, how="all")
    if ca.shape[1]:
        done += heatmap(ca.sort_values(ca.columns[0], ascending=False), figs / "cluster_agreement",
                        f"{ds} - agreement of predictions with the labels as clusterings", "plasma", 0, 1, cbar="score")

    if {"precision_macro", "recall_macro"} <= set(t):                          # precision vs recall with iso-F1 curves
        fig, ax = plt.subplots(figsize=(5.5, 5))
        r = np.linspace(0.01, 1, 200)
        for f in (0.2, 0.4, 0.6, 0.8):
            p = f * r / (2 * r - f)
            ok = (p > 0) & (p <= 1.0001)
            ax.plot(r[ok], p[ok], color="#bbb", lw=0.8)
            ax.annotate(f"F1={f}", (r[ok][-1], p[ok][-1]), fontsize=6, color="#888")
        ax.scatter(t.recall_macro, t.precision_macro, c=[color_for_method(m) for m in t.index], s=45, edgecolor="k", linewidth=0.4)
        for m, row in t.iterrows():
            ax.annotate(m, (row.recall_macro, row.precision_macro), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("macro recall")
        ax.set_ylabel("macro precision")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{ds} - precision vs recall (grey = iso-F1)")
        done += _save(fig, figs / "precision_vs_recall")

    if {"spatial_homophily_pred", "spatial_homophily_true"} <= set(t) and t.spatial_homophily_pred.notna().any():
        s = t.sort_values("spatial_homophily_pred", ascending=False)
        fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(s) + 2), 4))
        ax.bar(s.index, s.spatial_homophily_pred, color=[color_for_method(m) for m in s.index])
        ax.axhline(s.spatial_homophily_true.mean(), color="k", ls="--", lw=1, label="ground truth")
        ax.set_ylabel("share of spatial neighbours with the same label")
        ax.set_title(f"{ds} - spatial smoothness of the predictions (dashed = ground truth)")
        ax.tick_params(axis="x", rotation=60)
        ax.legend(frameon=False)
        done += _save(fig, figs / "spatial_homophily")

    mr = A.get("metric_ranks", pd.DataFrame())
    mr = mr[mr.dataset == ds] if len(mr) else mr
    if len(mr):
        p = mr.pivot_table(index="method", columns="metric", values="rank")
        p = p[[c for c in ("f1_macro", "f1_weighted", "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "mcc", "kappa",
                           "ari", "nmi", "g_mean", "jsd_scaled", "rare_accuracy") if c in p]]
        done += heatmap(p.loc[p.mean(1).sort_values().index], figs / "metric_ranks", f"{ds} - rank on each metric (1 = best)", "viridis_r",
                        cbar="rank", fmt="{:.0f}")

    fv = A.get("fold_variability", pd.DataFrame())
    fv = fv[fv.dataset == ds].sort_values("f1_macro_mean") if len(fv) else fv
    if len(fv):
        fig, ax = plt.subplots(figsize=(6.5, max(3, 0.3 * len(fv) + 1)))
        y = np.arange(len(fv))
        ax.errorbar(fv.f1_macro_mean, y, xerr=fv.f1_macro_ci95, fmt="o", color="k", capsize=3, label="mean +/- 95% CI")
        ax.hlines(y, fv.f1_macro_min, fv.f1_macro_max, color="#999", lw=3, alpha=0.5, label="min - max over folds")
        ax.set_yticks(y, fv.method)
        ax.set_xlabel("macro-F1 across folds")
        ax.set_title(f"{ds} - fold-to-fold variability")
        ax.legend(frameon=False, fontsize=7)
        done += _save(fig, figs / "fold_variability")

    sg = A.get("significance_vs_best", pd.DataFrame())
    sg = sg[sg.dataset == ds].sort_values("mean_difference") if len(sg) else sg
    if len(sg):
        fig, ax = plt.subplots(figsize=(6.5, max(3, 0.3 * len(sg) + 1)))
        ax.barh(sg.method, sg.mean_difference, color=["#d62728" if p < 0.05 else "#9ecae1" for p in sg.paired_t_p.fillna(1)])
        for i, r in enumerate(sg.itertuples()):
            ax.annotate(f"p={r.paired_t_p:.3f}" if r.paired_t_p == r.paired_t_p else "p=n/a", (r.mean_difference, i), fontsize=6, xytext=(3, 0),
                        textcoords="offset points", va="center")
        ax.set_xlabel(f"macro-F1 deficit vs best method ({sg.best_method.iloc[0]}); red = paired t-test p < 0.05")
        ax.set_title(f"{ds} - paired comparison with the best method (over folds)")
        done += _save(fig, figs / "significance_vs_best")
    return done


def cross_dataset_plots(A: dict, figs: Path) -> list[Path]:
    done: list[Path] = []
    mr = A.get("method_ranking", pd.DataFrame())
    if len(mr) and mr.dataset.nunique() > 1:
        p = mr.pivot_table(index="method", columns="dataset", values="rank")
        done += heatmap(p.loc[p.mean(1).sort_values().index], figs / "rank_across_datasets", "Method rank on each dataset (1 = best overall performance)",
                        "viridis_r", cbar="rank", fmt="{:.0f}")
        sc = mr[~mr.method.isin(BASE)].dropna(subset=["f1_macro_mean"])
        order = sc.groupby("method").f1_macro_mean.median().sort_values(ascending=False).index
        fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(order) + 2), 4.2))
        ax.boxplot([sc[sc.method == m].f1_macro_mean for m in order], showmeans=True)
        ax.set_xticks(range(1, len(order) + 1), list(order))
        ax.set_ylabel("macro-F1 over datasets")
        ax.set_title("Spread of macro-F1 across datasets (box = quartiles, triangle = mean)")
        ax.tick_params(axis="x", rotation=60)
        done += _save(fig, figs / "score_spread_across_datasets")
    return done

"""Figures (PNG + SVG, as the repo's notebooks save SVG) built from ``analysis/`` tables and saved results.

Colours come from the repo: method categories from ``plotting/method_colors.json`` and datasets from
``plotting/dataset_colors.json``. Heatmap styles follow ``visualizations.ipynb`` (plasma overall-performance
heatmap, ranking / efficiency / scalability scatter).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .repo_assets import CATEGORIES, category, color_for_dataset, color_for_method, method_colors  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})


def _save(fig, path: Path) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    outs = []
    for ext in ("png", "svg"):
        p = path.with_suffix(f".{ext}")
        fig.savefig(p, dpi=250 if ext == "png" else None, bbox_inches="tight")
        outs.append(p)
    plt.close(fig)
    return outs


def _csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p) if p.exists() and p.stat().st_size > 2 else pd.DataFrame()


def _cat_legend(ax, methods=None, **kw):
    cats = [c for c in CATEGORIES if methods is None or any(category(m) == c for m in methods)]
    ax.legend(handles=[plt.Line2D([], [], marker="s", ls="", color=method_colors()[c], label=c) for c in cats],
              frameon=False, fontsize=7, **kw)


def heatmap(df: pd.DataFrame, path: Path, title="", cmap="plasma", vmin=None, vmax=None, annot=True, cbar="",
            xlabel="", ylabel="", fmt="{:.2f}") -> list[Path]:
    if df.empty:
        return []
    r, c = df.shape
    fig, ax = plt.subplots(figsize=(max(4, 0.45 * c + 2.5), max(3, 0.32 * r + 1.5)))
    im = ax.imshow(df.to_numpy(float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(c), df.columns.astype(str), rotation=90, fontsize=7 if c > 25 else 8)
    ax.set_yticks(range(r), df.index.astype(str), fontsize=7 if r > 30 else 8)
    if annot and r * c <= 400:
        v = df.to_numpy(float)
        lo, hi = np.nanmin(v), np.nanmax(v)
        for i in range(r):
            for j in range(c):
                if not np.isnan(v[i, j]):
                    ax.text(j, i, fmt.format(v[i, j]), ha="center", va="center", fontsize=6,
                            color="white" if (v[i, j] - lo) / max(hi - lo, 1e-9) < 0.5 and cmap in ("plasma", "viridis", "Blues") else "black")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=cbar)
    return _save(fig, path)


# ----------------------------------------------------------------------------- dataset figures
def composition_plots(rep: Path, figs: Path, ds: str) -> list[Path]:
    done = []
    for f in sorted(rep.glob("composition_level*.csv")):
        lv = f.stem.split("_")[1]
        c = pd.read_csv(f)
        fig, ax = plt.subplots(figsize=(6, max(3, 0.25 * len(c) + 1)))
        ax.barh(c.cell_type[::-1], c.n_cells[::-1], color=["#d62728" if r else "#4c78a8" for r in c.is_rare[::-1]])
        ax.set_xscale("log")
        ax.set_xlabel("cells (log)")
        ax.set_title(f"{ds} - {lv} class abundance (red = rare, <1%)")
        ax.tick_params(axis="y", labelsize=7)
        done += _save(fig, figs / f"abundance_{lv}")
        fig, ax = plt.subplots(figsize=(12, 1.8))               # stacked bar as in celltype_composition.ipynb
        cmap = plt.get_cmap("tab20", len(c))
        left = 0
        for i, r in enumerate(c.itertuples()):
            ax.barh(["Phenotypes"], [r.percent], left=left, color=cmap(i), label=r.cell_type, height=0.5)
            left += r.percent
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percentage")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=6, frameon=False, ncol=1 + len(c) // 20)
        ax.set_title(f"{ds} - {lv} composition")
        done += _save(fig, figs / f"composition_stacked_{lv}")
    for name, cmap, lab in (("marker_zscores_by_celltype", "RdBu_r", "mean z-score"),
                            ("neighborhood_enrichment_log2", "RdBu_r", "log2 enrichment")):
        d = _csv(rep / f"{name}.csv")
        if len(d):
            d = d.set_index(d.columns[0])
            lim = 2 if "marker" in name else max(1, np.nanpercentile(np.abs(d.to_numpy()), 95))
            done += heatmap(d, figs / name, f"{ds} - {name.replace('_', ' ')}", cmap, -lim, lim, annot=False, cbar=lab,
                            xlabel="marker" if "marker" in name else "neighbour type", ylabel="cell type")
    return done


# ----------------------------------------------------------------------------- results figures
def ranking_plots(A: dict, figs: Path, ds: str) -> list[Path]:
    mr, fr = A["method_ranking"], A["final_results"]
    g = mr[mr.dataset == ds].sort_values("overall_performance")
    done = []
    if len(g):
        err = fr[(fr.dataset == ds) & (fr.eval_level == "level3")].set_index("method").get("f1_weighted_std")
        fig, ax = plt.subplots(figsize=(6, max(3, 0.28 * len(g) + 1)))
        ax.barh(g.method, g.overall_performance, color=[color_for_method(m) for m in g.method],
                xerr=None if err is None else err.reindex(g.method).fillna(0).to_numpy())
        ax.set_xlabel("overall performance (weighted score)")
        ax.set_title(f"{ds} - method ranking")
        _cat_legend(ax, g.method, loc="lower right")
        done += _save(fig, figs / "method_ranking")
    f = fr[fr.dataset == ds]
    mets = [c[:-5] for c in f.columns if c.endswith("_mean") and c[:-5] in
            ("f1_weighted", "hierarchical_f1", "f1_macro", "mcc", "ari", "jsd_scaled", "accuracy", "g_mean", "kappa", "nmi",
             "sensitivity_macro", "specificity_macro", "rare_accuracy", "abundant_accuracy")]
    for lv in sorted(f.eval_level.unique()):
        t = f[f.eval_level == lv].set_index("method")[[m + "_mean" for m in mets]]
        t.columns = mets
        done += heatmap(t.dropna(axis=1, how="all").sort_values("f1_macro", ascending=False), figs / f"metrics_{lv}",
                        f"{ds} - all metrics ({lv})", "plasma", 0, 1, cbar="score")
    return done


def overall_heatmap(A: dict, figs: Path) -> list[Path]:
    fr = A["final_results"]
    p = fr.pivot_table(index="method", columns=["dataset", "eval_level"], values="overall_performance")
    p = p.loc[p.mean(axis=1).sort_values(ascending=False).index]
    p.columns = [f"{a} {b}" for a, b in p.columns]
    return heatmap(p, figs / "overall_performance_heatmap", "Overall performance", "plasma", cbar="weighted score",
                   xlabel="dataset and cell type level")


def efficiency_plots(A: dict, figs: Path) -> list[Path]:
    ef = A["efficiency"].dropna(subset=["overall_performance", "run_time"])
    done = []
    if len(ef):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, (xcol, xl, log) in zip(axes, (("run_time", "runtime (s, log)", True), ("scalability", "scalability score", False))):
            for ds, g in ef.groupby("dataset"):
                ax.scatter(g[xcol].clip(lower=1e-3) if log else g[xcol], g.overall_performance, s=45, edgecolor="k",
                           linewidth=np.where(g.pareto_optimal, 1.6, 0.3), c=[color_for_method(m) for m in g.method])
                for r in g.itertuples():
                    ax.annotate(r.method, (max(getattr(r, xcol), 1e-3) if log else getattr(r, xcol), r.overall_performance),
                                fontsize=6, xytext=(3, 3), textcoords="offset points")
            if log:
                ax.set_xscale("log")
            ax.set_xlabel(xl)
            ax.set_ylabel("overall performance")
        axes[0].set_title("Performance vs runtime (bold edge = Pareto-optimal)")
        axes[1].set_title("Performance vs scalability")
        _cat_legend(axes[1], ef.method, loc="best")
        done += _save(fig, figs / "performance_vs_runtime_scalability")
    return done


def scaling_curves(A: dict, figs: Path) -> list[Path]:
    d = A["data_scarcity"]
    if d.empty:
        return []
    out = []
    for metric in ("f1_macro", "f1_weighted", "accuracy"):
        dsets = sorted(d.dataset.unique())
        fig, axes = plt.subplots(1, len(dsets), figsize=(4.8 * len(dsets) + 2, 4), squeeze=False, sharey=True)
        for ax, ds in zip(axes[0], dsets):
            for m, g in d[d.dataset == ds].groupby("method"):
                ax.plot(g.fraction * 100, g[metric], marker="o", ms=3, lw=1.3, color=color_for_method(m), label=m,
                        ls="--" if m in ("most_frequent", "stratified") else "-")
            ax.set_xscale("log")
            ax.set_xlabel("training data (% of dataset, log)")
            ax.set_title(ds)
        axes[0][0].set_ylabel(metric.replace("_", " "))
        axes[0][-1].legend(frameon=False, fontsize=6, bbox_to_anchor=(1.01, 1), loc="upper left")
        out += _save(fig, figs / f"scaling_curves_{metric}")
    return out


def class_plots(A: dict, figs: Path, ds: str) -> list[Path]:
    pcm, cd = A["per_class_by_method"], A["class_difficulty"]
    done = []
    g = pcm[pcm.dataset == ds]
    if len(g):
        order = g.groupby("cell_type").prevalence.mean().sort_values(ascending=False).index
        p = g.pivot_table(index="method", columns="cell_type", values="f1")[order]
        done += heatmap(p, figs / "per_class_f1_heatmap", f"{ds} - per-class F1 (classes ordered by abundance, common -> rare)",
                        "plasma", 0, 1, cbar="F1", ylabel="method", xlabel="cell type")
    c = cd[cd.dataset == ds].sort_values("mean_f1")
    if len(c):
        fig, axes = plt.subplots(1, 2, figsize=(12, max(3.5, 0.25 * len(c) + 1)))
        axes[0].barh(c.cell_type, c.mean_f1, xerr=[c.mean_f1 - c.min_f1, c.max_f1 - c.mean_f1], color="#4c78a8", ecolor="#999")
        axes[0].set_xlabel("F1 (mean, bars = best/worst method)")
        axes[0].set_title("Class difficulty (hardest at top)")
        axes[0].tick_params(axis="y", labelsize=7)
        axes[1].scatter(c.prevalence * 100, c.mean_f1, s=30, color="#4c78a8")
        for r in c.itertuples():
            axes[1].annotate(r.cell_type, (r.prevalence * 100, r.mean_f1), fontsize=6, xytext=(3, 3), textcoords="offset points")
        axes[1].set_xscale("log")
        axes[1].set_xlabel("class prevalence (% of cells, log)")
        axes[1].set_ylabel("mean F1")
        pv = A.get("prevalence_vs_f1", pd.DataFrame())
        if len(pv[pv.dataset == ds]):
            axes[1].set_title(f"Abundance vs accuracy (Spearman rho = {pv[pv.dataset == ds].spearman_logprev_vs_f1.iloc[0]:.2f})")
        done += _save(fig, figs / "class_difficulty")
    return done


def composition_recovery_plot(A: dict, figs: Path, ds: str, top: int = 8) -> list[Path]:
    cr, mr = A["composition_recovery"], A["method_ranking"]
    cr = cr[cr.dataset == ds]
    if cr.empty:
        return []
    methods = [m for m in mr[mr.dataset == ds].sort_values("rank").method if m in set(cr.method)][:top]
    n = len(methods)
    cols = min(4, n)
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(3.2 * cols, 3.2 * ((n + cols - 1) // cols)), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, m in zip(axes.ravel(), methods):
        g = cr[cr.method == m]
        ax.scatter(g.true_percentage, g.predicted_percentage, s=14, color=color_for_method(m))
        lim = max(g.true_percentage.max(), g.predicted_percentage.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k:", lw=0.8)
        ax.set_title(m, fontsize=8)
        ax.set_xlabel("true %")
        ax.set_ylabel("predicted %")
    fig.suptitle(f"{ds} - cell-type composition recovery (top {n} methods)", y=1.0)
    fig.tight_layout()
    return _save(fig, figs / "composition_recovery")


def misc_plots(A: dict, figs: Path, ds: str) -> list[Path]:
    done = []
    sub = lambda t: t[t.dataset == ds] if len(t) else t
    lc = A["level_comparison"]
    g = sub(lc).set_index("method").drop(columns="dataset") if len(lc) else lc
    if len(g):
        fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(g) + 2), 4))
        w = 0.8 / g.shape[1]
        for i, c in enumerate(g.columns):
            ax.bar(np.arange(len(g)) + i * w, g[c], w, label=c)
        ax.set_xticks(np.arange(len(g)) + 0.4 - w / 2, g.index, rotation=90, fontsize=7)
        ax.set_ylabel("macro F1")
        ax.set_title(f"{ds} - performance by cell-type granularity")
        ax.legend(frameon=False)
        done += _save(fig, figs / "level_comparison")
    fr = A["final_results"]
    s = fr[(fr.dataset == ds) & (fr.eval_level == "level3")].dropna(subset=["stability"]).sort_values("stability") if len(fr) else fr
    if len(s):
        fig, ax = plt.subplots(figsize=(6, max(3, 0.25 * len(s) + 1)))
        ax.barh(s.method, s.stability, color=[color_for_method(m) for m in s.method])
        ax.set_xlabel("stability = 1 - std(weighted F1)/0.2 across folds")
        ax.set_title(f"{ds} - stability")
        done += _save(fig, figs / "stability")
    fi = sub(A["feature_importance_by_method"])
    if len(fi):
        top = A["feature_importance_consensus"].query("dataset == @ds").sort_values("consensus_rank").marker.head(25)
        p = fi[fi.marker.isin(top)].pivot_table(index="method", columns="marker", values="importance_norm")[list(top)]
        done += heatmap(p, figs / "feature_importance", f"{ds} - marker importance (normalised per method, top consensus markers)",
                        "viridis", cbar="importance share", fmt="{:.2f}")
    tc = sub(A["top_confusions"]).head(15)
    if len(tc):
        fig, ax = plt.subplots(figsize=(6, 0.3 * len(tc) + 1.2))
        ax.barh([f"{a} -> {b}" for a, b in zip(tc.true, tc.predicted)][::-1], tc.total_errors[::-1], color="#e45756")
        ax.set_xlabel("errors summed over methods")
        ax.set_title(f"{ds} - most frequent confusions")
        ax.tick_params(axis="y", labelsize=7)
        done += _save(fig, figs / "top_confusions")
    sg = sub(A["spatial_gain"])
    if len(sg):
        fig, ax = plt.subplots(figsize=(5, 0.4 * len(sg) + 1.2))
        ax.barh([f"{a} vs {b}" for a, b in zip(sg.spatial_method, sg.non_spatial_reference)], sg.gain,
                color=np.where(sg.gain >= 0, "#54a24b", "#e45756"))
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("overall-score gain from spatial information")
        done += _save(fig, figs / "spatial_gain")
    uq = sub(A["unsupervised_qc"])
    uq = uq.set_index("method") if len(uq) else uq
    if len(uq):
        cols = [c for c in ("silhouette", "marker_enrichment", "pseudo_consistency", "knn_consistency", "davies_bouldin") if c in uq]
        fig, axes = plt.subplots(1, len(cols), figsize=(3.4 * len(cols), max(3, 0.25 * len(uq) + 1)), squeeze=False)
        for ax, c in zip(axes[0], cols):
            v = uq[c].dropna().sort_values()
            ax.barh(v.index, v, color=[color_for_method(m) for m in v.index])
            ax.set_title(c + (" (lower = better)" if c == "davies_bouldin" else ""), fontsize=8)
            ax.tick_params(axis="y", labelsize=7)
        done += _save(fig, figs / "unsupervised_qc")
    return done


def confusion_and_maps(root: Path, ds: str, figs: Path) -> list[Path]:
    done = []
    for f in sorted((root / "analysis" / "confusion_matrices" / ds).glob("*.csv")):
        cm = pd.read_csv(f, index_col=0)
        labs = sorted(set(cm.index) | set(cm.columns))
        cm = cm.reindex(index=labs, columns=labs, fill_value=0).astype(float)
        cm = cm.div(cm.sum(1).replace(0, 1), axis=0)
        done += heatmap(cm, figs / "confusion" / f.stem, f"{ds} - {f.stem} (row-normalised)", "Blues", 0, 1, annot=False,
                        cbar="fraction of true class", xlabel="predicted", ylabel="true")
    res = _csv(root / "benchmark_results.csv")
    if len(res):                                              # one spatial map per method, from a saved prediction file
        r = res[(res.dataset == ds) & (res.status == "ok") & res.pred_file.fillna("").astype(str).ne("")]
        order = {"cv": 0, "holdout": 1, "all": 2, "progressive": 3}
        for m, g in r.groupby("method"):
            g = g.assign(o=g.split.map(order)).sort_values(["o", "fold"])
            f = next((Path(x) for x in g.pred_file if Path(x).exists()), None)
            if f is not None:
                done += _spatial_map(f, figs / "spatial" / m, f"{ds} - {m}")
    return done


def _spatial_map(pred_csv: Path, path: Path, title: str) -> list[Path]:
    d = pd.read_csv(pred_csv)
    if not {"x", "y"} <= set(d.columns):
        return []
    if "image" in d:
        d = d[d.image == d.image.value_counts().index[0]]     # largest image
    d = d.assign(reference=d["gt_cell_type"] if "gt_cell_type" in d else d.get("true_phenotype"))
    cols = [c for c in ("reference", "predicted_phenotype") if c in d and d[c].notna().any()]
    labels = sorted(set(np.concatenate([d[c].astype(str) for c in cols])))
    cmap = plt.get_cmap("tab20", max(len(labels), 1))
    cid = {l: cmap(i) for i, l in enumerate(labels)}
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols) + 2, 5), squeeze=False)
    for ax, c in zip(axes[0], cols):
        ax.scatter(d.x, d.y, c=[cid[v] for v in d[c].astype(str)], s=3, linewidths=0)
        ax.set_title(("Reference labels" if c == "reference" else "Predicted") + f" - {title}", fontsize=9)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.axis("off")
    axes[0][-1].legend(handles=[plt.Line2D([], [], marker="o", ls="", color=cid[l], label=l) for l in labels], frameon=False,
                       fontsize=6, bbox_to_anchor=(1.01, 1), loc="upper left", ncol=1 + len(labels) // 25)
    return _save(fig, path)


def make_all(out: str | Path) -> list[Path]:
    """Create every figure the saved results allow; each section is independent and failures are reported."""
    import logging
    root = Path(out)
    a = root / "analysis"
    A = {p.stem: _csv(p) for p in a.glob("*.csv")}
    for k in ("method_ranking", "final_results", "efficiency", "data_scarcity", "per_class_by_method", "class_difficulty",
              "prevalence_vs_f1", "composition_recovery", "level_comparison", "feature_importance_by_method",
              "feature_importance_consensus", "top_confusions", "spatial_gain", "unsupervised_qc"):
        A.setdefault(k, pd.DataFrame())
    done: list[Path] = []

    def guard(label, fn, *args):
        try:
            done.extend(fn(*args))
        except Exception:
            logging.getLogger("spcelleval").exception("figure '%s' failed", label)

    dsets = sorted(A["final_results"].dataset.unique()) if len(A["final_results"]) else \
        sorted(p.name for p in root.iterdir() if (p / "dataset_report").is_dir())
    for ds in dsets:
        figs = root / ds / "figures"
        guard(f"{ds}/composition", composition_plots, root / ds / "dataset_report", figs, ds)
        guard(f"{ds}/misc", misc_plots, A, figs, ds)
        if len(A["final_results"]):
            guard(f"{ds}/ranking", ranking_plots, A, figs, ds)
            guard(f"{ds}/classes", class_plots, A, figs, ds) if len(A["per_class_by_method"]) else None
            guard(f"{ds}/composition_recovery", composition_recovery_plot, A, figs, ds) if len(A["composition_recovery"]) else None
            guard(f"{ds}/confusion+maps", confusion_and_maps, root, ds, figs)
    if len(A["final_results"]):
        guard("overall_heatmap", overall_heatmap, A, root / "figures")
        guard("efficiency", efficiency_plots, A, root / "figures")
    guard("scaling", scaling_curves, A, root / "figures")
    mc = A.get("metric_correlation", pd.DataFrame())
    if len(mc):
        guard("metric_correlation", lambda: heatmap(mc.set_index("metric"), root / "figures" / "metric_correlation",
                                                    "Correlation of metrics", "RdBu_r", -1, 1, cbar="Pearson r"))
    return done

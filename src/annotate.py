"""Cluster-first annotation of a NEW dataset with no ground truth (``python cli.py annotate``).

    over-cluster (Leiden, high resolution) -> match each micro-cluster to the signature matrix -> confidence / 'Ambiguous'
    -> review export (CSV an expert edits, heatmap, spatial overlay) -> [--review] -> propagate with a supervised model
    -> QC that needs no labels.

Reuses ``models/marker.py`` (conform_matrix, add_nested_negatives, align, marker_scores), ``models/traditional._louvain_labels``
(fallback when scanpy is missing), ``evaluation/metrics`` (unsupervised_qc, spatial_metrics) and ``evaluation/plots`` (heatmap,
_spatial_map). The repo's run_leiden_clustering.py is NOT used: it needs ground-truth labels (greedy cluster->label mapping).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .models.marker import add_nested_negatives, align, conform_matrix, load_marker_matrix, marker_scores
from .preprocessing import load_dataset, transform

log = logging.getLogger("spcelleval")
AMBIGUOUS = "Ambiguous"


# ----------------------------------------------------------------------------- 1. over-cluster
def over_cluster(Z: np.ndarray, resolution: float, k: int, seed: int) -> np.ndarray:
    """Leiden on a kNN graph of the z-scored markers (scanpy / igraph); Louvain from the repo's own helper if scanpy is missing.
    Cluster ids are renumbered 0.. by decreasing size so they are stable between runs."""
    t0 = time.perf_counter()
    try:
        import anndata as ad
        import scanpy as sc
        a = ad.AnnData(Z.astype(np.float32))
        sc.pp.neighbors(a, n_neighbors=min(k, len(Z) - 1), use_rep="X", random_state=seed)
        try:
            sc.tl.leiden(a, resolution=resolution, key_added="c", random_state=seed, flavor="igraph", n_iterations=2, directed=False)
        except (TypeError, ImportError, ValueError):
            sc.tl.leiden(a, resolution=resolution, key_added="c", random_state=seed)
        raw, how = a.obs["c"].astype(int).to_numpy(), "scanpy Leiden"
    except ImportError:
        from sklearn.neighbors import kneighbors_graph
        from .models.traditional import _louvain_labels
        A = kneighbors_graph(Z, min(k, len(Z) - 1), include_self=False, n_jobs=-1)
        raw, how = _louvain_labels(A, resolution, seed), "Louvain fallback (scanpy not installed)"
    order = pd.Series(raw).value_counts().index
    lab = pd.Series(raw).map({c: i for i, c in enumerate(order)}).to_numpy()
    log.info("over-clustering: %s, resolution %g, k=%d -> %d micro-clusters in %.1fs (smallest %d cells, largest %d)",
             how, resolution, k, lab.max() + 1, time.perf_counter() - t0, np.bincount(lab).min(), np.bincount(lab).max())
    return lab


# ----------------------------------------------------------------------------- 2. match to the signature
def match_clusters(X: np.ndarray, Zdf: pd.DataFrame, cl: np.ndarray, matrix: pd.DataFrame, markers: list[str],
                   min_score: float, min_margin: float, min_support: float, top_n: int = 5):
    """One row per micro-cluster. Score = mean over its cells of the repo's signed marker z-score (marker_scores) per type;
    confidence = margin over the runner-up, discounted when the best score is low or a signature marker is not above average
    (weakest median z of the assigned type's positive markers < ``min_support``). Returns (table, types, cell_scores)."""
    types, M = align(matrix, markers)
    S = marker_scores(X, M)                                                         # cells x types
    sc = pd.DataFrame(S, columns=types).groupby(cl).mean()
    med = Zdf.groupby(cl).median()
    cell_best = np.array(types, dtype=object)[S.argmax(1)]
    pos = {t: [markers[j] for j in np.where(M[i] > 0)[0]] for i, t in enumerate(types)}
    rows = []
    for c in sc.index:
        s = sc.loc[c].sort_values(ascending=False)
        best, run = s.index[0], (s.index[1] if len(s) > 1 else "")
        margin = float(s.iloc[0] - (s.iloc[1] if len(s) > 1 else 0.0))
        weakest = float(med.loc[c, pos[best]].min()) if pos[best] else float("nan")
        support_ok, score_ok, margin_ok = bool(weakest >= min_support), bool(s.iloc[0] >= min_score), bool(margin >= min_margin)
        conf = float(np.clip(margin / (2 * min_margin), 0, 1)) * (1.0 if support_ok else 0.5) * (1.0 if score_ok else 0.5)
        m = cl == c
        ok = support_ok and score_ok and margin_ok
        rows.append(dict(cluster=int(c), n_cells=int(m.sum()), assigned_type=best if ok else AMBIGUOUS, best_guess=best,
                         confidence=round(conf, 4), score=round(float(s.iloc[0]), 4), margin=round(margin, 4), runner_up=run,
                         runner_up_score=round(float(s.iloc[1]), 4) if len(s) > 1 else float("nan"),
                         min_signature_marker_z=round(weakest, 4), weakest_marker=(med.loc[c, pos[best]].idxmin() if pos[best] else ""),
                         signature_purity=round(float((cell_best[m] == best).mean()), 4),
                         top_markers=",".join(med.loc[c].sort_values(ascending=False).index[:top_n]),
                         status="confident" if ok else AMBIGUOUS,
                         why_ambiguous=";".join(n for n, good in (("low_margin", margin_ok), ("low_score", score_ok),
                                                                   ("weak_signature_marker", support_ok)) if not good)))
    t = pd.DataFrame(rows)
    log.info("matched %d clusters to %d signature types: %d confident (%d cells), %d ambiguous (%d cells)", len(t), len(types),
             (t.status == "confident").sum(), t.loc[t.status == "confident", "n_cells"].sum(),
             (t.status == AMBIGUOUS).sum(), t.loc[t.status == AMBIGUOUS, "n_cells"].sum())
    for r in t[t.status == AMBIGUOUS].itertuples():
        log.info("  ambiguous cluster %d (%d cells): best guess %s vs %s, margin %.3f, score %.3f, weakest marker %s z=%.2f [%s]",
                 r.cluster, r.n_cells, r.best_guess, r.runner_up, r.margin, r.score, r.weakest_marker, r.min_signature_marker_z, r.why_ambiguous)
    return t, types, S, med


# ----------------------------------------------------------------------------- 3. review export / import
def export_review(table: pd.DataFrame, med: pd.DataFrame, ds, cl: np.ndarray, out: Path) -> Path:
    """review_table.csv (fill the empty ``expert_label`` column, or edit a copy to just ``cluster,label``), a cluster x marker median
    heatmap and a spatial overlay of the ambiguous clusters. Plot failures are logged, never fatal."""
    (out / "figures").mkdir(parents=True, exist_ok=True)
    review = out / "review_table.csv"
    table.assign(expert_label="").to_csv(review, index=False)
    try:
        from .evaluation.plots import _spatial_map, heatmap
        order = table.sort_values(["assigned_type", "cluster"]).cluster
        h = med.loc[order]
        h.index = [f"{c} | {table.set_index('cluster').assigned_type[c]}" for c in order]
        heatmap(h, out / "figures" / "cluster_marker_medians", "Median z-scored marker per micro-cluster (rows: id | assigned type)",
                cmap="RdBu_r", vmin=-2, vmax=2, cbar="median z", annot=False)
        if ds.xy is not None:
            amb = set(table.loc[table.status == AMBIGUOUS, "cluster"])
            ov = pd.DataFrame({"x": ds.xy[:, 0], "y": ds.xy[:, 1],
                               "predicted_phenotype": [f"cluster {c}" if c in amb else "confident" for c in cl]})
            if ds.groups is not None:
                ov["image"] = ds.groups
            (out / "tables").mkdir(exist_ok=True)
            ov.to_csv(out / "tables" / "ambiguous_overlay.csv", index=False)
            _spatial_map(out / "tables" / "ambiguous_overlay.csv", out / "figures" / "ambiguous_spatial", f"{ds.name}: ambiguous clusters (largest image)")
        else:
            log.warning("no x/y columns: spatial overlay of ambiguous clusters skipped")
    except Exception:
        log.exception("review figures failed (the review table was still written)")
    log.info("review table: %s  (edit expert_label, then re-run with --review %s)", review, review)
    return review


def read_review(path: str | Path, clusters: set[int]) -> dict[int, str]:
    """cluster -> label from a CSV with a ``cluster`` column and a ``label`` or ``expert_label`` column. Blank labels are ignored;
    'Ambiguous' sends the cluster to the supervised model."""
    df = pd.read_csv(path)
    col = next((c for c in ("label", "expert_label") if c in df.columns), None)
    if "cluster" not in df.columns or col is None:
        raise ValueError(f"{path}: needs a 'cluster' column and a 'label' (or 'expert_label') column")
    df = df.dropna(subset=[col])
    df = df[df[col].astype(str).str.strip() != ""]
    unknown = sorted(set(df.cluster.astype(int)) - clusters)
    if unknown:
        raise ValueError(f"{path}: cluster id(s) {unknown[:10]} do not exist in this clustering")
    labels = dict(zip(df.cluster.astype(int), df[col].astype(str).str.strip()))
    log.info("expert review: %d cluster(s) relabelled from %s", len(labels), path)
    return labels


# ----------------------------------------------------------------------------- 4. propagate
def propagate(Z: np.ndarray, label: np.ndarray, known: np.ndarray, seed: int, n_jobs: int, max_train: int = 200_000):
    """Train on the cells whose label is trusted (``known``), predict the rest. XGBoost (the repo's classic-ML model), random forest
    if xgboost is not installed. Returns (labels of every cell, confidence = max class probability for the predicted cells)."""
    from sklearn.preprocessing import LabelEncoder
    out, conf = label.copy(), np.ones(len(label))
    tr, te = np.where(known)[0], np.where(~known)[0]
    if len(te) == 0:
        log.info("propagation: nothing to predict (no ambiguous cells)")
        return out, conf
    if len(tr) > max_train:
        tr = np.sort(np.random.default_rng(seed).choice(tr, max_train, replace=False))
    le = LabelEncoder().fit(label[tr])
    if len(le.classes_) == 1:
        out[te], conf[te] = le.classes_[0], 1.0
        log.warning("propagation: only one trusted class (%s); assigned to all %d remaining cells", le.classes_[0], len(te))
        return out, conf
    try:
        from xgboost import XGBClassifier
        model, name = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist", n_jobs=n_jobs,
                                    random_state=seed, verbosity=0), "XGBoost"
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        model, name = RandomForestClassifier(n_estimators=200, n_jobs=n_jobs, random_state=seed, class_weight="balanced"), "random forest (xgboost missing)"
    t0 = time.perf_counter()
    model.fit(Z[tr], le.transform(label[tr]))
    P = model.predict_proba(Z[te])
    out[te], conf[te] = le.inverse_transform(P.argmax(1)), P.max(1)
    log.info("propagation: %s trained on %d trusted cells (%d classes), predicted %d cells in %.1fs; mean confidence %.3f, %d below 0.5",
             name, len(tr), len(le.classes_), len(te), time.perf_counter() - t0, conf[te].mean(), int((conf[te] < 0.5).sum()))
    return out, conf


# ----------------------------------------------------------------------------- 5. QC without ground truth
def qc_report(Z: np.ndarray, final: np.ndarray, cl: np.ndarray, table: pd.DataFrame, ds, matrix: pd.DataFrame, source: np.ndarray,
              k: int, out: Path, seed: int) -> dict:
    from .evaluation.metrics import spatial_metrics, unsupervised_qc
    q = out / "qc"
    q.mkdir(parents=True, exist_ok=True)
    w = table.n_cells / table.n_cells.sum()
    res = dict(n_cells=int(len(final)), n_clusters=int(len(table)), n_types=int(len(set(final))),
               share_cells_confident_clusters=float(w[table.status == "confident"].sum()),
               share_cells_propagated=float((source == "propagated").mean()),
               cluster_purity_vs_signature=float((table.signature_purity * w).sum()),
               cluster_purity_vs_signature_confident=float(np.average(table.signature_purity[table.status == "confident"],
                                                                      weights=table.n_cells[table.status == "confident"]))
               if (table.status == "confident").any() else float("nan"))
    res.update(unsupervised_qc(Z, final, ds.markers, matrix, seed=seed, k=k))                 # silhouette, marker_enrichment, knn_consistency
    sp = spatial_metrics(final, final, ds.xy, ds.groups, k, max_n=10**8)                      # ground-truth-free: homophily of OUR labels
    res["spatial_homophily"] = sp["spatial_homophily_pred"]
    if ds.xy is None:
        log.warning("QC: no x/y columns, spatial coherence not computed")
    comp = pd.crosstab(ds.groups if ds.groups is not None else np.repeat("all", len(final)), final, normalize="index")
    comp.index.name = "image"
    comp.to_csv(q / "composition_per_image.csv")
    pd.Series(final).value_counts().rename_axis("cell_type").rename("n_cells").to_csv(q / "composition_overall.csv")
    try:
        from .evaluation.plots import heatmap
        heatmap(comp, out / "figures" / "composition_per_image", "Cell-type composition per image (fraction)", cmap="Blues", cbar="fraction", annot=comp.size <= 400)
    except Exception:
        log.exception("composition figure failed")
    (q / "qc_summary.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    for key, v in res.items():
        log.info("QC %-40s %s", key, f"{v:.4f}" if isinstance(v, float) else v)
    return res


# ----------------------------------------------------------------------------- driver
def annotate(data: str, matrix_path: str, out: str | Path, resolution: float = 10.0, k: int = 15, review: str | None = None,
             min_score: float = 0.5, min_margin: float = 0.25, min_support: float = 0.0, transform_method: str = "auto",
             cofactor: float = 5.0, modality: str = "codex", seed: int = 0, n_jobs: int = 4, max_cells: int = 0,
             spatial_k: int = 10) -> pd.DataFrame:
    out = Path(out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ds = load_dataset(data, modality=modality)
    if ds.y is not None:
        log.warning("the table has a label column; it is ignored (annotate never reads labels)")
    ds = transform(ds.subsample(max_cells, seed), transform_method, cofactor)
    X = ds.X.to_numpy(float)
    Z = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    Zdf = pd.DataFrame(Z, columns=ds.markers)
    log.info("annotate %s: %d cells x %d markers, x/y %s, image column %s", ds.name, len(ds), len(ds.markers),
             ds.xy is not None, ds.groups is not None)

    m, dropped = conform_matrix(load_marker_matrix(matrix_path), ds.markers)
    if m.shape[1] == 0:
        raise ValueError("no column of the signature matrix matches the dataset's marker names")
    if dropped:
        log.warning("signature columns with no matching marker were ignored: %s", ", ".join(dropped))
    m = add_nested_negatives(m)
    log.info("signature: %d types x %d matched markers", len(m), m.shape[1])

    cache = out / "tables" / "clusters.csv"
    if review and cache.exists() and len(pd.read_csv(cache)) == len(ds):          # a review refers to the cluster ids of the first run
        cl = pd.read_csv(cache).cluster.to_numpy()
        log.info("re-using the clustering saved in %s (%d clusters), so the review's cluster ids stay valid", cache, cl.max() + 1)
    else:
        cl = over_cluster(Z, resolution, k, seed)
        pd.DataFrame({"cluster": cl}).to_csv(cache, index=False)

    table, types, S, med = match_clusters(X, Zdf, cl, m, ds.markers, min_score, min_margin, min_support)
    export_review(table, med, ds, cl, out)

    label = pd.Series(cl).map(table.set_index("cluster").assigned_type).to_numpy(dtype=object)     # 'Ambiguous' or the type
    conf = pd.Series(cl).map(table.set_index("cluster").confidence).to_numpy(float)
    source = np.where(label == AMBIGUOUS, "ambiguous", "cluster").astype(object)
    if review:
        for c, lab in read_review(review, set(table.cluster)).items():
            msk = cl == c
            label[msk] = AMBIGUOUS if lab.lower() == AMBIGUOUS.lower() else lab
            source[msk] = "ambiguous" if lab.lower() == AMBIGUOUS.lower() else "expert"
            conf[msk] = 0.0 if lab.lower() == AMBIGUOUS.lower() else 1.0
    known = label != AMBIGUOUS
    if not known.any():
        raise ValueError("no confident or reviewed cluster to train on: relax --min-margin / --min-score or label clusters in the review file")

    final, pconf = propagate(Z, label, known, seed, n_jobs)
    source = np.where(source == "ambiguous", "propagated", source)
    conf = np.where(known, conf, pconf)

    pred = pd.DataFrame({"cell_index": np.arange(len(ds)), "cluster": cl, "final_label": final, "label_source": source,
                         "confidence": np.round(conf, 4)})
    for c in ("cell_id", "image", "x", "y"):
        if c in ds.meta.columns:
            pred.insert(1, c, ds.meta[c].to_numpy())
    pred.to_csv(out / "annotations.csv", index=False)
    log.info("wrote %s: %d cells; label sources %s", out / "annotations.csv", len(pred), pred.label_source.value_counts().to_dict())

    qc_report(Z, final.astype(str), cl, table, ds, m, source, spatial_k, out, seed)
    log.info("annotate finished in %.1fs; outputs in %s", time.perf_counter() - t0, out)
    return pred

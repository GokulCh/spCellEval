"""Supervised metrics and ground-truth-free QC metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (adjusted_rand_score, cohen_kappa_score, confusion_matrix, davies_bouldin_score,
                             f1_score, matthews_corrcoef, normalized_mutual_info_score, silhouette_score)
from sklearn.neighbors import NearestNeighbors

from ..models.base import MethodUnavailable
from ..models.marker import align, marker_scores
from .repo_assets import (ancestors, calculate_cell_type_distribution, calculate_r2_and_pearson, gmean_score,
                          hierarchical_f1_score)

NAN = float("nan")


def composition_table(y_true, y_pred) -> pd.DataFrame:
    """True vs predicted cell-type percentages: the benchmark notebook's own function."""
    return calculate_cell_type_distribution(None, pd.Series(np.asarray(y_pred).astype(str)),
                                            pd.Series(np.asarray(y_true).astype(str)))


def supervised_metrics(y_true, y_pred, rare_frac: float = 0.01, level: str = "level3") -> dict:
    """All classification + composition metrics used by the benchmark notebooks, plus sensitivity,
    specificity and rare/abundant accuracy.

    f1_macro / f1_weighted / mcc / kappa / ari / nmi / g_mean / r2 / pearson / jsd(_scaled) /
    hierarchical_f1 follow ``eval_mapping.ipynb`` (macro F1 is sklearn's, over true+predicted labels).
    A class is *rare* when it is < ``rare_frac`` of the evaluated cells; rare/abundant accuracy is the
    pooled recall over those classes' cells.
    """
    y_true, y_pred = np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm).astype(float)
    sup, pred = cm.sum(1), cm.sum(0)
    fn, fp = sup - tp, pred - tp
    tn = cm.sum() - tp - fn - fp
    present = sup > 0
    rare = present & (sup / cm.sum() < rare_frac)
    abund = present & ~rare
    acc = lambda m: tp[m].sum() / sup[m].sum() if m.any() else NAN
    comp = composition_table(y_true, y_pred)
    r2, pear = calculate_r2_and_pearson(comp) if len(comp) > 1 else (NAN, NAN)
    jsd = float(jensenshannon(comp.predicted_percentage / 100, comp.true_percentage / 100, base=2))
    hier = NAN
    if level == "level3" and np.isin(y_true, list(ancestors())).any():
        hier = hierarchical_f1_score(y_true, y_pred, ancestors())
    return dict(
        accuracy=tp.sum() / cm.sum(),
        f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
        f1_weighted=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        hierarchical_f1=hier,
        mcc=matthews_corrcoef(y_true, y_pred), kappa=cohen_kappa_score(y_true, y_pred),
        ari=adjusted_rand_score(y_true, y_pred), nmi=normalized_mutual_info_score(y_true, y_pred),
        g_mean=gmean_score(y_true, y_pred),
        sensitivity_macro=(tp / np.maximum(sup, 1))[present].mean(),
        specificity_macro=(tn / np.maximum(tn + fp, 1))[present].mean(),
        rare_accuracy=acc(rare), abundant_accuracy=acc(abund), n_rare_classes=int(rare.sum()),
        r2=r2, pearson=pear, jsd=jsd, jsd_scaled=1 - jsd,
    )


def per_class_table(y_true, y_pred) -> pd.DataFrame:
    """One row per class: support, prevalence, precision, recall (sensitivity), specificity, F1."""
    y_true, y_pred = np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm).astype(float)
    sup, pred = cm.sum(1), cm.sum(0)
    fn, fp = sup - tp, pred - tp
    tn = cm.sum() - tp - fn - fp
    return pd.DataFrame(dict(
        cell_type=labels, support=sup, n_predicted=pred, prevalence=sup / cm.sum(),
        precision=tp / np.maximum(pred, 1), recall=tp / np.maximum(sup, 1),
        specificity=tn / np.maximum(tn + fp, 1), f1=2 * tp / np.maximum(2 * tp + fp + fn, 1)))


def top_confusions(y_true, y_pred, n: int = 15) -> pd.DataFrame:
    """Most frequent (true -> predicted) mistakes with the share of the true class they represent."""
    d = pd.DataFrame(dict(true=np.asarray(y_true).astype(str), predicted=np.asarray(y_pred).astype(str)))
    tot = d.true.value_counts()
    c = d[d.true != d.predicted].value_counts().reset_index(name="count").head(n)
    c["fraction_of_true_class"] = c["count"] / c["true"].map(tot)
    return c


def unsupervised_qc(X, labels, markers, matrix=None, pseudo=None, pseudo_ok=None, seed: int = 0,
                    max_n: int = 5000, k: int = 10) -> dict:
    """Metrics that need no ground truth.

    silhouette / davies_bouldin: cluster separation in marker space (subsampled to ``max_n``).
    marker_enrichment: mean (score of assigned type - mean score of other types) under the marker matrix.
    pseudo_consistency: agreement with confident marker pseudo-labels.
    knn_consistency: mean fraction of feature-space neighbours sharing a cell's label.
    """
    labels = np.asarray(labels).astype(str)
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(len(X), min(max_n, len(X)), replace=False))
    Xs, ls = X[sub], labels[sub]
    n_lab = len(np.unique(ls))
    out = dict(n_predicted_types=len(np.unique(labels)), silhouette=NAN, davies_bouldin=NAN,
               knn_consistency=NAN, marker_enrichment=NAN, pseudo_consistency=NAN)
    if 2 <= n_lab < len(ls):
        out["silhouette"] = float(silhouette_score(Xs, ls))
        out["davies_bouldin"] = float(davies_bouldin_score(Xs, ls))
    if len(ls) > k + 1:
        nb = NearestNeighbors(n_neighbors=k + 1).fit(Xs).kneighbors(Xs, return_distance=False)[:, 1:]
        out["knn_consistency"] = float((ls[nb] == ls[:, None]).mean())
    if matrix is not None:
        try:
            types, M = align(matrix, markers)
            S = marker_scores(X, M)
            pos = {t: i for i, t in enumerate(types)}
            has = np.array([l in pos for l in labels])
            if has.any():
                idx = np.array([pos[l] for l in labels[has]])
                Sh = S[has]
                own = Sh[np.arange(len(idx)), idx]
                others = (Sh.sum(1) - own) / max(S.shape[1] - 1, 1)
                out["marker_enrichment"] = float((own - others).mean())
        except MethodUnavailable:
            pass
    if pseudo is not None and pseudo_ok is not None and pseudo_ok.any():
        out["pseudo_consistency"] = float((labels[pseudo_ok] == np.asarray(pseudo)[pseudo_ok]).mean())
    return out

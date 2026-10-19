"""Tier 4 - spatial / graph-aware methods and spatial post-processing."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse import csr_matrix, identity
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.exceptions import ConvergenceWarning

from .base import Result, Task, register
from .params import of

warnings.filterwarnings("ignore", category=ConvergenceWarning)
from .traditional import _need_labels


def spatial_graph(xy: np.ndarray, groups: np.ndarray | None, k: int) -> csr_matrix:
    """Binary kNN adjacency (no self loops), built within each image/sample so cells never link across images."""
    n = len(xy)
    groups = np.zeros(n, int) if groups is None else groups
    rows, cols = [], []
    for g in np.unique(groups):
        ix = np.where(groups == g)[0]
        if len(ix) < 2:
            continue
        kk = min(k, len(ix) - 1)
        nb = NearestNeighbors(n_neighbors=kk + 1).fit(xy[ix]).kneighbors(xy[ix], return_distance=False)[:, 1:]
        rows.append(np.repeat(ix, kk))
        cols.append(ix[nb].ravel())
    r, c = (np.concatenate(rows), np.concatenate(cols)) if rows else (np.array([], int),) * 2
    return csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))


def _row_norm(A: csr_matrix) -> csr_matrix:
    d = np.asarray(A.sum(1)).ravel()
    d[d == 0] = 1
    return csr_matrix(A.multiply(1 / d[:, None]))


def spatial_vote(labels: np.ndarray, xy: np.ndarray, groups: np.ndarray | None, k: int) -> np.ndarray:
    """Replace each label by the majority among the cell itself and its k spatial neighbours."""
    labels = np.asarray(labels).astype(str)
    classes, inv = np.unique(labels, return_inverse=True)
    onehot = csr_matrix((np.ones(len(inv)), (np.arange(len(inv)), inv)), shape=(len(inv), len(classes)))
    counts = (spatial_graph(xy, groups, k) + identity(len(inv), format="csr")) @ onehot
    return classes[np.asarray(counts.todense()).argmax(1)]


@register("spatial_gnn", 4, "supervised", ("sklearn", "scipy"), needs_xy=True)
def spatial_gnn(t: Task) -> Result:
    """Simplified graph convolution (SGC): [X, AX, ..., A^hops X] over the spatial kNN graph + logistic regression.

    Neighbour *features* of train and test cells are used at inference; neighbour labels never are.
    """
    y, p = _need_labels(t), of("spatial_gnn", t.params)
    A = _row_norm(spatial_graph(t.xy, t.groups, t.k_neighbors))
    hops = [t.X]
    for _ in range(p["hops"]):
        hops.append(A @ hops[-1])
    F = np.hstack(hops)
    if p["model"] == "hgb":      # non-linear gates (CD45+ AND CD3+ AND NOT CD4) that a linear model cannot express
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.utils.class_weight import compute_sample_weight
        m = HistGradientBoostingClassifier(max_iter=p["max_iter"], learning_rate=0.1, early_stopping=True, random_state=t.seed)
        m.fit(F[t.train_idx], y, sample_weight=compute_sample_weight("balanced", y))
        return Result(m.predict(F[t.test_idx]))
    m = make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=p["max_iter"], class_weight="balanced",
                                                           random_state=t.seed))
    m.fit(F[t.train_idx], y)
    imp = np.abs(m[-1].coef_).mean(0).reshape(p["hops"] + 1, t.X.shape[1]).sum(0)
    return Result(m.predict(F[t.test_idx]), imp)


@register("knn_smooth", 4, "supervised", ("sklearn", "scipy"), needs_xy=True)
def knn_smooth(t: Task) -> Result:
    """Spatial neighbourhood smoothing of marker expression (0.5*self + 0.5*neighbour mean) + random forest."""
    y, p = _need_labels(t), of("knn_smooth", t.params)
    A = _row_norm(spatial_graph(t.xy, t.groups, t.k_neighbors))
    NB = A @ t.X
    # concat: own profile + neighbourhood mean (mixing them 50/50 blurs rare cells that sit among other types: only ~38% of a CRC cell's neighbours share its type)
    Xs = np.hstack([t.X, NB]) if p["concat"] else (1 - p["alpha"]) * t.X + p["alpha"] * NB
    m = RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"], class_weight="balanced", n_jobs=t.n_jobs, random_state=t.seed)
    m.fit(Xs[t.train_idx], y)
    imp = m.feature_importances_
    return Result(m.predict(Xs[t.test_idx]), imp.reshape(-1, t.X.shape[1]).sum(0))

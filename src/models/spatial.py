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
    """Simplified graph convolution (SGC): [X, AX, A^2 X] over the spatial kNN graph + logistic regression.

    Neighbour *features* of train and test cells are used at inference; neighbour labels never are.
    """
    y = _need_labels(t)
    A = _row_norm(spatial_graph(t.xy, t.groups, t.k_neighbors))
    AX = A @ t.X
    F = np.hstack([t.X, AX, A @ AX])
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced",
                                                           random_state=t.seed))
    m.fit(F[t.train_idx], y)
    p = t.X.shape[1]
    imp = np.abs(m[-1].coef_).mean(0).reshape(3, p).sum(0)
    return Result(m.predict(F[t.test_idx]), imp)


@register("knn_smooth", 4, "supervised", ("sklearn", "scipy"), needs_xy=True)
def knn_smooth(t: Task) -> Result:
    """Spatial neighbourhood smoothing of marker expression (0.5*self + 0.5*neighbour mean) + random forest."""
    y = _need_labels(t)
    A = _row_norm(spatial_graph(t.xy, t.groups, t.k_neighbors))
    Xs = 0.5 * t.X + 0.5 * (A @ t.X)
    m = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=t.n_jobs, random_state=t.seed)
    m.fit(Xs[t.train_idx], y)
    return Result(m.predict(Xs[t.test_idx]), m.feature_importances_)

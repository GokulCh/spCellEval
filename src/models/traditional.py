"""Tier 1 methods that have NO counterpart in the repository: SVM, Louvain and SPADE.

Logistic regression / random forest / XGBoost / baselines run the repo's ``run_classic_ml_default.py`` and Leiden /
FlowSOM run its ``run_leiden_clustering.py`` / ``run_flowsom.R`` (see ``scripts.py``) - they are not re-implemented here.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin
from sklearn.neighbors import kneighbors_graph
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from .base import MethodUnavailable, Result, Task, register


def _need_labels(t: Task) -> np.ndarray:
    if t.y_train is None or len(np.unique(t.y_train)) < 2:
        raise MethodUnavailable("needs >= 2 training classes (no labels / pseudo-labels available)")
    return t.y_train


def _coef_importance(model, n_feat: int):
    c = np.abs(getattr(model, "coef_", np.zeros((1, n_feat))))
    return c.mean(0)


@register("svm", 1, "supervised", ("sklearn",))
def svm(t: Task) -> Result:
    """RBF SVM up to 20k training cells, LinearSVC beyond (SVC is O(n^2))."""
    y = _need_labels(t)
    big = len(y) > 20_000
    clf = LinearSVC(class_weight="balanced", random_state=t.seed) if big else \
        SVC(kernel="rbf", class_weight="balanced", random_state=t.seed)
    m = make_pipeline(StandardScaler(), clf)
    m.fit(t.X_train, y)
    return Result(m.predict(t.X_test), _coef_importance(clf, t.X.shape[1]) if big else None)


def _k(t: Task) -> int:
    return int(t.n_clusters or (len(np.unique(t.y_train)) if t.y_train is not None else 20))


def _cid(a) -> np.ndarray:
    return np.array([f"c{i}" for i in a])


@register("louvain", 1, "cluster", ("networkx",))
def louvain(t: Task) -> Result:
    """Louvain communities on a kNN graph (networkx)."""
    import networkx as nx
    g = nx.from_scipy_sparse_array(kneighbors_graph(t.X_test, t.k_neighbors, include_self=False))
    comm = nx.community.louvain_communities(g, resolution=t.params.get("resolution", 1.0), seed=t.seed)
    lab = np.zeros(len(t.X_test), int)
    for i, c in enumerate(comm):
        lab[list(c)] = i
    return Result(_cid(lab))


@register("spade", 1, "cluster", ("sklearn",))
def spade(t: Task) -> Result:
    """SPADE-style: downsample (<=5k, uniform), agglomerative cluster, up-sample by nearest centroid."""
    X, rng = t.X_test, np.random.default_rng(t.seed)
    sub = rng.choice(len(X), min(5000, len(X)), replace=False)   # ponytail: uniform, SPADE is density-dependent
    lab = AgglomerativeClustering(n_clusters=min(_k(t), len(sub)), linkage="ward").fit_predict(X[sub])
    cents = np.stack([X[sub][lab == c].mean(0) for c in np.unique(lab)])
    return Result(_cid(pairwise_distances_argmin(X, cents)))

"""Tier 1 methods that have NO counterpart in the repository: SVM, Louvain and SPADE.

Logistic regression / random forest / XGBoost / baselines run the repo's ``run_classic_ml_default.py`` and Leiden /
FlowSOM run its ``run_leiden_clustering.py`` / ``run_flowsom.R`` (see ``scripts.py``) - they are not re-implemented here.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import pairwise_distances_argmin
from sklearn.neighbors import kneighbors_graph
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from .base import MethodUnavailable, Result, Task, register
from .params import of


def _need_labels(t: Task) -> np.ndarray:
    if t.y_train is None or len(np.unique(t.y_train)) < 2:
        raise MethodUnavailable("needs >= 2 training classes (no labels / pseudo-labels available)")
    return t.y_train


def _coef_importance(model, n_feat: int):
    c = np.abs(getattr(model, "coef_", np.zeros((1, n_feat))))
    return c.mean(0)


@register("svm", 1, "supervised", ("sklearn",))
def svm(t: Task) -> Result:
    """SVM: exact RBF up to max_rbf_cells training cells; beyond that an RBF kernel approximation (Nystroem) + LinearSVC."""
    y, p = _need_labels(t), of("svm", t.params)
    n_feat = t.X.shape[1]
    lin = dict(C=p["C"], class_weight=p["class_weight"], dual=False, max_iter=2000, random_state=t.seed)   # dual=False: n_samples >> n_features
    if p["kernel"] == "linear":
        clf = LinearSVC(**lin)
    elif len(y) > p["max_rbf_cells"]:              # SVC is O(n^2); the approximation keeps the RBF decision surface at linear cost
        var = 1.0 if p["scale"] else float(t.X_train.var())          # standardised markers have unit variance
        g = p["gamma"] if isinstance(p["gamma"], (int, float)) else 1.0 / (n_feat * (var if p["gamma"] == "scale" else 1.0))   # sklearn's scale / auto
        clf = make_pipeline(Nystroem(gamma=g, n_components=min(500, len(y)), random_state=t.seed), LinearSVC(**lin))
    else:
        clf = SVC(C=p["C"], kernel=p["kernel"], gamma=p["gamma"], class_weight=p["class_weight"], random_state=t.seed)
    m = make_pipeline(StandardScaler(), clf) if p["scale"] else make_pipeline(clf)
    m.fit(t.X_train, y)
    return Result(m.predict(t.X_test), _coef_importance(clf, n_feat) if isinstance(clf, LinearSVC) else None)


def _k(t: Task) -> int:
    return int(t.n_clusters or (len(np.unique(t.y_train)) if t.y_train is not None else 20))


def _cid(a) -> np.ndarray:
    return np.array([f"c{i}" for i in a])


def _louvain_labels(A, resolution: float, seed: int) -> np.ndarray:
    """Louvain communities of a sparse graph: igraph's C implementation when installed (leidenalg needs it, so usually),
    else networkx's pure-Python one (much slower on 10^5+ cells)."""
    try:
        import igraph as ig
    except ImportError:
        import networkx as nx
        comm = nx.community.louvain_communities(nx.from_scipy_sparse_array(A), resolution=resolution, seed=seed)
        lab = np.zeros(A.shape[0], int)
        for i, c in enumerate(comm):
            lab[list(c)] = i
        return lab
    import random
    random.seed(seed)
    ig.set_random_number_generator(random)
    U = A.maximum(A.T).tocoo()                      # symmetric, each edge once
    keep = U.row < U.col
    g = ig.Graph(n=A.shape[0], edges=list(zip(U.row[keep].tolist(), U.col[keep].tolist())))
    return np.asarray(g.community_multilevel(resolution=resolution).membership)


@register("louvain", 1, "cluster", ("sklearn",))
def louvain(t: Task) -> Result:
    """Louvain communities on a kNN graph (igraph if available, else networkx)."""
    A = kneighbors_graph(t.X_test, t.k_neighbors, include_self=False, n_jobs=t.n_jobs)
    return Result(_cid(_louvain_labels(A, of("louvain", t.params)["resolution"], t.seed)))


@register("spade", 1, "cluster", ("sklearn",))
def spade(t: Task) -> Result:
    """SPADE-style: downsample (<=max_cells, uniform), agglomerative cluster, up-sample by nearest centroid."""
    p = of("spade", t.params)
    X, rng = t.X_test, np.random.default_rng(t.seed)
    sub = rng.choice(len(X), min(p["max_cells"], len(X)), replace=False)   # ponytail: uniform, SPADE is density-dependent
    lab = AgglomerativeClustering(n_clusters=min(_k(t), len(sub)), linkage=p["linkage"]).fit_predict(X[sub])
    cents = np.stack([X[sub][lab == c].mean(0) for c in np.unique(lab)])
    return Result(_cid(pairwise_distances_argmin(X, cents)))

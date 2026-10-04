"""Tier 3 - reference mapping (labels from the training split act as the reference atlas)."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from .base import MethodUnavailable, Result, Task, register
from .traditional import _need_labels


@register("singler", 3, "supervised", ("scipy",))
def singler(t: Task) -> Result:
    """SingleR-style: Spearman correlation of each cell to per-type reference profiles.

    Native re-implementation of the scoring step (top-variable features, argmax correlation); the
    iterative fine-tuning step of the R package is not included.
    """
    y = _need_labels(t)
    types = np.unique(y)
    ref = np.stack([t.X_train[y == c].mean(0) for c in types])
    top = np.argsort(ref.var(0))[::-1][:min(500, ref.shape[1])]
    R, Q = rankdata(ref[:, top], axis=1), rankdata(t.X_test[:, top], axis=1)
    R, Q = R - R.mean(1, keepdims=True), Q - Q.mean(1, keepdims=True)
    corr = (Q @ R.T) / (np.linalg.norm(Q, axis=1, keepdims=True) * np.linalg.norm(R, axis=1) + 1e-12)
    return Result(types[corr.argmax(1)])


def _scvi_adata(t: Task, idx: np.ndarray, labelled: np.ndarray | None):
    import anndata as ad
    a = ad.AnnData(t.X[idx].astype(np.float32))
    a.obs["labels"] = "Unknown" if labelled is None else labelled
    return a


def _accel(t: Task) -> str:
    return "gpu" if t.device == "cuda" else "cpu"


@register("scanvi", 3, "supervised", ("scvi",))
def scanvi(t: Task) -> Result:
    """scANVI semi-supervised annotation: train cells labelled, test cells 'Unknown' (scvi-tools)."""
    import anndata as ad
    from scvi.model import SCANVI
    y = _need_labels(t)
    idx = np.concatenate([t.train_idx, t.test_idx])
    a = ad.AnnData(t.X[idx].astype(np.float32))
    a.obs["labels"] = np.concatenate([y, np.full(len(t.test_idx), "Unknown")])
    SCANVI.setup_anndata(a, labels_key="labels", unlabeled_category="Unknown")
    m = SCANVI(a, gene_likelihood="normal")      # data is already transformed, not counts
    m.train(max_epochs=t.params.get("epochs", 50), accelerator=_accel(t))
    return Result(np.asarray(m.predict(a[len(y):])).astype(str))


@register("scarches", 3, "supervised", ("scvi",))
def scarches(t: Task) -> Result:
    """scArches: scANVI reference model on the training cells, query surgery on the test cells."""
    from scvi.model import SCANVI
    y = _need_labels(t)
    ref = _scvi_adata(t, t.train_idx, y)
    SCANVI.setup_anndata(ref, labels_key="labels", unlabeled_category="Unknown")
    rm = SCANVI(ref, gene_likelihood="normal")
    rm.train(max_epochs=t.params.get("epochs", 50), accelerator=_accel(t))
    q = _scvi_adata(t, t.test_idx, None)
    qm = SCANVI.load_query_data(q, rm)
    qm.train(max_epochs=t.params.get("query_epochs", 20), plan_kwargs=dict(weight_decay=0.0),
             accelerator=_accel(t))
    return Result(np.asarray(qm.predict()).astype(str))

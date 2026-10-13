"""Tier 3 - reference mapping (labels from the training split act as the reference atlas)."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from .base import MethodUnavailable, Result, Task, register
from .params import of
from .traditional import _need_labels


@register("singler", 3, "supervised", ("scipy",))
def singler(t: Task) -> Result:
    """SingleR-style: Spearman correlation of each cell to per-type reference profiles.

    Native re-implementation of the scoring step (top-variable features, argmax correlation); the
    iterative fine-tuning step of the R package is not included.
    """
    y = _need_labels(t)
    n_top = of("singler", t.params)["n_top"]
    types = np.unique(y)
    ref = np.stack([t.X_train[y == c].mean(0) for c in types])
    top = np.argsort(ref.var(0))[::-1][:min(n_top, ref.shape[1])]
    R, Q = rankdata(ref[:, top], axis=1), rankdata(t.X_test[:, top], axis=1)
    R, Q = R - R.mean(1, keepdims=True), Q - Q.mean(1, keepdims=True)
    corr = (Q @ R.T) / (np.linalg.norm(Q, axis=1, keepdims=True) * np.linalg.norm(R, axis=1) + 1e-12)
    return Result(types[corr.argmax(1)])


def _scvi_adata(t: Task, idx: np.ndarray, labelled: np.ndarray | None):
    import anndata as ad
    a = ad.AnnData(t.X[idx].astype(np.float32))
    a.obs["labels"] = "Unknown" if labelled is None else labelled
    return a


def _train_kw(p: dict) -> dict:
    """batch_size / learning rate for scvi's train(); only when set (scvi keeps its defaults otherwise)."""
    # drop_last: a final batch of one cell breaks BatchNorm ("Expected more than 1 value per channel"), as seen at small training fractions
    return {"datasplitter_kwargs": {"drop_last": True}, **({"batch_size": p["batch_size"]} if p["batch_size"] else {}),
            **({"plan_kwargs": {"lr": p["lr"]}} if p["lr"] else {})}


def _accel(t: Task) -> str:
    return "gpu" if t.device == "cuda" else "cpu"


def _retry(t: Task, fit, tries: int = 3):
    """scvi training sometimes diverges to NaN in its first batches (seen on some folds, finite inputs). Retry with a new seed
    and half the learning rate; ``fit(i)`` must train and return the model, raising if it did not train."""
    import scvi
    for i in range(tries):
        scvi.settings.seed = int(getattr(t, "seed", 0) or 0) + i
        try:
            return fit(i)
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"scvi training attempt {i + 1} failed ({str(e)[:90]}); retrying with seed+{i + 1} and half the learning rate", flush=True)


def _check_trained(m) -> None:
    h = m.history.get("train_loss_epoch") if isinstance(m.history, dict) else None
    if h is None or not len(h) or not np.isfinite(float(h.iloc[-1, 0])):
        raise RuntimeError("scvi training diverged (NaN loss)")


def _lr(p: dict, i: int):
    return {**p, "lr": (p["lr"] or 1e-3) / 2 ** i}


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
    p = of("scanvi", t.params)

    def fit(i):
        q = _lr(p, i)
        m = SCANVI(a, n_latent=q["n_latent"], n_hidden=q["n_hidden"], n_layers=q["n_layers"], gene_likelihood="normal")      # data is already transformed, not counts
        m.train(max_epochs=q["epochs"], accelerator=_accel(t), **_train_kw(q))
        _check_trained(m)
        return m

    m = _retry(t, fit)
    return Result(np.asarray(m.predict(a[len(y):])).astype(str))


@register("scarches", 3, "supervised", ("scvi",))
def scarches(t: Task) -> Result:
    """scArches: scANVI reference model on the training cells, query surgery on the test cells."""
    from scvi.model import SCANVI
    y = _need_labels(t)
    ref = _scvi_adata(t, t.train_idx, y)
    SCANVI.setup_anndata(ref, labels_key="labels", unlabeled_category="Unknown")
    p = of("scarches", t.params)

    def fit(i):
        q = _lr(p, i)
        rm = SCANVI(ref, n_latent=q["n_latent"], n_hidden=q["n_hidden"], n_layers=q["n_layers"], gene_likelihood="normal")
        rm.train(max_epochs=q["epochs"], accelerator=_accel(t), **_train_kw(q))
        _check_trained(rm)
        q_ad = _scvi_adata(t, t.test_idx, None)
        qm = SCANVI.load_query_data(q_ad, rm)
        qm.train(max_epochs=q["query_epochs"], plan_kwargs=dict(weight_decay=0.0),
                 accelerator=_accel(t), datasplitter_kwargs={"drop_last": True})
        _check_trained(qm)
        return qm

    qm = _retry(t, fit)
    return Result(np.asarray(qm.predict()).astype(str))

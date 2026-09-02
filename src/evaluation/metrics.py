"""
metrics.py
==========
Supervised, unsupervised, and distribution-recovery metrics.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import (
    adjusted_rand_score,
    cohen_kappa_score,
    davies_bouldin_score,
    f1_score,
    matthews_corrcoef,
    normalized_mutual_info_score,
    silhouette_score,
)

from hierarchy import parse_ancestor_map

# Re-use frequency tier logic from k-fold utilities when available.
_METHODS_UTILS = Path(__file__).resolve().parents[1] / "methods" / "utils"
if str(_METHODS_UTILS) not in sys.path:
    sys.path.insert(0, str(_METHODS_UTILS))

try:
    from kfold_strategies import frequency_tiers_from_series  # noqa: E402
except ImportError:
    frequency_tiers_from_series = None  # type: ignore


@dataclass
class SupervisedMetrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    ari: float
    nmi: float
    mcc: float
    kappa: float
    hierarchical_f1: Optional[float] = None
    g_mean: Optional[float] = None
    r2_composition: Optional[float] = None
    pearson_composition: Optional[float] = None
    kl_divergence: Optional[float] = None
    jensen_shannon: Optional[float] = None
    n_cells: int = 0
    # Rare / common cell-type benchmark metrics (optional).
    rare_macro_f1: Optional[float] = None
    common_macro_f1: Optional[float] = None
    min_class_recall: Optional[float] = None
    rare_min_recall: Optional[float] = None
    n_rare_types: Optional[int] = None
    n_common_types: Optional[int] = None
    most_common_type: Optional[str] = None
    rarest_type: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def hierarchical_f1_score(
    y_true: Sequence,
    y_pred: Sequence,
    ancestor_map: Dict[str, list],
) -> float:
    scores = []
    for true_label, pred_label in zip(y_true, y_pred):
        true_anc = set(ancestor_map.get(str(true_label), []))
        pred_anc = set(ancestor_map.get(str(pred_label), []))
        if not true_anc or not pred_anc:
            scores.append(0.0)
            continue
        intersection = true_anc & pred_anc
        precision = len(intersection) / len(pred_anc)
        recall = len(intersection) / len(true_anc)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        scores.append(f1)
    return float(np.mean(scores)) if scores else 0.0


def gmean_score(y_true: Sequence, y_pred: Sequence) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0
    cm = pd.crosstab(pd.Series(y_true), pd.Series(y_pred), dropna=False)
    sensitivities = []
    for label in cm.index:
        row_sum = cm.loc[label].sum()
        if row_sum > 0:
            sensitivities.append(cm.loc[label, label] / row_sum if label in cm.columns else 0.0)
    sensitivities = [s for s in sensitivities if s > 0]
    if not sensitivities:
        return 0.0
    return float(np.prod(sensitivities) ** (1 / len(sensitivities)))


def _composition_table(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    pred_pct = y_pred.value_counts(normalize=True) * 100
    true_pct = y_true.value_counts(normalize=True) * 100
    df = pd.DataFrame({"predicted_percentage": pred_pct, "true_percentage": true_pct}).fillna(0)
    return df


def composition_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    dist = _composition_table(y_true, y_pred)
    if dist.empty:
        return {"r2": 0.0, "pearson": 0.0, "kl_divergence": 0.0, "jensen_shannon": 0.0}

    r2 = float(dist["predicted_percentage"].corr(dist["true_percentage"]) ** 2)
    pearson = float(dist["predicted_percentage"].corr(dist["true_percentage"]))
    p = dist["predicted_percentage"].values / 100
    t = dist["true_percentage"].values / 100
    kl = float(np.sum(p * np.log2((p + 1e-9) / (t + 1e-9))))
    js = float(jensenshannon(p, t, base=2))
    return {"r2": r2, "pearson": pearson, "kl_divergence": kl, "jensen_shannon": js}


def _tiered_subset_f1(
    y_true: pd.Series,
    y_pred: pd.Series,
    tiers: Dict[str, str],
    tier_name: str,
) -> Optional[float]:
    labels = [lbl for lbl, tier in tiers.items() if tier == tier_name]
    if not labels:
        return None
    mask = y_true.isin(labels)
    if not mask.any():
        return None
    return float(
        f1_score(
            y_true[mask],
            y_pred[mask],
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


def _min_recall_for_labels(y_true: pd.Series, y_pred: pd.Series, labels: list) -> Optional[float]:
    if not labels:
        return None
    recalls = []
    for label in labels:
        mask = y_true == label
        if not mask.any():
            continue
        recalls.append(float((y_pred[mask] == label).mean()))
    return float(min(recalls)) if recalls else None


def compute_rare_type_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    rare_fraction: float = 0.01,
    common_fraction: float = 0.05,
) -> dict:
    """Benchmark metrics focused on rare and common cell populations."""
    yt = y_true.dropna().astype(str)
    yp = y_pred.dropna().astype(str)
    if len(yt) == 0:
        return {}

    if frequency_tiers_from_series is not None:
        tiers = frequency_tiers_from_series(
            yt, rare_fraction=rare_fraction, common_fraction=common_fraction,
        )
    else:
        counts = yt.value_counts(normalize=True)
        tiers = {
            lbl: ("rare" if frac < rare_fraction else "common" if frac >= common_fraction else "intermediate")
            for lbl, frac in counts.items()
        }

    rare_labels = [lbl for lbl, t in tiers.items() if t == "rare"]
    common_labels = [lbl for lbl, t in tiers.items() if t == "common"]
    counts = yt.value_counts()

    return {
        "rare_macro_f1": _tiered_subset_f1(yt, yp, tiers, "rare"),
        "common_macro_f1": _tiered_subset_f1(yt, yp, tiers, "common"),
        "min_class_recall": _min_recall_for_labels(yt, yp, sorted(yt.unique())),
        "rare_min_recall": _min_recall_for_labels(yt, yp, rare_labels),
        "n_rare_types": len(rare_labels),
        "n_common_types": len(common_labels),
        "most_common_type": counts.index[0] if len(counts) else None,
        "rarest_type": counts.index[-1] if len(counts) else None,
    }


def per_class_metrics_table(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    rare_fraction: float = 0.01,
    common_fraction: float = 0.05,
) -> pd.DataFrame:
    """Per-phenotype recall, F1, support, and frequency tier."""
    yt = y_true.dropna().astype(str)
    yp = y_pred.dropna().astype(str)
    labels = sorted(set(yt) | set(yp))
    if not labels:
        return pd.DataFrame()

    if frequency_tiers_from_series is not None:
        tiers = frequency_tiers_from_series(
            yt, rare_fraction=rare_fraction, common_fraction=common_fraction,
        )
    else:
        norm = yt.value_counts(normalize=True)
        tiers = {
            lbl: ("rare" if norm.get(lbl, 0) < rare_fraction else "common" if norm.get(lbl, 0) >= common_fraction else "intermediate")
            for lbl in labels
        }

    support = yt.value_counts()
    total = int(support.sum())
    rows = []
    for label in labels:
        mask = yt == label
        sup = int(support.get(label, 0))
        rec = float((yp[mask] == label).mean()) if sup else 0.0
        f1 = float(f1_score(yt == label, yp == label, zero_division=0))
        rows.append({
            "phenotype": label,
            "support": sup,
            "fraction": sup / total if total else 0.0,
            "frequency_tier": tiers.get(label, "intermediate"),
            "recall": rec,
            "f1": f1,
        })
    return pd.DataFrame(rows).sort_values("support", ascending=False)


def compute_supervised_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    level: str = "level3",
    hierarchy_path: Optional[str] = None,
    rare_fraction: float = 0.01,
    common_fraction: float = 0.05,
    include_rare_metrics: bool = True,
) -> SupervisedMetrics:
    """Compute classification + clustering agreement metrics."""
    mask = y_true.notna() & y_pred.notna()
    yt = y_true[mask].astype(str)
    yp = y_pred[mask].astype(str)
    n = len(yt)
    if n == 0:
        return SupervisedMetrics(
            0, 0, 0, 0, 0, 0, 0, n_cells=0,
        )

    comp = composition_metrics(yt, yp)
    h_f1 = None
    if level == "level3":
        anc = parse_ancestor_map(hierarchy_path) if hierarchy_path else parse_ancestor_map()
        h_f1 = hierarchical_f1_score(yt, yp, anc)

    rare_metrics = (
        compute_rare_type_metrics(
            yt, yp, rare_fraction=rare_fraction, common_fraction=common_fraction,
        )
        if include_rare_metrics
        else {}
    )

    return SupervisedMetrics(
        accuracy=float((yt == yp).mean()),
        macro_f1=float(f1_score(yt, yp, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(yt, yp, average="weighted", zero_division=0)),
        ari=float(adjusted_rand_score(yt, yp)),
        nmi=float(normalized_mutual_info_score(yt, yp)),
        mcc=float(matthews_corrcoef(yt, yp)),
        kappa=float(cohen_kappa_score(yt, yp)),
        hierarchical_f1=h_f1,
        g_mean=gmean_score(yt, yp),
        r2_composition=comp["r2"],
        pearson_composition=comp["pearson"],
        kl_divergence=comp["kl_divergence"],
        jensen_shannon=comp["jensen_shannon"],
        n_cells=n,
        **rare_metrics,
    )


@dataclass
class UnsupervisedMetrics:
    silhouette: Optional[float] = None
    davies_bouldin: Optional[float] = None
    spatial_entropy: Optional[float] = None
    marker_consistency: Optional[float] = None
    marker_purity: Optional[float] = None
    neighborhood_consistency: Optional[float] = None
    n_cells: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# Biological marker purity rules (fallback when no dataset rules are passed).
# Prefer load_marker_purity_rules() from marker_rules.py for dataset-specific rules.
from marker_rules import IMMUCAN_FALLBACK as DEFAULT_MARKER_RULES  # noqa: E402


def spatial_entropy(x: np.ndarray, y: np.ndarray, grid_size: int = 50) -> float:
    """Shannon entropy of the 2D spatial occupancy histogram."""
    if len(x) == 0:
        return 0.0
    x_range = max(float(x.max() - x.min()), 1e-9)
    y_range = max(float(y.max() - y.min()), 1e-9)
    x_bin = np.floor((x - x.min()) / x_range * (grid_size - 1)).astype(int)
    y_bin = np.floor((y - y.min()) / y_range * (grid_size - 1)).astype(int)
    counts = pd.crosstab(x_bin, y_bin).values.flatten()
    counts = counts[counts > 0]
    probs = counts / counts.sum()
    return float(scipy_entropy(probs, base=2))


def marker_consistency_score(
    marker_data: pd.DataFrame,
    cluster_labels: pd.Series,
) -> float:
    """Mean within-cluster / between-cluster variance ratio across markers (higher = better)."""
    if marker_data.empty or cluster_labels.nunique() < 2:
        return 0.0
    scores = []
    for col in marker_data.columns:
        vals = marker_data[col].values
        global_var = np.var(vals)
        if global_var < 1e-12:
            continue
        within = []
        for cl in cluster_labels.unique():
            v = vals[cluster_labels == cl]
            if len(v) > 1:
                within.append(np.var(v))
        if not within:
            continue
        ratio = global_var / (np.mean(within) + 1e-9)
        scores.append(min(ratio, 10.0))
    return float(np.mean(scores)) if scores else 0.0


def marker_purity_score(
    df: pd.DataFrame,
    label_col: str,
    marker_rules: Optional[Dict[str, Dict[str, list]]] = None,
    threshold: float = 0.5,
) -> float:
    """Fraction of cells whose marker expression matches biological expectations."""
    rules = marker_rules or DEFAULT_MARKER_RULES
    if label_col not in df.columns:
        return 0.0

    label_series = df[label_col].astype(str)
    scores = []
    for label, rule in rules.items():
        cells = df[label_series == label]
        if cells.empty:
            continue
        cell_scores = []
        for m in rule.get("positive", []):
            col = next((c for c in df.columns if m.lower() in c.lower()), None)
            if col:
                cell_scores.append((cells[col] >= threshold).astype(float))
        for m in rule.get("negative", []):
            col = next((c for c in df.columns if m.lower() in c.lower()), None)
            if col:
                cell_scores.append((cells[col] < threshold).astype(float))
        if cell_scores:
            scores.append(pd.concat(cell_scores, axis=1).mean(axis=1).mean())
    return float(np.mean(scores)) if scores else 0.0


def neighborhood_consistency_score(
    df: pd.DataFrame,
    label_col: str,
    k: int = 10,
) -> float:
    """Fraction of cells whose label matches the majority label among k spatial neighbors."""
    if "x" not in df.columns or "y" not in df.columns:
        return 0.0
    from scipy.spatial import cKDTree

    coords = df[["x", "y"]].values.astype(float)
    labels = df[label_col].astype(str).values
    if len(coords) < k + 1:
        return 1.0

    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=min(k + 1, len(coords)))
    if indices.ndim == 1:
        indices = indices.reshape(-1, 1)

    consistent = 0
    for i, neighbors in enumerate(indices):
        neighbor_labels = [labels[j] for j in neighbors if j != i]
        if not neighbor_labels:
            continue
        majority = pd.Series(neighbor_labels).mode().iloc[0]
        if labels[i] == majority:
            consistent += 1
    return consistent / len(coords)


def _subsample_df(df: pd.DataFrame, max_cells: int, seed: int = 0) -> pd.DataFrame:
    if len(df) <= max_cells:
        return df
    return df.sample(n=max_cells, random_state=seed)


def compute_unsupervised_metrics(
    df: pd.DataFrame,
    cluster_col: str,
    marker_cols: list[str],
    *,
    max_cells: int = 20000,
    marker_rules: Optional[Dict[str, Dict[str, list]]] = None,
) -> UnsupervisedMetrics:
    """Metrics for cluster-quality analysis (subsamples large datasets for speed)."""
    n = len(df)
    work = _subsample_df(df, max_cells=max_cells) if n > max_cells else df

    sil = None
    db = None
    present_markers = [c for c in marker_cols if c in work.columns]
    if present_markers and work[cluster_col].nunique() > 1 and work[cluster_col].nunique() < len(work):
        X = work[present_markers].fillna(0).values
        labels = work[cluster_col].values
        try:
            sil = float(silhouette_score(X, labels, metric="euclidean"))
        except Exception:
            sil = None
        try:
            db = float(davies_bouldin_score(X, labels))
        except Exception:
            db = None

    sp_ent = None
    if "x" in work.columns and "y" in work.columns:
        sp_ent = spatial_entropy(work["x"].values, work["y"].values)

    mc = None
    mp = None
    nc = None
    if present_markers:
        mc = marker_consistency_score(work[present_markers], work[cluster_col])
        mp = marker_purity_score(work, cluster_col, marker_rules=marker_rules)
    if "x" in work.columns and "y" in work.columns:
        nc = neighborhood_consistency_score(work, cluster_col)

    return UnsupervisedMetrics(
        silhouette=sil,
        davies_bouldin=db,
        spatial_entropy=sp_ent,
        marker_consistency=mc,
        marker_purity=mp,
        neighborhood_consistency=nc,
        n_cells=n,
    )

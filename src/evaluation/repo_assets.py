"""Loaders for the benchmark's existing evaluation/plotting assets, so new code scores and colours
results exactly like the notebooks in ``src/evaluation`` and ``src/plotting``.

* ``hierarchy_mappings.pkl``  label -> {level_2_cell_type, level_1_cell_type}
* ``cell_type_hierarchy.txt``  indented tree used for hierarchical F1 (``parse_tree_file`` /
  ``hierarchical_f1_score`` are ported unchanged from ``eval_mapping.ipynb``)
* ``method_colors.json`` / ``dataset_colors.json``  RGB colours
* ``scalability_score.json``  per-method scaling scores (a python-literal file despite the name)
* score definitions from ``eval_mapping.ipynb`` / ``visualizations.ipynb`` (weights, stability, runtime scaling)
"""
from __future__ import annotations

import ast
import json
import pickle
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..repo import notebook_functions

SRC = Path(__file__).resolve().parents[1]
EVAL, PLOT = SRC / "evaluation", SRC / "plotting"

# weights from visualizations.ipynb (calculate_weighted_score); hierarchical F1 is absent at level 1/2
OVERALL_WEIGHTS = {"f1_weighted": 0.16, "hierarchical_f1": 0.24, "f1_macro": 0.20, "mcc": 0.20, "ari": 0.08,
                   "jsd_scaled": 0.12}
STABILITY_THRESH = 0.2          # eval_mapping.ipynb: stability = 1 - std(weighted F1) / 0.2
RUNTIME_THRESH_S = 28800        # visualizations.ipynb: runtime_scaled = 1 - run_time / 8h

CATEGORIES = ["Supervised", "Unsupervised", "Prior-Knowledge-driven", "Pre-trained Models", "Baselines"]
_CATEGORY = {
    **dict.fromkeys(["logistic_regression", "random_forest", "xgboost", "svm", "spatial_gnn", "knn_smooth",
                     "cellsighter", "stellar", "maps", "scanvi", "scarches"], "Supervised"),
    **dict.fromkeys(["leiden", "louvain", "flowsom", "spade", "starling", "nimbus", "phenograph", "fusesom"], "Unsupervised"),
    **dict.fromkeys(["tacit", "astir", "scyan", "singler", "tribus", "marker_score"], "Prior-Knowledge-driven"),
    **dict.fromkeys(["deepcelltypes", "ribca"], "Pre-trained Models"),
    **dict.fromkeys(["most_frequent", "stratified"], "Baselines"),
}
_CAT_SCALE_KEY = {"Supervised": "Supervised", "Unsupervised": "Clustering_based",
                   "Prior-Knowledge-driven": "Prior-knowledge based", "Pre-trained Models": "Pre-trained models",
                   "Baselines": "Baselines"}
_NAME_FIX = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost",
             "most_frequent": "Most Frequent", "stratified": "Stratified sampler"}


def category(method: str) -> str:
    base = method.removesuffix("+vote")
    if base in _CATEGORY:
        return _CATEGORY[base]
    return next((c for k, c in sorted(_CATEGORY.items(), key=lambda kc: -len(kc[0])) if base.startswith(k)), "Supervised")


@lru_cache(maxsize=1)
def method_colors() -> dict[str, tuple]:
    return {k: tuple(v) for k, v in json.loads((PLOT / "method_colors.json").read_text()).items()}


@lru_cache(maxsize=1)
def dataset_colors() -> dict[str, tuple]:
    return {k: tuple(v) for k, v in json.loads((PLOT / "dataset_colors.json").read_text()).items()}


def color_for_method(method: str) -> tuple:
    return method_colors().get(category(method), (0.5, 0.5, 0.5))


def color_for_dataset(name: str, i: int = 0) -> tuple:
    import matplotlib.pyplot as plt
    return dataset_colors().get(name) or plt.get_cmap("tab10")(i % 10)[:3]


@lru_cache(maxsize=1)
def _scalability() -> tuple[dict, dict]:
    txt = (EVAL / "scalability_score.json").read_text()
    cat = ast.literal_eval(re.search(r"methods2scalability\s*=\s*(\{.*?\})", txt, re.S).group(1))
    per = ast.literal_eval(re.search(r"scaling_scores\s*=\s*(\{.*?\})", txt, re.S).group(1))
    return cat, per


def scaling_score(method: str) -> float:
    """Repo scaling score for a method, falling back to its category default."""
    cat, per = _scalability()
    base = method.removesuffix("+vote")
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    table = {norm(k): v for k, v in per.items()}
    for cand in (_NAME_FIX.get(base, base), base, base.split("_res")[0]):
        if norm(cand) in table:
            return float(table[norm(cand)])
    return float(cat[_CAT_SCALE_KEY[category(method)]])


@lru_cache(maxsize=1)
def hierarchy_mappings() -> dict:
    with open(EVAL / "hierarchy_mappings.pkl", "rb") as f:
        return pickle.load(f)


def to_level(labels, level: str, true_cols: dict | None = None) -> np.ndarray:
    """Map level-3 labels to level_2 / level_1 using the repo mapping; unmapped labels stay unchanged
    (same convention as eval_mapping.ipynb). ``true_cols`` can provide dataset-native columns."""
    if level == "level3":
        return np.asarray(labels).astype(str)
    if true_cols and level in true_cols:
        return np.asarray(true_cols[level]).astype(str)
    key = {"level2": "level_2_cell_type", "level1": "level_1_cell_type"}[level]
    m = hierarchy_mappings()
    l2_to_l1 = {v["level_2_cell_type"]: v["level_1_cell_type"] for v in m.values()}   # for labels already at level 2
    return np.array([(m.get(l) or {}).get(key) or (l2_to_l1.get(l) if level == "level1" else None) or l
                     for l in np.asarray(labels).astype(str)])


# The metric helpers below are the notebook's own functions, extracted at import time (not copies).
_NB = EVAL / "eval_mapping.ipynb"
_FNS = notebook_functions(_NB, ("calculate_cell_type_distribution", "calculate_r2_and_pearson", "parse_tree_file",
                                "hierarchical_f1_score", "gmean_score", "calculate_time"))
calculate_cell_type_distribution = _FNS["calculate_cell_type_distribution"]   # (df, predictions, true_label)
calculate_r2_and_pearson = _FNS["calculate_r2_and_pearson"]
parse_tree_file = _FNS["parse_tree_file"]
hierarchical_f1_score = _FNS["hierarchical_f1_score"]
gmean_score = _FNS["gmean_score"]
calculate_time = _FNS["calculate_time"]                                         # fold_times.txt -> (train, inference)


@lru_cache(maxsize=1)
def ancestors() -> dict:
    return parse_tree_file(str(EVAL / "cell_type_hierarchy.txt"))


def overall_score(row, level: str = "level3") -> float:
    """Weighted overall performance as in visualizations.ipynb (metrics absent from the row are skipped)."""
    return float(sum(row[m] * w for m, w in OVERALL_WEIGHTS.items()
                     if m in row and not (m == "hierarchical_f1" and level != "level3") and not np.isnan(row[m])))

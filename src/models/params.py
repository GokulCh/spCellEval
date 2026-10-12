"""Per-dataset, per-method parameters, and the search space the tuner explores.

``configs/method_params.json`` holds the values (``python cli.py tune`` writes the best ones there)::

    {"default": {"leiden": {"resolutions": [0.8]}},
     "IMMUcan": {"xgboost": {"n_estimators": 250, "max_depth": 7}}}

Precedence, lowest first: the defaults in ``SPACE`` < the ``"default"`` section < the dataset's section < per-run overrides.
Every parameter a method accepts is listed in ``SPACE`` as ``name: (default, tuning candidates)`` (an empty candidate list
= never tuned). ``None`` means "derive it from the data" (e.g. ``n_clusters`` = number of classes) or "leave the script's
own default". Script / classic methods also accept ``extra_args`` (a list of raw CLI tokens appended to the script call).
List all of it with ``python cli.py params``.
"""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

from ..repo import ROOT

CONFIG = ROOT / "configs" / "method_params.json"
_C = lambda *v: list(v)

SPACE: dict[str, dict[str, tuple]] = {
    # ---- native (src/models)
    "svm": {"C": (1.0, _C(0.1, 1.0, 10.0, 100.0)), "kernel": ("rbf", _C("rbf", "linear")), "gamma": ("scale", _C("scale", "auto")),
            "class_weight": ("balanced", _C("balanced", None)), "max_rbf_cells": (20000, []), "scale": (True, _C(True, False))},
    "louvain": {"resolution": (1.0, _C(0.5, 0.8, 1.0, 1.5, 2.0)), "k_neighbors": (None, _C(5, 10, 15, 30))},
    "spade": {"n_clusters": (None, _C(10, 20, 30, 50)), "linkage": ("ward", _C("ward", "average", "complete")), "max_cells": (5000, _C(2000, 5000, 10000))},
    "singler": {"n_top": (500, _C(100, 250, 500, 1000))},
    "scanvi": {"epochs": (50, _C(25, 50, 100)), "n_latent": (10, _C(5, 10, 30)), "n_hidden": (128, _C(64, 128, 256)),
               "n_layers": (1, _C(1, 2)), "batch_size": (128, _C(64, 128, 256)), "lr": (None, _C(0.0005, 0.001, 0.002))},
    "scarches": {"epochs": (50, _C(25, 50, 100)), "query_epochs": (20, _C(10, 20, 50)), "n_latent": (10, _C(5, 10, 30)), "n_hidden": (128, _C(64, 128, 256)),
                 "n_layers": (1, _C(1, 2)), "batch_size": (128, _C(64, 128, 256)), "lr": (None, _C(0.0005, 0.001, 0.002))},
    "spatial_gnn": {"k_neighbors": (None, _C(5, 10, 15, 30)), "hops": (2, _C(1, 2, 3)), "C": (1.0, _C(0.1, 1.0, 10.0)), "max_iter": (300, [])},
    "knn_smooth": {"k_neighbors": (None, _C(5, 10, 15, 30)), "alpha": (0.5, _C(0.25, 0.5, 0.75)), "n_estimators": (200, _C(100, 200, 400)),
                   "max_depth": (None, _C(None, 20, 40))},
    "marker_score": {"min_score": (0.0, _C(0.0, 0.25, 0.5))},
    # ---- the repo's classic-ML script (forwarded as --model_kwargs)
    "logistic_regression": {"C": (1.0, _C(0.001, 0.01, 0.1, 1.0, 10.0, 100.0)), "max_iter": (100, _C(100, 300)), "class_weight": ("balanced", _C("balanced", None)),
                            "scaling": ("Yes", _C("Yes", "No"))},
    "random_forest": {"n_estimators": (100, _C(100, 250, 500)), "max_depth": (None, _C(None, 20, 40)), "min_samples_split": (2, _C(2, 5, 7)),
                      "min_samples_leaf": (1, _C(1, 2, 4)), "max_features": ("sqrt", _C("sqrt", "log2", 0.5)), "class_weight": ("balanced", _C("balanced", None)),
                      "scaling": ("Yes", _C("Yes", "No"))},
    "xgboost": {"n_estimators": (100, _C(100, 250, 500)), "max_depth": (6, _C(4, 6, 8, 10)), "learning_rate": (0.3, _C(0.03, 0.1, 0.3)),
                "min_child_weight": (1, _C(1, 3)), "subsample": (1.0, _C(0.7, 1.0)), "colsample_bytree": (1.0, _C(0.7, 1.0)),
                "reg_lambda": (1.0, _C(0.5, 1.0, 5.0)), "gamma": (0.0, _C(0.0, 1.0)), "early_stopping_rounds": (10, _C(10, 20)),
                "scaling": ("Yes", _C("Yes", "No"))},
    "most_frequent": {}, "stratified": {},
    # ---- repo scripts (each value is the script's own CLI option)
    "leiden": {"resolutions": ([0.5, 0.8, 1.0, 2.0], _C([0.3], [0.5], [0.8], [1.0], [1.5], [2.0])), "n_neighbors": (10, _C(5, 10, 15, 30)),
               "pca": (False, _C(False, True)), "n_pcs": (30, _C(10, 20, 30, 50)), "normalization": (False, _C(False, True))},
    "flowsom": {"n_clusters": (None, _C(10, 15, 20, 30, 40)), "xdim": (10, _C(5, 10, 15)), "ydim": (10, _C(5, 10, 15)),
                "rlen": (10, _C(10, 30, 100)), "mst": (1, _C(1, 3)), "distf": (2, _C(1, 2))},
    "tacit": {"r": (10, _C(10, 25, 50, 100, 200)), "p": (10, _C(5, 10, 20))},
    "scyan": {"prior_std": (0.3, _C(0.1, 0.3, 0.5, 1.0)), "patience": (10, _C(5, 10, 20)), "temperature": (0.5, _C(0.25, 0.5, 1.0)),
              "scaling": (None, []), "log1p": (False, []),
              "lr": (None, _C(0.00025, 0.0005, 0.001)), "hidden_size": (None, _C(16, 32, 64)), "n_hidden_layers": (None, _C(4, 6, 8)),
              "n_layers": (None, _C(5, 7, 10)), "max_epochs": (None, _C(50, 100, 200)), "min_delta": (None, _C(0.25, 0.5, 1.0)),
              "batch_size": (None, _C(2048, 8192)), "max_samples": (None, _C(100000, 200000, 500000)), "modulo_temp": (None, _C(2, 3, 5))},
    "astir": {"max_epochs": (50, _C(25, 50, 100)), "n_init": (5, []), "n_init_epochs": (5, []), "delta_loss": (0.001, []),
              "learning_rate": (0.001, _C(0.0005, 0.001, 0.005)), "batch_size": (128, _C(64, 128, 256)), "scaling": (None, []), "log1p": (False, []),
              "threshold": (0.7, _C(0.5, 0.7, 0.9)), "assignment_type": ("threshold", _C("threshold", "max"))},
    "tribus": {"sigma": (1.0, _C(0.5, 1.0, 2.0)), "learning_rate": (1.0, _C(0.5, 1.0, 2.0)), "clustering_threshold": (100.0, _C(50.0, 100.0, 200.0)),
               "undefined_threshold": (0.0005, []), "other_threshold": (0.4, _C(0.2, 0.4, 0.6)), "depth": (3, []),
               "scaling": (None, []), "log1p": (False, []), "normalization": (None, []), "tuning": (None, []),
               "outlier_quantile": (0.999, _C(0.99, 0.999, 1.0))},
    "starling": {"k_cluster": (None, _C(10, 20, 30, 50)), "initial_clustering": ("KM", _C("KM", "GMM")), "dist_option": ("T", _C("T", "N")),
                 "singlet_prop": (0.6, _C(0.4, 0.6, 0.8)), "model_regularizer": (1, _C(0, 1, 5)), "lr": (0.001, _C(0.0005, 0.001, 0.005)),
                 "transform": (None, []), "scaling": (None, []), "model_cell_size": (None, []),
                 "max_epochs": (None, _C(50, 100, 200)), "patience": (None, _C(3, 10, 30)), "min_delta": (None, []),
                 "model_zplane_overlap": (None, [])},
    "maps": {"max_epochs": (2, _C(2, 10, 30, 100)), "min_epochs": (1, []), "batch_size": (128, _C(64, 128, 256)), "patience": (10, _C(10, 30, 100)),
             "learning_rate": (0.001, _C(0.0003, 0.001, 0.003)), "dropout": (0.1, _C(0.0, 0.1, 0.3)),
             "standard_split": (False, _C(False, True))},
    "stellar": {"hid_dim": (160, _C(64, 160, 256)), "distance_threshold": (0.0, []), "epochs": (50, _C(25, 50, 100)),
                "lr": (0.001, _C(0.0005, 0.001, 0.005)), "weight_decay": (0.0001, []), "image_batch_size": (16, []),
                "distance_multiplier": (3.0, _C(2.0, 3.0, 5.0))},
}
EXTRA = {"leiden", "flowsom", "tacit", "scyan", "astir", "tribus", "starling", "maps", "stellar",
         "logistic_regression", "random_forest", "xgboost", "most_frequent", "stratified"}      # methods that run a script
for _m in EXTRA:
    SPACE[_m]["extra_args"] = ([], [])


def defaults(method: str) -> dict:
    return {k: v[0] for k, v in SPACE.get(method, {}).items()}


def _read(path) -> dict:
    p = Path(path or CONFIG)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def resolve(dataset: str, method: str, overrides: dict | None = None, path=None, explicit: bool = False) -> dict:
    """Parameters of ``method`` on ``dataset``. ``explicit=True`` returns only what the config / overrides set (scripts and
    the classic-ML script then keep their own defaults for everything else); otherwise the defaults are filled in."""
    cfg = _read(path)
    got = {**cfg.get("default", {}).get(method, {}), **cfg.get(dataset, {}).get(method, {}), **(overrides or {})}
    unknown = sorted(set(got) - set(SPACE.get(method, {})))
    if unknown:
        raise ValueError(f"unknown parameter(s) {unknown} for method '{method}'. Valid: {sorted(SPACE.get(method, {})) or 'none'}")
    return got if explicit else {**defaults(method), **got}


def save(dataset: str, method: str, values: dict, path=None) -> None:
    """Merge ``values`` into the config under ``dataset`` (other entries and the file's ``_comment`` are kept)."""
    p = Path(path or CONFIG)
    cfg = _read(p)
    cfg.setdefault(dataset, {}).setdefault(method, {}).update(values)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def trials(method: str, n: int, seed: int = 0, start: dict | None = None) -> list[dict]:
    """Up to ``n`` parameter sets from the method's candidate grid. The first is always ``start`` (the current setting), the
    rest the full grid if it fits, else a random sample of it."""
    grid = {k: v[1] for k, v in SPACE.get(method, {}).items() if v[1]}
    first = {k: (start or {}).get(k, SPACE[method][k][0]) for k in grid}
    combos = [dict(zip(grid, c)) for c in itertools.product(*grid.values())] if grid else []
    combos = [c for c in combos if c != first]
    random.Random(seed).shuffle(combos)
    return ([first] + combos[:max(n - 1, 0)]) if grid else []


def of(method: str, given: dict) -> dict:
    """``Task.params`` with the method's defaults filled in (what the native methods read)."""
    return {**defaults(method), **given}


_JSON = {bool: "boolean", int: "integer", float: "number", str: "string", list: "array", type(None): "null"}


def schema() -> dict:
    """JSON Schema of the params file (editors then autocomplete and flag unknown / mistyped parameters)."""
    def types(vals):
        ts = {_JSON[type(v)] for v in vals}
        return sorted(ts - {"integer"} if "number" in ts else ts)
    sec = {"type": "object", "additionalProperties": False, "properties": {
        m: {"type": "object", "additionalProperties": False,
            "properties": {k: {"type": types([d, *c])} for k, (d, c) in sp.items()}} for m, sp in SPACE.items()}}
    return {"$schema": "http://json-schema.org/draft-07/schema#", "title": "spCellEval method parameters",
            "type": "object", "properties": {"$schema": {"type": "string"}, "_comment": {"type": "string"}, "default": sec}, "additionalProperties": sec}

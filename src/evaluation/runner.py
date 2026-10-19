"""Benchmark runner: orchestrates the repo's own pipeline and scores whatever it produces.

    load -> transform -> workspace (repo layout; folds via run_kfold_creator / DataSetHandler)
         -> methods: classic ML  = methods/classic_ml/run_classic_ml_default.py   (cv / hold-out / progressive variants)
                     scripts     = methods/<name>/run_*.py or .R                   (one run over all cells, as in the repo)
                     native      = src/models (only SVM, Louvain, SPADE, SingleR, scANVI, scArches, spatial methods,
                                   marker_score - methods the repo does not have)
         -> predictions_*.csv -> metrics (notebook functions) -> raw tables

Raw outputs (analysis re-runs from these alone):
    <out>/benchmark_results.csv   one row per method x split x fold x fraction (status, runtime, metrics, pred_file)
    <out>/level_metrics.csv       classification metrics at level3/level2/level1 (repo hierarchy mapping)
    <out>/per_class_results.csv   precision / recall / specificity / F1 / support per class per run
    <out>/feature_importance.csv  per-marker importance (from native models and the repo's pickled classic models)
    <out>/<dataset>/dataset_report/, <dataset>/workspace/ (the repo-layout data, folds, classic-ML results), <dataset>/<method>/...
"""
from __future__ import annotations

import importlib.util
import json
import logging
import threading
import time
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..models import REGISTRY, MethodUnavailable, Task, resolve_device, run_isolated
from ..models import params as mparams
from ..models.marker import AUTO_NOTE, PRIOR_METHODS, conform_matrix, draft_matrix, find_marker_matrix, load_marker_matrix, matrix_agreement, pseudo_labels
from ..models.scripts import ScriptCtx, classic_importance, parse_fold_times, run_classic, run_script
from ..models.spatial import spatial_vote
from ..preprocessing import build_workspace, load_dataset, make_variant, progressive, transform
from ..preprocessing.data import LEVEL_COLUMN
from ..preprocessing.splits import DEFAULT_FRACTIONS
from ..preprocessing.workspace import resolve_store, validation_as_test
from .analysis import dataset_report, level_labels
from .metrics import per_class_table, spatial_metrics, supervised_metrics, unsupervised_qc
from .repo_assets import to_level

log = logging.getLogger("spcelleval")
LEVELS = ["level3", "level2", "level1"]
FRAMES = ("results", "importance", "levels", "classes")
FILES = dict(results="benchmark_results.csv", importance="feature_importance.csv",
             levels="level_metrics.csv", classes="per_class_results.csv")
NAN = float("nan")


def _greedy():
    """The repo's cluster->label mapping (methods/utils/greedy_f1_utils.py)."""
    f = Path(__file__).resolve().parents[1] / "methods" / "utils" / "greedy_f1_utils.py"
    spec = importlib.util.spec_from_file_location("greedy_f1_utils", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.greedy_f1_score


@dataclass
class BenchConfig:
    data: list[str]
    out: str = "results/benchmark"
    methods: list[str] = field(default_factory=list)
    mode: str = "supervised"               # supervised | unsupervised
    split: str = "holdout"                 # holdout | cv | progressive | all
    kfold_method: str = "StratifiedKFold"  # StratifiedKFold | StratifiedGroupKFold | GroupShuffleSplit (repo DataSetHandler)
    folds: int = 5
    fractions: tuple = DEFAULT_FRACTIONS
    modality: str = "codex"
    transform: str = "auto"
    cofactor: float = 5.0
    normalize: str = "none"
    batch_correct: str = "none"
    level: str = "level3"
    marker_matrix: str | None = None
    order: str = "fast-first"             # fast-first | slow-first | listed (queue order of the methods)
    tacit_r: int | None = None             # shortcuts for the tacit parameters r / p (override the params file when set)
    tacit_p: int | None = None
    params_file: str | None = None         # method parameters per dataset (default configs/method_params.json, see models/params.py)
    overrides: dict = field(default_factory=dict)   # {method: {param: value}} on top of the params file (used by the tuner)
    report: bool = True                    # write the dataset report (the tuner turns it off)
    val_pct: float = 0.15                  # share of each training fold held out as the validation split
    pseudo_min_score: float = 0.5          # unsupervised mode: a marker pseudo-label is 'confident' above this score ...
    pseudo_min_margin: float = 0.25        # ... and this margin over the runner-up type
    tune_on_validation: bool = False      # tuner: the validation split replaces the test split in this run's private folds
    cache_data: bool = False               # tuner: load + transform each table once and reuse it across trials
    auto_matrix: bool = True               # build a draft decision matrix from the labels when a marker method has none
    max_cells: int = 0
    timeout: float = 0.0
    jobs: int = 1
    device: str = "auto"
    seed: int = 0
    k_neighbors: int = 10
    n_jobs_model: int = 1
    script_runs: int = 1                   # repeats for the repo scripts that support n_runs (stability needs > 1)
    save_predictions: bool = True
    resume: bool = False                   # skip methods whose results are already saved (ok / skipped) in <out>/<dataset>/_methods
    splits_dir: str | None = None          # where the folds live: None/'auto' (next to the data file), 'out', or a folder


# Rough runtime in minutes of each method on a ~235k-cell, 56-marker table (from a real run; it grows with dataset size).
# Only used to order the queue; unknown methods count as 10.
EST_MINUTES = {"marker_score": 1, "spade": 1, "singler": 2, "most_frequent": 2, "stratified": 2, "tacit": 5, "svm": 7,
               "logistic_regression": 8, "flowsom": 8, "astir": 10, "scyan": 10, "tribus": 10, "spatial_gnn": 10, "xgboost": 10,
               "louvain": 12, "random_forest": 15, "knn_smooth": 15, "maps": 15, "starling": 30, "scanvi": 40, "scarches": 40,
               "leiden": 40}


def order_methods(names: list[str], how: str = "fast-first") -> list[str]:
    """Queue order. Methods that cannot run here (they just report 'skipped') always go first; the rest by estimated cost.

    fast-first: quick results early, nothing waits behind a slow method.  slow-first: starts the slowest methods at once,
    which usually finishes the whole run sooner.  listed: the order given.
    """
    if how == "listed":
        return list(names)
    ready = lambda b: REGISTRY[b].status() == "ready"
    sign = 1 if how == "fast-first" else -1
    return sorted(names, key=lambda b: (0, 0) if not ready(b) else (1, sign * EST_MINUTES.get(b, 10)))


def out_dir_for(cfg: BenchConfig, name: str) -> Path:
    return (Path(cfg.out) / name).resolve()


@lru_cache(maxsize=8)
def _prepared(path, modality, level, max_cells, seed, transform_, cofactor, normalize, batch_correct):
    ds = load_dataset(path, modality=modality, level=level)
    return transform(ds.subsample(max_cells, seed), transform_, cofactor, normalize, batch_correct)


def _describe(ds, cfg: BenchConfig) -> None:
    """One log block describing the data every method is about to see (size, balance, value range, spatial info)."""
    X = ds.X.to_numpy(np.float32)
    msg = [f"{ds.name}: {len(ds):,} cells x {len(ds.markers)} markers; values {np.nanmin(X):.3g}..{np.nanmax(X):.3g} (mean {np.nanmean(X):.3g}), "
           f"transform={cfg.transform}, modality={cfg.modality}"]
    if ds.y is not None:
        vc = pd.Series(ds.y).value_counts()
        msg.append(f"{len(vc)} {cfg.level} classes; largest {vc.index[0]} ({vc.iloc[0] / len(ds):.1%}), smallest {vc.index[-1]} ({vc.iloc[-1]:,} cells); "
                   f"{int((vc < 100).sum())} classes under 100 cells")
    msg.append(f"images: {len(np.unique(ds.groups)) if ds.groups is not None else 'none'}; spatial coordinates: {'yes' if ds.xy is not None else 'no'}")
    log.info("\n    ".join(msg))


def run_dataset(path: str, cfg: BenchConfig) -> dict[str, pd.DataFrame]:
    prep = _prepared if cfg.cache_data else _prepared.__wrapped__          # ponytail: cached datasets are shared, methods must not mutate them
    ds = prep(str(path), cfg.modality, cfg.level, cfg.max_cells, cfg.seed, cfg.transform, cfg.cofactor, cfg.normalize, cfg.batch_correct)
    X, n = ds.X.to_numpy(np.float32), len(ds)
    sup = cfg.mode == "supervised"
    _describe(ds, cfg)
    if sup and ds.y is None:
        raise ValueError(f"{ds.name}: supervised mode needs a '{cfg.level}' label column")
    try:
        if cfg.report:
            dataset_report(ds, cfg.out, cfg.level)
    except Exception:
        log.exception("%s: dataset report failed (continuing)", ds.name)

    def P(method: str, explicit: bool = False) -> dict:
        """The method's parameters for this dataset: params file, then cfg.overrides (and the tacit shortcuts)."""
        ov = {**{k: v for k, v in (("r", cfg.tacit_r), ("p", cfg.tacit_p)) if v is not None and method == "tacit"},
              **cfg.overrides.get(method, {})}
        return mparams.resolve(ds.name, method, ov, cfg.params_file, explicit)

    mpath = cfg.marker_matrix or find_marker_matrix(ds.name, cfg.level)
    auto_prior = False
    needs_matrix = (not sup) or any(w.removesuffix("+vote") in PRIOR_METHODS for w in cfg.methods)
    if mpath is None and cfg.auto_matrix and needs_matrix and ds.y is not None:
        mpath = out_dir_for(cfg, ds.name) / "auto_marker_matrix.csv"
        mpath.parent.mkdir(parents=True, exist_ok=True)
        draft_matrix(X, ds.markers, ds.y).to_csv(mpath, float_format="%d")
        auto_prior = True
        log.warning("[%s] no marker decision matrix was given or bundled, so one was created from the dataset's own labels: %s. "
                    "TACIT / Scyan / Astir / marker_score results that use it are circular (not independent prior knowledge). "
                    "Pass --marker-matrix to use a real one, or --no-auto-matrix to skip those methods.", ds.name, mpath)
    matrix = load_marker_matrix(mpath) if mpath else None
    if matrix is not None:           # published signature files use raw channel names ("CD31 - vasculature:Cyc_19_ch_3"): map them to this dataset's markers
        matrix, dropped = conform_matrix(matrix, ds.markers)
        per = (matrix != 0).sum(1)
        log.info("[%s] marker matrix %s: %d cell types x %d dataset markers matched (markers per type %d..%d, %d negative entries)%s",
                 ds.name, Path(mpath).name, len(matrix), matrix.shape[1], per.min(), per.max(), int((matrix < 0).sum().sum()),
                 f"; {len(dropped)} matrix column(s) match no dataset marker and are ignored: {', '.join(dropped)}" if dropped else "")
        unused = [m for m in ds.markers if m not in matrix.columns]
        if unused:
            log.info("[%s] markers not used by the matrix: %s", ds.name, ", ".join(unused))
        absent = sorted(set(np.unique(ds.y)) - set(matrix.index)) if ds.y is not None else []
        if absent:
            log.warning("[%s] cell types in the data that the matrix cannot predict: %s", ds.name, ", ".join(absent))
    pseudo = ok = None
    if matrix is not None:
        try:
            pseudo, ok = pseudo_labels(X, ds.markers, matrix, cfg.pseudo_min_score, cfg.pseudo_min_margin)
        except Exception as e:                       # matrix does not match the markers
            log.warning("%s: marker matrix unusable (%s)", ds.name, e)
            matrix = None
    if not sup and pseudo is None and ds.y is None:
        raise ValueError(f"{ds.name}: unsupervised mode needs a marker decision matrix (--marker-matrix) or a label column")

    out_dir = (Path(cfg.out) / ds.name).resolve()      # absolute: the repo scripts run from their own folders
    out_dir.mkdir(parents=True, exist_ok=True)
    if matrix is not None and ds.y is not None:       # does the signature describe the labelled cells? (explains the prior-knowledge methods' ceiling)
        try:
            ag = matrix_agreement(X, ds.markers, matrix, ds.y)
            ag.to_csv(out_dir / "matrix_agreement.csv", index=False)
            weak = ag[ag.pct_cells_passing_all < 50]
            log.info("[%s] signature vs labels: on average only %.0f%% of a type's cells have all its signature markers above average; "
                     "%d of %d types are under 50%% (%s). Full table: %s", ds.name, ag.pct_cells_passing_all.mean(), len(weak), len(ag),
                     ", ".join(f"{r.cell_type} {r.pct_cells_passing_all:.0f}% [weakest {r.weakest_marker} z={r.weakest_marker_mean_z:.2f}]"
                               for r in weak.itertuples()) or "none", out_dir / "matrix_agreement.csv")
        except Exception as e:                       # a diagnostic must never stop the run
            log.warning("[%s] signature diagnostic skipped: %s", ds.name, e)
    try:                                              # the exact parameters every method ran with (for the text report / reproduction)
        used = {m: P(m) for m in dict.fromkeys(w.removesuffix("+vote") for w in cfg.methods)}
        (out_dir / "params_used.json").write_text(json.dumps(used, indent=2, default=str), encoding="utf-8")
    except ValueError as e:
        log.warning("%s: parameters not recorded: %s", ds.name, e)
    if os.name == "nt" and len(str(out_dir.resolve())) > 110:
        log.warning("output path is long (%d chars); the repo's classic-ML script writes files ~130 chars deeper and "
                    "Windows stops at 260 - use a shorter --out", len(str(out_dir.resolve())))
    device, split = resolve_device(cfg.device), (cfg.split if sup else ("holdout" if cfg.split == "holdout" else "cv"))
    lab = ds.y if sup else (pseudo if pseudo is not None else ds.y)         # labels the methods see
    conf = np.ones(n, int) if (sup or ok is None) else ok.astype(int)
    extra = {"spc_confident": conf, **({} if ds.y is None or sup else {"gt_cell_type": ds.y})}
    if cfg.tune_on_validation and cfg.splits_dir in (None, "auto"):
        raise ValueError("tune_on_validation swaps each fold's test file for its validation file, so it needs a private splits_dir (the tuner sets one)")
    # a subsampled run (--max-cells) must not publish its folds next to the user's data: they would not match the full table
    store = resolve_store(cfg.splits_dir, path, ds.name, len(cfg.data) > 1) if sup and not (cfg.max_cells and cfg.splits_dir in (None, "auto")) else None
    ws_all = build_workspace(ds, lab, out_dir / "workspace", cfg.level, cfg.kfold_method, cfg.seed, cfg.folds,
                             val_pct=cfg.val_pct, make_folds=sup, extra=extra, store=store,
                             validation_split_as_test=cfg.tune_on_validation)
    ws_tr = ws_all if sup else None
    if not sup and (conf == 1).sum() >= cfg.folds * 10 and len(np.unique(lab[conf == 1])) >= 2:
        ws_tr = build_workspace(ds, lab, out_dir / "workspace_pseudo", cfg.level, cfg.kfold_method, cfg.seed, cfg.folds,
                                val_pct=cfg.val_pct, rows=np.where(conf == 1)[0], extra=extra,       # semi-supervised: confident cells only
                                validation_split_as_test=cfg.tune_on_validation)
    folds = ws_tr.folds() if ws_tr else []
    jobs: list[dict] = []
    if folds:
        if split in ("holdout", "all"):
            jobs.append(dict(split="holdout", fold=0, fraction=NAN, tr=folds[0]["train"], te=folds[0]["test"]))
        if split in ("cv", "all"):
            jobs += [dict(split="cv", fold=i, fraction=NAN, tr=f["train"], te=f["test"]) for i, f in enumerate(folds)]
        if split in ("progressive", "all"):
            jobs += [dict(split="progressive", fold=0, fraction=fr, tr=sub, te=folds[0]["test"])
                     for fr, sub in progressive(folds[0]["train"], lab, n, cfg.fractions, cfg.seed)]

    greedy = _greedy()
    lv_true = level_labels(ds, cfg.level) if ds.y is not None else {}
    eval_levels = [lv for lv in LEVELS[LEVELS.index(cfg.level):] if lv in lv_true]
    n_classes = int(len(np.unique(lab)))

    # ---------------------------------------------------------------- scoring + saving of one prediction table
    def frame(rows, pred) -> pd.DataFrame:
        d = pd.DataFrame(dict(spc_row=rows, predicted_phenotype=pred, true_phenotype=lab[rows]))
        if ds.y is not None and not sup:
            d["gt_cell_type"] = ds.y[rows]
        d["spc_confident"] = conf[rows]
        if sup:
            for lv, y in lv_true.items():
                if lv != cfg.level:
                    d[LEVEL_COLUMN[lv]] = y[rows]
        if ds.xy is not None:
            d["x"], d["y"] = ds.xy[rows, 0], ds.xy[rows, 1]
        if ds.groups is not None:
            d["image"] = ds.groups[rows]
        return d

    def score(df: pd.DataFrame, key: dict):
        row, lvl_rows, cls_rows = {}, [], []
        tcol = "true_phenotype" if sup else ("gt_cell_type" if "gt_cell_type" in df else None)
        yp = df["predicted_phenotype"].fillna("undefined").astype(str).to_numpy()
        if tcol and ds.y is not None:
            yt0 = df[tcol].astype(str).to_numpy()
            for lv in eval_levels:
                col = LEVEL_COLUMN[lv]
                yt = yt0 if lv == cfg.level else (df[col].astype(str).to_numpy() if col in df else to_level(yt0, lv))
                yl = yp if lv == cfg.level else to_level(yp, lv)
                mt = supervised_metrics(yt, yl, level=lv)
                lvl_rows.append(dict(**key, eval_level=lv, **mt))
                if lv == cfg.level:
                    row.update(mt)
                    if "x" in df and df["x"].notna().all():
                        row.update(spatial_metrics(yt, yl, df[["x", "y"]].to_numpy(float), df["image"].to_numpy() if "image" in df else None, cfg.k_neighbors))
                    cls_rows = [dict(**key, **c) for c in per_class_table(yt, yl).to_dict("records")]
        if not sup and "spc_row" in df:
            r = df["spc_row"].to_numpy(int)
            ps = df["true_phenotype"].astype(str).to_numpy()
            row.update(unsupervised_qc(X[r], yp, ds.markers, matrix, ps, df["spc_confident"].to_numpy() == 1, cfg.seed,
                                       k=cfg.k_neighbors))
        return row, lvl_rows, cls_rows

    def save(df, name, job_split, fold, fraction) -> str:
        if job_split == "progressive":
            p = out_dir / "_progressive" / name / f"predictions_{fraction}.csv"
        else:
            tag = {"holdout": "holdout", "all": "all", "cv": f"cv{fold}"}[job_split]
            p = out_dir / name / cfg.level / f"predictions_{tag}.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        return str(p)

    # ---------------------------------------------------------------- one method -> list of record tuples
    def emit(name, impl, tier, split_, fold, fraction, run, df, imp, pred_file, n_train=NAN):
        key = dict(dataset=ds.name, method=name, split=split_, fold=fold, fraction=fraction)
        row = dict(**key, tier=tier, mode=cfg.mode, impl=impl, n_train=n_train, n_test=NAN if df is None else len(df),
                   status=run["status"], error=(run.get("error") or "").strip().splitlines()[-1][:1500] if run.get("error") else "",
                   runtime_s=run.get("runtime_s", NAN), mem_mb=run.get("mem_mb"), device=device, pred_file=pred_file or "",
                   prior_source=(AUTO_NOTE if auto_prior else "supplied matrix") if name.removesuffix("+vote") in PRIOR_METHODS else "")
        lv_rows, cl_rows = [], []
        if df is not None and run["status"] == "ok":
            r, lv_rows, cl_rows = score(df, key)
            row.update(r)
        im = [] if imp is None or split_ == "progressive" else [dict(**key, marker=m, importance=float(v)) for m, v in zip(ds.markers, imp)]
        log.info("[%s] %-24s %-11s fold=%s frac=%s -> %s %s | %.1fs", ds.name, name, split_, fold, fraction, run["status"],
                 " ".join(f"{k}={row[k]:.3f}" for k in ("f1_macro", "f1_weighted", "mcc", "accuracy", "unassigned_fraction") if k in row and row[k] == row[k])
                 or row["error"], run.get("runtime_s", NAN))
        return row, im, lv_rows, cl_rows

    def with_votes(recs_in, base, wanted):
        """Derive name+vote records from a base run's prediction table (spatial majority vote over the cells' neighbours)."""
        out = []
        for (split_, fold, fraction, run, df, imp, pfile, ntr, impl, tier) in recs_in:
            if base in wanted:
                out.append(emit(base, impl, tier, split_, fold, fraction, run, df, imp, pfile, ntr))
            if base + "+vote" in wanted:
                vname, vdf, vrun, vfile = base + "+vote", None, dict(run), ""
                if run["status"] == "ok":
                    if df is None or "x" not in df:
                        vrun.update(status="skipped", error="dataset has no x/y coordinates")
                    else:
                        vdf = df.copy()
                        vdf["predicted_phenotype"] = spatial_vote(df.predicted_phenotype.astype(str).to_numpy(),
                                                                  df[["x", "y"]].to_numpy(), df["image"].to_numpy() if "image" in df else None,
                                                                  cfg.k_neighbors)
                        vfile = save(vdf, vname, "all" if split_ == "all" else split_, fold, fraction) if cfg.save_predictions else ""
                out.append(emit(vname, impl, tier, split_, fold, fraction, vrun, vdf, None, vfile, ntr))
        return out

    def _run_method(base: str, wanted: set[str]):
        m = REGISTRY[base]
        skip = lambda why: [emit(nm, m.impl, m.tier, "not_run", 0, NAN, dict(status="skipped", error=why), None, None, "")
                            for nm in wanted if nm.removesuffix("+vote") == base]
        try:
            m.check()
        except MethodUnavailable as e:
            return skip(str(e).split(": ", 1)[-1])
        recs = []                                   # (split, fold, fraction, run, df, imp, pred_file, n_train, impl, tier)
        add = lambda *a: recs.append((*a, m.impl, m.tier))
        try:
            if m.impl == "classic":
                if not folds:
                    return skip("no training folds (needs labels or >= 2 pseudo-label classes)")
                units = [j for j in jobs if j["split"] in ("cv", "holdout", "progressive")]
                has_cv, has_ho = any(j["split"] == "cv" for j in units), any(j["split"] == "holdout" for j in units)
                # The hold-out IS cv fold 1, and a progressive fraction that covers the whole train fold is the same
                # training set again: train once and reuse that run for them (same data, same seed -> same model).
                src = "cv" if has_cv else ("holdout" if has_ho else None)
                full = [j for j in units if j["split"] == "progressive" and src and j["tr"] is folds[0]["train"]]
                runs = []
                if has_cv:
                    runs.append(("cv", lambda: ws_tr.root, None, None))
                for j in units:
                    if j["split"] == "holdout" and not has_cv:
                        runs.append(("holdout", lambda: make_variant(ws_tr, f"{base}_h", 0), None, None))
                    if j["split"] == "progressive" and not any(j is x for x in full):
                        runs.append(("progressive", lambda j=j: make_variant(ws_tr, f"{base}_p{j['fraction']}", 0, j["tr"]), j["fraction"], j))

                def twins(sp):
                    """The (split, fraction) records that share the fold-1 run of split sp instead of being trained again."""
                    return ([("holdout", NAN)] if sp == "cv" and has_ho else []) + ([("progressive", x["fraction"]) for x in full] if sp == src else [])

                for sp, root_fn, fr, j in runs:
                    log.info("[%s] %-24s classic-ML script, %s run%s", ds.name, base, sp, f" (fraction {fr})" if fr is not None else "")
                    try:
                        r = run_classic(base, root_fn(), ws_tr, cfg.seed, cfg.n_jobs_model, cfg.timeout, P(base, True), device)
                    except Exception as e:                                   # one failing unit must not drop the others
                        for sp2, fr2 in [(sp, fr if fr is not None else NAN), *twins(sp)]:
                            add(sp2, 0, fr2, dict(status="failed", error=f"{type(e).__name__}: {e}"), None, None, "", NAN)
                        continue
                    if r["status"] != "ok" or not r["files"]:
                        for sp2, fr2 in [(sp, fr if fr is not None else NAN), *twins(sp)]:
                            add(sp2, 0, fr2, dict(status=r["status"] if r["status"] != "ok" else "failed",
                                error=r["error"] or "no predictions written", runtime_s=r["runtime_s"]), None, None, "", NAN)
                        continue
                    for i, f in enumerate(r["files"]):
                        d = pd.read_csv(f)
                        sp_fold = i if sp == "cv" else 0
                        ntr = len(folds[i]["train"]) if sp == "cv" else (len(j["tr"]) if j else len(folds[0]["train"]))
                        run_ = dict(status="ok", runtime_s=r["fold_times"].get(i + 1, r["runtime_s"] / len(r["files"])))
                        imp = classic_importance(r["out"], base, i + 1, len(ds.markers)) if sp != "progressive" else None
                        add(sp, sp_fold, fr if fr is not None else NAN, run_, d, imp, str(f), ntr)
                        if i == 0:
                            for sp2, fr2 in twins(sp):
                                add(sp2, 0, fr2, run_, d, None if sp2 == "progressive" else imp, str(f), ntr)
            elif m.impl == "script":
                spec = m.spec
                ctx = ScriptCtx(ws_all, out_dir / ("_runs" if spec.nested else base) / (base if spec.nested else cfg.level),
                                cfg.script_runs, cfg.seed, device, n_classes,
                                Path(cfg.marker_matrix) if cfg.marker_matrix else (Path(mpath) if auto_prior else None),
                                has_area="area" in ds.meta, params=P(base, True))
                if spec.mode == "folds":                                  # e.g. MAPS: reads the repo's fold dir (cv only)
                    if not folds:
                        return skip("no training folds")
                    r = run_script(base, ctx, cfg.timeout)
                    for f in r["files"]:
                        add("cv", max(f["run"] - 1, 0), NAN, dict(status="ok", runtime_s=r["runtime_s"] / max(len(r["files"]), 1)),
                            pd.read_csv(f["path"]), None, str(f["path"]), NAN)
                    if not r["files"]:
                        add("cv", 0, NAN, dict(status=r["status"] if r["status"] != "ok" else "failed", error=r["error"] or "no predictions written",
                                              runtime_s=r["runtime_s"]), None, None, "", NAN)
                else:
                    r = run_script(base, ctx, cfg.timeout)
                    files = [f for f in r["files"] if f["level"] in (None, cfg.level)]
                    if r["status"] != "ok" or not files:
                        add("all", 0, NAN, dict(status=r["status"] if r["status"] != "ok" else "failed",
                            error=r["error"] or "no predictions written", runtime_s=r["runtime_s"]), None, None, "", NAN)
                    for f in files:
                        vdir = f["path"].parent.parent if f["level"] else f["path"].parent
                        t = parse_fold_times(vdir / "fold_times.txt").get(f["run"], r["runtime_s"] / max(len(files), 1))
                        d = pd.read_csv(f["path"], index_col=False)
                        if "cell_type" in d and "true_phenotype" not in d:
                            d = d.rename(columns={"cell_type": "true_phenotype"})
                        name = f["variant"].split("/")[-1] if f["variant"] else base
                        recs.append(("all", f["run"], NAN, dict(status="ok", runtime_s=t), d, None, str(f["path"]), NAN, "script", m.tier, name))
            else:                                                             # native
                recs += [(*r, m.impl, m.tier) for r in native_runs(base, m)]
        except MethodUnavailable as e:
            return skip(str(e).split(": ", 1)[-1])
        except Exception as e:                                                # never stop the batch
            log.exception("%s crashed", base)
            return [emit(nm, m.impl, m.tier, "not_run", 0, NAN, dict(status="failed", error=f"{type(e).__name__}: {e}"), None, None, "")
                    for nm in wanted if nm.removesuffix("+vote") == base]
        out = []
        for rec in recs:
            nm = rec[10] if len(rec) == 11 else base      # script variants: leiden_res0_5, flowsom_meta_clusters ...
            w = wanted if nm == base else {nm} | ({nm + "+vote"} if base + "+vote" in wanted else set())
            out += with_votes([rec[:10]], nm, w)
        return out

    def run_method(base: str, wanted: set[str]):
        """Every method gets the same progress lines: started (if it can run), a heartbeat, finished."""
        m = REGISTRY[base]
        try:
            m.check()
            runnable = True
        except MethodUnavailable:
            runnable = False                      # reported as 'skipped' by _run_method
        t0, stop = time.perf_counter(), threading.Event()
        if runnable:
            try:
                log.info("[%s] %-24s started (%s) device=%s threads=%s params=%s", ds.name, base, m.impl, device, cfg.n_jobs_model,
                         json.dumps(P(base), default=str)[:600])
            except ValueError:
                log.info("[%s] %-24s started (%s)", ds.name, base, m.impl)

            def heartbeat():
                while not stop.wait(300):
                    log.info("[%s] %-24s still running (%.0f min)", ds.name, base, (time.perf_counter() - t0) / 60)
            threading.Thread(target=heartbeat, daemon=True).start()
        try:
            return _run_method(base, wanted)
        finally:
            stop.set()
            if runnable:
                log.info("[%s] %-24s finished in %.0fs", ds.name, base, time.perf_counter() - t0)

    def native_runs(base, m):
        recs = []
        p = P(base)
        task_kw = dict(markers=ds.markers, xy=ds.xy, groups=ds.groups, marker_matrix=matrix, device=device, seed=cfg.seed,
                       n_jobs=cfg.n_jobs_model, k_neighbors=p.get("k_neighbors") or cfg.k_neighbors,
                       n_clusters=p.get("n_clusters") or n_classes, timeout=cfg.timeout, params=p)
        if m.trainable:
            if not folds:
                raise MethodUnavailable("no training folds (needs labels or >= 2 pseudo-label classes)")
            seen = {}                                            # same train/test arrays -> same run (hold-out = cv fold 1 = the full progressive fraction)
            for j in jobs:
                k = (id(j["tr"]), id(j["te"]))
                if k in seen:
                    log.info("[%s] %-24s %s fold=%s reuses the identical run already done", ds.name, base, j["split"], j["fold"])
                    recs.append((j["split"], j["fold"], j["fraction"], *seen[k], len(j["tr"])))
                    continue
                task = Task(X, train_idx=j["tr"], test_idx=j["te"], y_train=lab[j["tr"]], **task_kw)
                r = run_isolated(base, task, cfg.timeout)
                df, imp = None, None
                if r["status"] == "ok":
                    df = frame(j["te"], r["result"].labels.astype(str))
                    imp = r["result"].importance
                pf = save(df, base, j["split"], j["fold"], j["fraction"]) if (df is not None and cfg.save_predictions) else ""
                seen[k] = (r, df, imp, pf)
                recs.append((j["split"], j["fold"], j["fraction"], r, df, imp, pf, len(j["tr"])))
        else:                                                                  # clusterers / priors: one run over all cells
            allr = np.arange(n)
            task = Task(X, train_idx=allr[:0], test_idx=allr, y_train=lab, **task_kw)
            r = run_isolated(base, task, cfg.timeout)
            df = None
            if r["status"] == "ok":
                pred = r["result"].labels.astype(str)
                if m.kind == "cluster":
                    pred = np.asarray(greedy(pd.DataFrame(dict(t=lab, c=pred)), "t", "c")["mapped_predictions"]).astype(str)
                df = frame(allr, pred)
            pf = save(df, base, "all", 0, NAN) if (df is not None and cfg.save_predictions) else ""
            recs.append(("all", 0, NAN, r, df, None, pf, NAN))
        return recs

    wanted_all = list(dict.fromkeys(cfg.methods))
    bases = order_methods(list(dict.fromkeys(w.removesuffix("+vote") for w in wanted_all)), cfg.order)
    log.info("[%s] %d methods queued, %d at a time: %s", ds.name, len(bases), max(cfg.jobs, 1), ", ".join(bases))
    # Each method's rows go to disk the moment it finishes (<out>/<dataset>/_methods/<method>/<frame>.csv) and are not kept in
    # memory; the run-wide csvs are merged from those files. A 'done' marker means every row is ok / skipped, so --resume re-runs
    # only the failed / timed-out methods.
    mdir, lock = out_dir / "_methods", threading.Lock()
    if not cfg.resume:
        shutil.rmtree(mdir, ignore_errors=True)

    def merged() -> dict[str, pd.DataFrame]:
        got = {k: [pd.read_csv(f) for b in bases if (f := mdir / b / f"{k}.csv").exists()] for k in FRAMES}
        return {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in got.items()}

    def run_and_dump(b: str) -> None:
        d = mdir / b
        if cfg.resume and (d / "done").exists():
            log.info("[%s] %-24s skipped: results already saved (--resume)", ds.name, b)
            return
        outs = run_method(b, {w for w in wanted_all if w.removesuffix("+vote") == b})
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        for i, k in enumerate(FRAMES):
            rows = [o[0] for o in outs] if i == 0 else [r for o in outs for r in o[i]]
            if rows:
                pd.DataFrame(rows).to_csv(d / f"{k}.csv", index=False)
        if all(o[0]["status"] in ("ok", "skipped") for o in outs):
            (d / "done").write_text("")
        if len(cfg.data) == 1:                                   # keep the run-wide csvs current, so `visualize` works on a partial run
            with lock:
                for k, v in merged().items():
                    v.to_csv(Path(cfg.out) / FILES[k], index=False)

    with ThreadPoolExecutor(max(cfg.jobs, 1)) as ex:
        list(ex.map(run_and_dump, bases))
    return merged()


def run_benchmark(cfg: BenchConfig) -> pd.DataFrame:
    root = Path(cfg.out)
    root.mkdir(parents=True, exist_ok=True)
    allf: dict[str, list] = {k: [] for k in FRAMES}
    for p in cfg.data:
        try:
            for k, v in run_dataset(p, cfg).items():
                allf[k].append(v)
        except Exception as e:                         # a bad dataset must not stop the batch
            log.error("dataset %s failed: %s", p, e, exc_info=True)
    out = {}
    for k, lst in allf.items():
        out[k] = pd.concat(lst, ignore_index=True) if lst else pd.DataFrame()
        out[k].to_csv(root / FILES[k], index=False)
    return out["results"]

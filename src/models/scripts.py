"""Methods that run the repository's own scripts, and ingestion of the predictions they write.

Nothing about these methods is re-implemented: each entry only knows how to call the existing script
(``methods/classic_ml/run_classic_ml_default.py``, ``methods/leiden/run_leiden_clustering.py``,
``methods/scyan/run_scyan.py`` ...) with the arguments it already defines, on the dataset workspace written in
the repo's own layout, and where to find the ``predictions_*.csv`` it produces.

A script that is dataset-bound or image-based (CellSighter, STELLAR, Nimbus, DeepCellTypes) is registered with the
reason it cannot be driven from a quantification table; it is reported as ``skipped``.
Optional interpreter overrides (e.g. a conda env per method) go in ``configs/script_methods.json``.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from ..repo import METHODS, ROOT
from ..preprocessing.workspace import Workspace
from .base import MethodUnavailable, register_external
from .marker import find_bundled, find_marker_matrix, load_marker_matrix, to_astir_yaml

CONFIG = ROOT / "configs" / "script_methods.json"
CLASSIC_KWARGS = {"logistic_regression": "logistic_regression_model_kwargs_gridsearch.json",
                  "random_forest": "random_forest_model_kwargs.json", "xgboost": "xgboost_model_kwargs.json"}
CLASSIC_MODELS = ["logistic_regression", "random_forest", "xgboost", "most_frequent", "stratified"]


@dataclass
class ScriptCtx:
    ws: Workspace
    out: Path
    n_runs: int = 1
    seed: int = 0
    device: str = "cpu"
    n_clusters: int = 20
    marker_matrix: Path | None = None           # user-supplied decision matrix (csv)
    resolutions: tuple = (0.5, 0.8, 1.0, 2.0)
    has_area: bool = False

    @property
    def dataset(self) -> str:
        return self.ws.name


@dataclass
class ScriptSpec:
    script: str                                  # path under src/methods
    runtime: str                                 # python | Rscript
    build: Callable[[ScriptCtx], list[str]] | None
    mode: str = "all"                            # 'all' = one run over every cell | 'folds' = reads the fold dir (cv only)
    reason: str = ""                             # non-empty -> cannot be driven from a quantification table
    nested: bool = False                         # script writes <variant>/<level>/predictions_*.csv under its output dir
    name: str = ""

    def python_override(self) -> str | None:
        cfg = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
        return cfg.get(self.name, {}).get("python") if self.runtime == "python" else None

    def interpreter(self) -> str:
        cfg = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
        o = cfg.get(self.name, {})
        if self.runtime == "python":
            return o.get("python") or sys.executable
        return o.get("Rscript") or shutil.which("Rscript") or "Rscript"


# ----------------------------------------------------------------------------- marker files
def _csv_matrix(ctx: ScriptCtx, prefer: str) -> Path:
    p = ctx.marker_matrix or find_marker_matrix(ctx.dataset, ctx.ws.level, prefer)
    if p is None:
        raise MethodUnavailable(f"needs a decision matrix: pass --marker-matrix (none bundled for '{ctx.dataset}')")
    return Path(p)


def _astir_yaml(ctx: ScriptCtx) -> Path:
    if ctx.marker_matrix is None:
        y = find_bundled(ctx.dataset, "astir", "cell_types_{ds}.yml")
        if y:
            return y
    m = ctx.marker_matrix or find_marker_matrix(ctx.dataset, ctx.ws.level)
    if m is None:
        raise MethodUnavailable(f"needs a marker file: no astir yml bundled for '{ctx.dataset}' and no --marker-matrix")
    return to_astir_yaml(load_marker_matrix(m), ctx.ws.root / "astir_markers.yml")


def _tribus_xlsx(ctx: ScriptCtx) -> Path:
    p = find_bundled(ctx.dataset, "tribus", "logic_table_{ds}_" + ctx.ws.level + ".xlsx")
    if p is None:
        raise MethodUnavailable(f"tribus needs a logic_table xlsx (bundled only for IMMUcan, cHL_2_MIBI), none for '{ctx.dataset}'")
    return p


# ----------------------------------------------------------------------------- command builders (the scripts' own CLIs)
def _leiden(c: ScriptCtx) -> list[str]:
    return ["-i", str(c.ws.quant), "-o", str(c.out), "-m", *c.ws.markers, "-it", str(c.n_runs), "-l", "short",
            "-r", *[str(r) for r in c.resolutions]]


def _flowsom(c: ScriptCtx) -> list[str]:
    return ["-i", str(c.ws.quant), "-m", *c.ws.markers, "-o", str(c.out), "-flow", str(c.n_clusters), "-it", str(c.n_runs)]


def _tacit(c: ScriptCtx) -> list[str]:
    return ["--input_path", str(c.ws.quant), "--decision_matrix_path", str(_csv_matrix(c, "TACIT")),
            "--separate_col", c.ws.split_col, "--output_path", str(c.out), "-n", str(c.n_runs)]


def _scyan(c: ScriptCtx) -> list[str]:
    return ["--dataset_path", str(c.ws.quant), "--split_col", c.ws.split_col, "--decision_matrix_path",
            str(_csv_matrix(c, "scyan")), "--granularity_level", c.ws.level, "--output_path", str(c.out),
            "--n_runs", str(c.n_runs), "--seed", str(c.seed), "--accelerator", "gpu" if c.device == "cuda" else "cpu"]


def _astir(c: ScriptCtx) -> list[str]:
    return ["--quant_path", str(c.ws.quant), "--decision_matrix_path", str(_astir_yaml(c)), "--separate_col",
            c.ws.last_marker, "--output_path", str(c.out), "--n_runs", str(c.n_runs), "--random_seed", str(c.seed),
            "--device", "cuda" if c.device == "cuda" else "cpu"]


def _tribus(c: ScriptCtx) -> list[str]:
    return ["--dataset_path", str(c.ws.quant), "--columns_to_use", ",".join(c.ws.markers), "--decision_matrix_path",
            str(_tribus_xlsx(c)), "--granularity_level", c.ws.level, "--output_path", str(c.out), "--n_runs",
            str(c.n_runs), "--seed", str(c.seed)]


def _starling(c: ScriptCtx) -> list[str]:
    a = ["--dataset_path", str(c.ws.quant), "--split_col_name", c.ws.split_col, "--output_path", str(c.out),
         "--n_runs", str(c.n_runs), "--seed", str(c.seed), "--k_cluster", str(min(c.n_clusters, 50))]
    return a if c.has_area else a + ["--model_cell_size", "N"]


def _maps(c: ScriptCtx) -> list[str]:
    return [str(c.ws.kdir), str(c.out), str(c.ws.labels)]


# name -> (tier, kind, spec, python modules required, executables required)
SPECS: dict[str, tuple] = {
    "leiden": (1, "cluster", ScriptSpec("leiden/run_leiden_clustering.py", "python", _leiden, nested=True), ("scanpy", "leidenalg", "anndata"), ()),
    "flowsom": (1, "cluster", ScriptSpec("FlowSOM/run_flowsom.R", "Rscript", _flowsom, nested=True), (), ("Rscript",)),
    "tacit": (2, "prior", ScriptSpec("TACIT/run_TACIT.R", "Rscript", _tacit), (), ("Rscript",)),
    "scyan": (2, "prior", ScriptSpec("scyan/run_scyan.py", "python", _scyan), ("scyan", "anndata"), ()),
    "astir": (2, "prior", ScriptSpec("astir/run_astir.py", "python", _astir), ("astir", "torch"), ()),
    "tribus": (2, "prior", ScriptSpec("tribus/run_tribus.py", "python", _tribus), ("tribus",), ()),
    "starling": (3, "cluster", ScriptSpec("starling/run_starling.py", "python", _starling), ("starling", "anndata", "torch"), ()),
    # NOTE: run_maps.py hard-codes max_epochs=2 and uses fold_i_test.csv as its *training* file (see the script)
    "maps": (2, "supervised", ScriptSpec("MAPS/run_maps.py", "python", _maps, mode="folds"), ("maps",), ()),
    "cellsighter": (2, "supervised", ScriptSpec("CellSighter/run_cellsighter.py", "python", None,
                    reason="image-based (needs images + segmentation masks), driven by its own json config"), (), ()),
    "stellar": (2, "supervised", ScriptSpec("Stellar/run_stellar.py", "python", None,
                reason="dataset-bound (--dataset immucan|chl, reads images), not a quantification-table method"), (), ()),
    "nimbus": (3, "cluster", ScriptSpec("Nimbus/nimbus.py", "python", None,
               reason="image-based (needs multiplex tiffs + masks)"), (), ()),
    "deepcelltypes": (3, "prior", ScriptSpec("deepcelltypes/run_deepcelltypes.py", "python", None,
                      reason="image-based (needs images + masks + mpp)"), (), ()),
    "ribca": (3, "prior", ScriptSpec("", "python", None, reason="no implementation in the repository"), (), ()),
}


def register_all() -> None:
    for name, (tier, kind, spec, mods, exes) in SPECS.items():
        spec.name = name
        register_external(name, tier, kind, "script", mods, exes, spec, f"repo script {spec.script}" if spec.script else "")
    for m in CLASSIC_MODELS:
        register_external(m, 1 if m not in ("most_frequent", "stratified") else 1, "supervised", "classic",
                          ("xgboost", "sklearn"), (), None, "methods/classic_ml/run_classic_ml_default.py")


# ----------------------------------------------------------------------------- running
def _fail_status(rc: int) -> str:
    return "oom" if rc in (-9, 137, 3221225477) else "failed"


def _run(cmd: list[str], cwd: Path, timeout: float) -> dict:
    out = dict(status="ok", error="", runtime_s=float("nan"))
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout or None)
        if p.returncode:
            lines = [l.strip() for l in (p.stderr or p.stdout).splitlines() if l.strip()]
            out.update(status=_fail_status(p.returncode), error=f"exit {p.returncode}: " + " | ".join(lines[-4:])[:450])
    except subprocess.TimeoutExpired:
        out.update(status="timeout", error=f"exceeded {timeout:.0f}s")
    except FileNotFoundError as e:
        out.update(status="skipped", error=f"{e.filename or cmd[0]} not found")
    out["runtime_s"] = time.perf_counter() - t0
    return out


def parse_fold_times(path: Path) -> dict[int, float]:
    """Seconds per fold/run from a repo-style fold_times.txt ('Fold 3 training_time: 1.2' / 'Fold 3 inference_time, 4 seconds')."""
    t: dict[int, float] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            m = re.match(r"\s*Fold\s+(\d+)\D*?[:,]\s*([\d.]+)", line)
            if m:
                t[int(m.group(1))] = t.get(int(m.group(1)), 0.0) + float(m.group(2))
    return t


def run_classic(model: str, root: Path, ws: Workspace, seed: int, n_jobs: int, timeout: float) -> dict:
    """Run methods/classic_ml/run_classic_ml_default.py in its ``--main_dir`` mode on ``root`` (the workspace root or a
    single-fold variant root; both use the repo's datasets/<name>/quantification/processed layout). The script then
    writes ``<root>/results/<name>/<model>_default_<kfold>/<level>/`` with predictions_fold_<i>.csv, fold_times.txt and
    the pickled models."""
    spec = ScriptSpec("classic_ml/run_classic_ml_default.py", "python", None, name=model)
    marker_set = set(ws.markers)
    dumb = [c for c in pd.read_csv(ws.quant, nrows=0).columns if c not in marker_set]
    cmd = [spec.interpreter(), str(METHODS / spec.script), "--main_dir", str(root), "--model", model,
           "--kfold_method", ws.kfold_method, "--granularity_level", ws.level, "--random_state", str(seed),
           "--n_jobs_model", str(n_jobs), "--scaling", "Yes", "--dumb_columns", ",".join(dumb), "--verbose", "0"]
    kw = CLASSIC_KWARGS.get(model)
    if kw:
        cmd += ["--model_kwargs", str(METHODS / "classic_ml" / kw)]
    r = _run(cmd, METHODS / "classic_ml", timeout)
    out = root / "results" / ws.name / f"{model}_default_{ws.kfold_method}" / ws.level
    r["out"] = out
    r["files"] = sorted(out.glob("predictions_fold_*.csv"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    r["fold_times"] = parse_fold_times(out / "fold_times.txt")
    return r


def classic_importance(out: Path, model: str, fold: int, n_features: int):
    """Per-marker importance from the model pickle the repo script saved (feature_importances_ or |coef_|)."""
    import pickle
    f = out / "model" / f"{model}_fold_{fold}_model_default.pkl"
    if not f.exists():
        return None
    try:
        m = pickle.loads(f.read_bytes())
        if hasattr(m, "feature_importances_"):
            return m.feature_importances_
        if hasattr(m, "coef_"):
            return abs(m.coef_).mean(0)
    except Exception:
        return None
    return None


def run_script(name: str, ctx: ScriptCtx, timeout: float) -> dict:
    """Run one repo script; returns status/error/runtime and the predictions it wrote under ``ctx.out``."""
    spec: ScriptSpec = SPECS[name][2]
    if spec.reason:
        raise MethodUnavailable(f"{name}: {spec.reason}")
    ctx.out.mkdir(parents=True, exist_ok=True)
    script = METHODS / spec.script
    cmd = [spec.interpreter(), str(script), *spec.build(ctx)]
    r = _run(cmd, script.parent, timeout)
    r["files"] = collect_predictions(ctx.out)
    r["fold_times"] = {}
    return r


_LEVEL = re.compile(r"^level_?([123])$")


def collect_predictions(out: Path) -> list[dict]:
    """Every predictions*.csv under ``out``: {path, variant (sub-folder, e.g. leiden_res0_5), level, run}."""
    res = []
    for p in sorted(out.rglob("predictions*.csv")):
        parts = p.relative_to(out).parts[:-1]
        lv = next((f"level{_LEVEL.match(x).group(1)}" for x in parts if _LEVEL.match(x)), None)
        variant = "/".join(x for x in parts if not _LEVEL.match(x))
        nums = re.findall(r"\d+", p.stem)
        res.append(dict(path=p, variant=variant, level=lv, run=int(nums[-1]) if nums else 0))
    return res

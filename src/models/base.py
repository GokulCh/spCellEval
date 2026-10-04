"""Method registry, task/result containers and fault-isolated execution.

Every method is a function ``fn(Task) -> Result`` registered with ``@register``. ``run_isolated``
executes it in a child process with a hard timeout, so a hang, OOM kill or crash in one method never
stops the batch.
"""
from __future__ import annotations

import importlib.util
import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

TIERS = {1: "Traditional / Unsupervised", 2: "Marker / Prior-Knowledge",
         3: "Reference-Mapping & Foundation", 4: "Spatial / Graph-Aware"}


class MethodUnavailable(RuntimeError):
    """Raised when a method's package/config/input is missing; reported as status=skipped."""


@dataclass
class Task:
    X: np.ndarray                         # all cells x markers (already transformed)
    markers: list[str]
    train_idx: np.ndarray
    test_idx: np.ndarray
    y_train: np.ndarray | None            # labels (or marker pseudo-labels) for train_idx
    xy: np.ndarray | None = None
    groups: np.ndarray | None = None
    marker_matrix: pd.DataFrame | None = None   # decision matrix: rows = cell types, cols = markers (+1/-1)
    device: str = "cpu"                   # cpu | cuda
    seed: int = 0
    n_jobs: int = 1
    k_neighbors: int = 10
    n_clusters: int | None = None
    timeout: float = 0.0
    params: dict = field(default_factory=dict)

    @property
    def X_train(self): return self.X[self.train_idx]

    @property
    def X_test(self): return self.X[self.test_idx]


@dataclass
class Result:
    labels: np.ndarray                    # one label per test cell (cluster ids for kind="cluster")
    importance: np.ndarray | None = None  # per-marker importance, if the model exposes one


@dataclass
class Method:
    """impl: 'native' (function in src/models), 'classic' (repo's run_classic_ml_default.py) or 'script' (a repo run_*.py / .R)."""
    name: str
    fn: Callable[[Task], Result] | None
    tier: int
    kind: str                             # supervised | cluster | prior
    requires: tuple[str, ...] = ()
    needs_xy: bool = False
    description: str = ""
    impl: str = "native"
    executables: tuple[str, ...] = ()
    spec: object = None                   # ScriptSpec for impl == 'script'

    @property
    def trainable(self) -> bool:
        return self.kind == "supervised"

    def check(self) -> None:
        import shutil
        reason = getattr(self.spec, "reason", "")
        if reason:
            raise MethodUnavailable(f"{self.name}: {reason}")
        override = getattr(self.spec, "python_override", lambda: None)()
        missing = [] if override else [m for m in self.requires if importlib.util.find_spec(m) is None]
        if missing:
            raise MethodUnavailable(f"{self.name}: missing package(s) {', '.join(missing)}")
        for exe in self.executables:
            if shutil.which(exe) is None:
                raise MethodUnavailable(f"{self.name}: '{exe}' not found on PATH")

    def status(self) -> str:
        try:
            self.check()
            return "ready"
        except MethodUnavailable as e:
            return str(e).split(": ", 1)[-1]


REGISTRY: dict[str, Method] = {}


def register(name: str, tier: int, kind: str, requires: tuple[str, ...] = (), needs_xy: bool = False):
    def deco(fn):
        REGISTRY[name] = Method(name, fn, tier, kind, requires, needs_xy, (fn.__doc__ or "").strip().split("\n")[0])
        return fn
    return deco


def register_external(name: str, tier: int, kind: str, impl: str, requires=(), executables=(), spec=None,
                      doc: str = "") -> None:
    """Register a method that runs repository code (classic-ML script / run_*.py / R script)."""
    REGISTRY[name] = Method(name, None, tier, kind, tuple(requires), False, doc, impl, tuple(executables), spec)


def resolve_device(pref: str = "auto") -> str:
    """cpu | gpu | auto -> 'cpu' or 'cuda'."""
    if pref == "cpu":
        return "cpu"
    try:
        import torch
        ok = torch.cuda.is_available()
    except Exception:
        ok = False
    if pref == "gpu" and not ok:
        print("[warn] --device gpu requested but CUDA is unavailable; falling back to CPU")
    return "cuda" if ok else "cpu"


def select_methods(spec: list[str] | str | None) -> list[str]:
    """Names, tier keywords (tier1..tier4), 'trainable' or 'all' -> ordered method names.

    A ``+vote`` suffix on a name adds spatial majority voting post-processing.
    """
    if not spec or spec == "all" or spec == ["all"]:
        return sorted(REGISTRY, key=lambda n: (REGISTRY[n].tier, n))
    spec = spec.split(",") if isinstance(spec, str) else spec
    out: list[str] = []
    for s in (x.strip() for x in spec if x.strip()):
        if s.startswith("tier") and s[4:].isdigit():
            out += sorted((n for n, m in REGISTRY.items() if m.tier == int(s[4:])))
        elif s == "trainable":
            out += sorted(n for n, m in REGISTRY.items() if m.trainable)
        elif s.removesuffix("+vote") in REGISTRY:
            out.append(s)
        else:
            raise ValueError(f"unknown method '{s}'. Available: {', '.join(sorted(REGISTRY))}")
    return list(dict.fromkeys(out))


def run_inprocess(name: str, task: Task) -> Result:
    m = REGISTRY[name]
    m.check()
    if m.needs_xy and task.xy is None:
        raise MethodUnavailable(f"{name}: dataset has no x/y coordinates")
    return m.fn(task)


def _rss_mb() -> float | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except Exception:
        return None


def _child(q, name: str, task: Task) -> None:
    import src.models  # noqa: F401  (registers methods in the spawned interpreter)
    t0 = time.perf_counter()
    try:
        res = run_inprocess(name, task)
        q.put(("ok", res, time.perf_counter() - t0, _rss_mb()))
    except MethodUnavailable as e:
        q.put(("skipped", str(e), 0.0, None))
    except MemoryError:
        q.put(("oom", "MemoryError", time.perf_counter() - t0, None))
    except BaseException:
        q.put(("failed", traceback.format_exc(limit=4), time.perf_counter() - t0, None))


def run_isolated(name: str, task: Task, timeout: float = 0.0) -> dict:
    """Run one method; returns {status, result, runtime_s, mem_mb, error}.

    timeout > 0 runs in a killable child process (also survives OOM kills / segfaults);
    timeout == 0 runs in-process (debugging, tests).
    """
    out = dict(status="failed", result=None, runtime_s=np.nan, mem_mb=None, error="")
    if timeout <= 0:
        t0 = time.perf_counter()
        try:
            out.update(status="ok", result=run_inprocess(name, task), mem_mb=_rss_mb())
        except MethodUnavailable as e:
            out.update(status="skipped", error=str(e))
        except MemoryError:
            out.update(status="oom", error="MemoryError")
        except Exception:
            out.update(status="failed", error=traceback.format_exc(limit=4))
        out["runtime_s"] = time.perf_counter() - t0
        return out

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(q, name, task), daemon=True)
    t0 = time.perf_counter()
    p.start()
    msg = None
    while msg is None:
        try:
            msg = q.get(timeout=min(1.0, max(timeout - (time.perf_counter() - t0), 0.01)))
        except queue.Empty:
            if time.perf_counter() - t0 > timeout:
                out.update(status="timeout", error=f"exceeded {timeout:.0f}s", runtime_s=timeout)
                break
            if not p.is_alive():
                try:
                    msg = q.get(timeout=1.0)     # result may have landed just before exit
                except queue.Empty:
                    out.update(status="oom" if p.exitcode in (-9, 137, 3221225477) else "failed",
                               error=f"worker died (exit code {p.exitcode}; likely out of memory)",
                               runtime_s=time.perf_counter() - t0)
                    break
    if msg is not None:
        status, payload, rt, mem = msg
        out.update(status=status, runtime_s=rt, mem_mb=mem)
        out["result" if status == "ok" else "error"] = payload
    if p.is_alive():
        p.kill()
    p.join(5)
    return out

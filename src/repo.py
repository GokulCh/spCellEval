"""Access to the benchmark's existing code, so it is called rather than copied.

* ``load_module``       import a repo script by path (``methods/utils/run_kfold_creator.py`` ...)
* ``notebook_functions`` pull named function definitions out of an evaluation notebook and return the
                         real callables (the notebooks cannot be imported, but their functions are pure).
"""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
METHODS = SRC / "methods"
UTILS = METHODS / "utils"
EVAL = SRC / "evaluation"
PLOT = SRC / "plotting"


def load_module(path: str | Path, name: str):
    """Import a script file; its directory is put on sys.path so sibling imports (``from data_handler import``) work."""
    path = Path(path)
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=None)
def notebook_functions(notebook: str | Path, names: tuple[str, ...]) -> dict:
    """Return {name: function} for the named top-level ``def``s found in the notebook's code cells."""
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import jensenshannon
    from sklearn import metrics
    ns = dict(np=np, pd=pd, jensenshannon=jensenshannon, **{k: getattr(metrics, k) for k in dir(metrics) if not k.startswith("_")})
    nb = json.loads(Path(notebook).read_text(encoding="utf8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        try:
            tree = ast.parse("".join(cell["source"]))
        except SyntaxError:                       # cells with %magics / shell escapes
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names and node.name not in ns.get("_found", ()):
                exec(compile(ast.Module([node], []), str(notebook), "exec"), ns)
                ns.setdefault("_found", set()).add(node.name)
    missing = [n for n in names if n not in ns.get("_found", ())]
    if missing:
        raise ImportError(f"{notebook}: function(s) not found: {missing}")
    return {n: ns[n] for n in names}

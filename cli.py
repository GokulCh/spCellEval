"""spCellEval command line: guided full-pipeline wizard, or scripted sub-commands.

    python cli.py                       # interactive wizard (pick stages, answer prompts, confirm, run)
    python cli.py pipeline --data d.csv --stages analyze,preprocess,benchmark,visualize
    python cli.py pipeline --data raw.csv --stages convert,analyze,benchmark,visualize    # raw CODEX table -> converted first
    python cli.py convert --data raw.csv --out results/converted        # just the conversion (process_crc_codex.py)
    python cli.py pipeline --config results/pipeline/pipeline_config.json     # re-run a saved configuration
    python cli.py methods               # list all methods and their availability
    python cli.py analyze    --data d.csv        # dataset report only
    python cli.py preprocess --data d.csv --out results/prep
    python cli.py train      --data d.csv --methods xgboost,random_forest
    python cli.py benchmark  --data a.csv b.csv --split all --timeout 1800 --jobs 4
    python cli.py benchmark  --data d.csv --mode unsupervised --marker-matrix m.csv
    python cli.py visualize  --results results/benchmark   # re-run analysis + figures from saved results

Methods accept names, tier1..tier4, 'trainable', 'ready' (only runnable here) or 'all'; a +vote suffix adds
spatial majority voting.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
import traceback
from pathlib import Path

FRACTIONS = "0.01,0.05,0.10,0.25,0.50,0.80"
STAGES = ["convert", "analyze", "preprocess", "benchmark", "visualize"]
DEFAULT_STAGES = ["analyze", "preprocess", "benchmark", "visualize"]       # convert is only for RAW tables
STAGE_HELP = {"convert": "convert a RAW table with the repo's process_crc_codex.py (rename markers, drop label-leaking columns, arcsinh)",
              "analyze": "dataset analysis (composition, most/least common types, marker profiles, neighbourhoods)",
              "preprocess": "splits only: 5-fold CV + 80/20 hold-out + progressive training subsets, written in the repo's fold format (no methods run)",
              "benchmark": "model execution, training and multi-method benchmarking",
              "visualize": "result analysis, summary tables, insights and figures"}
DATA_SUFFIXES = (".csv", ".tsv", ".txt", ".parquet", ".h5ad")


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    from src.preprocessing.data import MODALITIES
    p = argparse.ArgumentParser(prog="spcelleval", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    p._subs = {}

    def common(sp, split_default, data_required=True):
        sp.add_argument("--data", nargs="+" if data_required else "*", required=data_required, default=[],
                        help="quantification tables or folders of them (csv/tsv/parquet/h5ad)")
        sp.add_argument("--out", default="results/benchmark", help="export directory")
        sp.add_argument("--mode", choices=["supervised", "unsupervised"], default="supervised",
                        help="supervised: splits + ground truth; unsupervised: marker pseudo-labels + QC metrics")
        sp.add_argument("--methods", default="all", help="comma list, 'all', 'ready', 'trainable' or tier1..tier4")
        sp.add_argument("--split", choices=["holdout", "cv", "progressive", "all"], default=split_default)
        sp.add_argument("--folds", type=int, default=5, help="CV folds (hold-out 80/20 = one fold of a stratified 5-fold split)")
        sp.add_argument("--kfold-method", choices=["StratifiedKFold", "StratifiedGroupKFold", "GroupShuffleSplit"],
                        default="StratifiedKFold", help="fold method of the repo's DataSetHandler")
        sp.add_argument("--fractions", default=FRACTIONS, help="progressive training fractions of the dataset")
        sp.add_argument("--splits-dir", help="folder that holds the folds. Default: the folder of the --data file (its processed/ folder). "
                                             "'out': keep them inside the run's own workspace. Existing folds are reused only if the data, "
                                             "labels and split settings match; otherwise new ones replace them. Your csv is never modified.")
        sp.add_argument("--script-runs", type=int, default=1, help="repeats for repo scripts that support n_runs (stability needs > 1)")
        sp.add_argument("--modality", choices=MODALITIES, default="codex")
        sp.add_argument("--transform", choices=["auto", "arcsinh", "log1p", "none"], default="auto")
        sp.add_argument("--cofactor", type=float, default=5.0)
        sp.add_argument("--normalize", choices=["none", "zscore", "minmax", "robust"], default="none")
        sp.add_argument("--batch-correct", choices=["none", "median"], default="none")
        sp.add_argument("--level", choices=["level1", "level2", "level3"], default="level3")
        sp.add_argument("--marker-matrix", help="decision matrix CSV (default: bundled one matching the dataset name)")
        sp.add_argument("--max-cells", type=int, default=0, help="random subsample for quick runs (0 = all)")
        sp.add_argument("--timeout", type=float, default=0, help="per-method seconds; >0 isolates each method in a killable process")
        sp.add_argument("--jobs", type=int, default=1, help="methods run concurrently")
        sp.add_argument("--device", choices=["cpu", "gpu", "auto"], default="auto")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--k-neighbors", type=int, default=10)
        sp.add_argument("--no-plots", action="store_true")

    def add(name, **kw):
        p._subs[name] = sub.add_parser(name, **kw)
        return p._subs[name]

    def convert_args(sp):
        sp.add_argument("--dataset-name", help="name for the converted dataset (single raw file; default: file name)")
        for opt in ("label-column", "cell-id", "image", "patient", "region", "x", "y"):
            sp.add_argument(f"--raw-{opt}", help=f"raw table's {opt.replace('-', ' ')} column (default: guessed from the header)")
        sp.add_argument("--raw-cofactor", type=float, default=5.0,
                        help="arcsinh cofactor applied by the converter (0 = keep raw intensities)")
        sp.add_argument("--raw-marker-regex", default=r":Cyc_\d+_ch_\d+$", help="regex that identifies marker columns")
        sp.add_argument("--raw-label-map", help="JSON {raw label: benchmark label}")
        sp.add_argument("--raw-hierarchy", help="JSON {cell_type: [level_2, level_1]}")

    sp = add("convert", help="convert a RAW quantification table with the repo's process_crc_codex.py")
    sp.add_argument("--data", nargs="+", required=True, help="raw tables (csv/tsv/xlsx/parquet)")
    sp.add_argument("--out", default="results/converted", help="main dir; writes datasets/<name>/quantification/processed/")
    convert_args(sp)

    sp = add("pipeline", help="chain stages: [convert ->] analyze -> preprocess -> benchmark -> visualize")
    common(sp, "all", data_required=False)
    convert_args(sp)
    sp.set_defaults(out=None)                     # None: results/pipeline, or "next to the data" for a splits-only run
    sp.add_argument("--stages", default=",".join(DEFAULT_STAGES),
                    help=f"comma list from {', '.join(STAGES)} (add 'convert' when --data is a raw table)")
    sp.add_argument("--config", help="JSON file written by an earlier pipeline run; command-line flags override it")
    common(add("train", help="train/run selected methods on a single split"), "holdout")
    common(add("benchmark", help="multi-method benchmark (hold-out + CV + progressive)"), "all")

    sp = add("preprocess", help="load, transform, write the repo-layout table and create folds with the repo's run_kfold_creator")
    sp.add_argument("--data", nargs="+", required=True)
    sp.add_argument("--out", default=None, help="folder for the splits (default: the folder that holds --data, i.e. processed/)")
    for a, kw in [("--modality", dict(choices=MODALITIES, default="codex")),
                  ("--transform", dict(choices=["auto", "arcsinh", "log1p", "none"], default="auto")),
                  ("--cofactor", dict(type=float, default=5.0)),
                  ("--normalize", dict(choices=["none", "zscore", "minmax", "robust"], default="none")),
                  ("--batch-correct", dict(choices=["none", "median"], default="none")),
                  ("--level", dict(choices=["level1", "level2", "level3"], default="level3")),
                  ("--folds", dict(type=int, default=5)), ("--fractions", dict(default=FRACTIONS)),
                  ("--splits-dir", dict(default=None, help="where the folds live (see `benchmark --help`)")),
                  ("--kfold-method", dict(choices=["StratifiedKFold", "StratifiedGroupKFold", "GroupShuffleSplit"], default="StratifiedKFold")),
                  ("--seed", dict(type=int, default=0))]:
        sp.add_argument(a, **kw)

    sp = add("analyze", help="dataset report: composition, rare/common types, marker profiles, neighbourhoods")
    sp.add_argument("--data", nargs="+", required=True)
    sp.add_argument("--out", default="results/benchmark")
    sp.add_argument("--modality", choices=MODALITIES, default="codex")
    sp.add_argument("--level", choices=["level1", "level2", "level3"], default="level3")
    sp.add_argument("--transform", choices=["auto", "arcsinh", "log1p", "none"], default="auto")
    sp.add_argument("--cofactor", type=float, default=5.0)

    sp = add("visualize", help="figures + summary tables from a results directory")
    sp.add_argument("--results", default="results/benchmark")
    add("methods", help="list methods, tiers and availability")
    return p


# ------------------------------------------------------------------ helpers
def expand_data(items) -> list[str]:
    """Files, folders (all tables inside) or comma-separated mixes -> list of existing files."""
    out: list[str] = []
    for raw in ([items] if isinstance(items, str) else items or []):
        for it in str(raw).split(","):
            it = it.strip().strip("'\"")
            if not it:
                continue
            p = Path(it).expanduser()
            if p.is_dir():
                found = sorted(f for f in p.iterdir() if f.suffix.lower() in DATA_SUFFIXES)
                if not found:
                    raise ValueError(f"no {'/'.join(DATA_SUFFIXES)} tables inside folder '{it}'")
                out += [str(f) for f in found]
            elif p.is_file():
                out.append(str(p))
            else:
                raise ValueError(f"'{it}' does not exist")
    return list(dict.fromkeys(out))


def method_status() -> dict[str, str]:
    """name -> 'ready' or the reason it cannot run here."""
    from src.models import REGISTRY
    return {n: m.status() for n, m in REGISTRY.items()}


def resolve_methods(spec: str) -> list[str]:
    from src.models import select_methods
    st = method_status()
    parts = [s.strip() for s in str(spec).split(",") if s.strip()]
    ready = sorted(n for n, s in st.items() if s == "ready")
    names = [x for s in parts for x in (ready if s == "ready" else [s])]
    return select_methods(names or "all")


# ------------------------------------------------------------------ stages
def cmd_methods(_) -> None:
    from src.models import REGISTRY, TIERS
    st = method_status()
    for tier in TIERS:
        print(f"\nTier {tier}: {TIERS[tier]}")
        for n, m in sorted(REGISTRY.items()):
            if m.tier == tier:
                print(f"  {n:<20} {m.kind:<10} {st[n]}")


def cmd_convert(a) -> list[str]:
    """Run the repo's process_crc_codex.py on each raw table; returns the processed quantification CSV paths."""
    from src.preprocessing import convert as cv
    files = a.data
    if a.dataset_name and len(files) != 1:
        raise ValueError("--dataset-name only applies to a single raw file")
    out = []
    for f in files:
        cols = cv.header(f)
        sug = cv.suggest(cols)
        opts = {k: getattr(a, "raw_" + k) or sug[k] for k in cv.OPTIONS}
        missing = [k for k, v in opts.items() if not v]
        if missing:
            raise ValueError(f"{Path(f).name}: cannot find the {', '.join(missing)} column(s); pass "
                             + " ".join(f"--raw-{m.replace('_', '-')} <column>" for m in missing)
                             + (" (the converter needs a label column; an unlabeled table cannot be converted)" if "label_column" in missing else ""))
        bad = [f"{k}='{v}'" for k, v in opts.items() if v not in cols]
        if bad:
            raise ValueError(f"{Path(f).name}: column(s) not in the table: {', '.join(bad)}")
        if cv.n_marker_columns(cols, a.raw_marker_regex) == 0:
            raise ValueError(f"{Path(f).name}: no column matches the marker pattern '{a.raw_marker_regex}'; set --raw-marker-regex")
        name = a.dataset_name or cv.dataset_name(f)
        print(f"converting {Path(f).name} -> dataset '{name}'  ({', '.join(f'{k}={v}' for k, v in opts.items())})")
        out.append(str(cv.run_converter(f, a.out, name, dict(opts, cofactor=a.raw_cofactor, marker_regex=a.raw_marker_regex,
                                                              label_map=a.raw_label_map, hierarchy=a.raw_hierarchy))))
    print("processed table(s): " + ", ".join(out))
    return out


def cmd_preprocess(a) -> None:
    """Splits only: the 5 folds (repo's run_kfold_creator), the 80/20 hold-out and the progressive training subsets.

    Where they go: --splits-dir if given, else --out if given, else the folder that holds the --data file
    (the repo's .../quantification/processed/). The data file itself is never modified.
    """
    import tempfile
    from src.preprocessing import build_workspace, export_variant, load_dataset, progressive, transform
    from src.preprocessing.workspace import resolve_store
    multi = len(a.data) > 1
    for path in a.data:
        ds = transform(load_dataset(path, modality=a.modality, level=a.level), a.transform, a.cofactor,
                       a.normalize, a.batch_correct)
        sd = getattr(a, "splits_dir", None)
        if sd and sd not in ("auto", "out"):
            target = resolve_store(sd, path, ds.name, multi)
        elif getattr(a, "out", None):
            target = Path(a.out) / ds.name if multi else Path(a.out)
        else:
            target = Path(path).resolve().parent
        if ds.y is None:
            print(f"{ds.name}: no label column ('{a.level}'), so there is nothing to split")
            continue
        with tempfile.TemporaryDirectory(prefix="spc_split_") as tmp:     # scratch: the working table is not kept
            ws = build_workspace(ds, ds.y, tmp, a.level, a.kfold_method, a.seed, a.folds, store=target)
            tag = f"{ws.kfold_method}_{ws.level}"
            folds = ws.folds()
            export_variant(ws, target / f"holdout_{tag}", 0)
            subsets = progressive(folds[0]["train"], ds.y, len(ds), [float(f) for f in a.fractions.split(",")], a.seed)
            for fr, sub in subsets:
                export_variant(ws, target / f"progressive_{tag}" / f"frac_{fr:g}", 0, sub)
        print(f"{ds.name}: {len(ds)} cells x {len(ds.markers)} markers. Splits are in {target}:")
        print(f"  {a.folds}-fold CV        kfolds_{tag}/  (+ labels_{tag}.csv)")
        print(f"  80/20 hold-out     holdout_{tag}/  (= fold 1)")
        print("  progressive        " + ", ".join(f"progressive_{tag}/frac_{fr:g}/ ({len(sub)} train cells)" for fr, sub in subsets))


def cmd_analyze(a) -> None:
    from src.evaluation import dataset_report
    from src.evaluation.plots import composition_plots
    from src.preprocessing import load_dataset, transform
    for path in a.data:
        ds = transform(load_dataset(path, modality=a.modality, level=a.level), a.transform, a.cofactor)
        r = dataset_report(ds, a.out, a.level)
        rep = Path(a.out) / ds.name / "dataset_report"
        composition_plots(rep, Path(a.out) / ds.name / "figures", ds.name)
        for lv, v in r.get("levels", {}).items():
            print(f"{ds.name} {lv}: {v['n_types']} types; most common {v['most_common']['cell_type']} "
                  f"({v['most_common']['percent']:.1f}%), least common {v['least_common']['cell_type']} "
                  f"({v['least_common']['percent']:.3f}%), rare: {v['rare_types']}")
        print(f"tables/figures in {rep.parent}")


def run_report(res, out: Path) -> str:
    """Per-method outcome table (ok / failed / timeout / oom / skipped runs) with the first error, saved and printed."""
    cols = ["ok", "failed", "timeout", "oom", "skipped"]
    t = res.groupby(["method", "status"]).size().unstack(fill_value=0).reindex(columns=cols, fill_value=0)
    t.insert(0, "outcome", ["OK" if r.failed + r.timeout + r.oom + r.skipped == 0 else
                            ("FAILED" if r.ok == 0 and r.skipped == 0 else "SKIPPED" if r.ok == 0 else "PARTIAL")
                            for r in t.itertuples()])
    err = res[res.status != "ok"].dropna(subset=["error"]).groupby("method").error.first()
    t["first_problem"] = t.index.map(lambda m: str(err.get(m, ""))[:160])
    t = t.sort_values(["outcome", "method"], key=lambda s: s.map({"OK": 0, "PARTIAL": 1, "SKIPPED": 2, "FAILED": 3}) if s.name == "outcome" else s)
    n = t.outcome.value_counts()
    txt = ("METHOD REPORT: " + ", ".join(f"{n.get(k, 0)} {k.lower()}" for k in ("OK", "PARTIAL", "SKIPPED", "FAILED"))
           + f"  ({len(t)} methods)\n" + t.to_string(max_colwidth=160) + "\n")
    (out / "summary").mkdir(exist_ok=True)
    (out / "summary" / "run_report.txt").write_text(txt, encoding="utf-8")
    t.to_csv(out / "summary" / "run_report.csv")
    return txt


def cmd_run(a) -> None:
    from src.evaluation import BenchConfig, run_benchmark, write_reports
    from src.evaluation.plots import make_all
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True,
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "benchmark.log", encoding="utf-8")])
    cfg = BenchConfig(
        data=a.data, out=a.out, methods=resolve_methods(a.methods), mode=a.mode, split=a.split, folds=a.folds,
        kfold_method=a.kfold_method, script_runs=a.script_runs, fractions=tuple(float(f) for f in a.fractions.split(",")),
        modality=a.modality,
        transform=a.transform, cofactor=a.cofactor, normalize=a.normalize, batch_correct=a.batch_correct,
        level=a.level, marker_matrix=a.marker_matrix, max_cells=a.max_cells, timeout=a.timeout, jobs=a.jobs,
        device=a.device, seed=a.seed, k_neighbors=a.k_neighbors, splits_dir=getattr(a, "splits_dir", None))
    res = run_benchmark(cfg)
    if res.empty:
        raise RuntimeError("no results produced - see benchmark.log")
    write_reports(a.out)
    print(f"\nstatus counts: {res.status.value_counts().to_dict()}")
    print("\n" + run_report(res, out))
    print(f"tables: {out / 'summary'}")
    if not a.no_plots:
        print(f"figures: {len(make_all(a.out))} files in {out / 'figures'} and per-dataset figures/")


def cmd_visualize(a) -> None:
    from src.evaluation import write_reports
    from src.evaluation.plots import make_all
    if not (Path(a.results) / "benchmark_results.csv").exists():
        raise FileNotFoundError(f"{a.results}/benchmark_results.csv not found - run the benchmark stage first")
    write_reports(a.results)
    print(f"{len(make_all(a.results))} figure files written under {a.results}")


def cmd_pipeline(a) -> None:
    """Run the chosen stages in order; a failing stage is reported and later independent stages still run."""
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad or not stages:
        raise ValueError(f"unknown stage(s) {bad}; choose from {STAGES}")
    stages = [s for s in STAGES if s in stages]                       # canonical order
    needs_data = [s for s in stages if s != "visualize"]
    if needs_data and not a.data:
        raise ValueError(f"stages {needs_data} need --data")
    out_explicit = a.out is not None
    splits_only = stages == ["preprocess"]
    out = Path(a.out or "results/pipeline")
    if not (splits_only and not out_explicit):          # a splits-only run without --out leaves no run artefacts
        out.mkdir(parents=True, exist_ok=True)
        (out / "pipeline_config.json").write_text(json.dumps({k: v for k, v in vars(a).items() if k not in ("cmd", "config")}, indent=2))
    ns = lambda **kw: argparse.Namespace(**{**vars(a), **{"out": str(out)}, **kw})

    def convert_stage():
        a.data = cmd_convert(ns(out=str(out / "converted")))          # later stages work on the processed tables
        if a.raw_cofactor and a.transform != "none":
            print(f"note: the converter already applied arcsinh(x/{a.raw_cofactor:g}); using --transform none so it is not applied twice")
            a.transform = "none"

    runners = {"convert": convert_stage,
               "analyze": lambda: cmd_analyze(ns()),
               "preprocess": lambda: cmd_preprocess(ns(out=str(out) if (out_explicit and "benchmark" not in stages) else None)),
               "benchmark": lambda: cmd_run(ns(no_plots=True)),
               "visualize": lambda: cmd_visualize(ns(results=str(out)))}
    report, failed = [], set()
    for s in stages:
        print(f"\n===== stage: {s} - {STAGE_HELP[s]} =====")
        if s == "visualize" and "benchmark" in failed:
            report.append((s, "skipped (benchmark failed)", 0.0))
            continue
        if s in ("analyze", "preprocess", "benchmark") and "convert" in failed:
            report.append((s, "skipped (convert failed)", 0.0))
            continue
        t0 = time.perf_counter()
        try:
            runners[s]()
            report.append((s, "ok", time.perf_counter() - t0))
        except Exception as e:
            traceback.print_exc()
            failed.add(s)
            report.append((s, f"FAILED: {e}", time.perf_counter() - t0))
    print("\n===== pipeline summary =====")
    for s, st, t in report:
        print(f"  {s:<11} {st}  ({t:.1f}s)")
    if splits_only and not out_explicit:
        print("splits saved next to your data (see above); nothing else was written")
    else:
        print(f"outputs: {out}   (config saved to {out / 'pipeline_config.json'})")
    rr = out / "summary" / "run_report.txt"
    if "benchmark" in stages and rr.exists():
        print("\n" + rr.read_text(encoding="utf-8"))
    if failed:
        raise SystemExit(1)


# ------------------------------------------------------------------ interactive wizard
def ask(q: str, default: str = "", validate=None) -> str:
    """Prompt; re-asks until validate(value) returns without raising ValueError."""
    while True:
        v = input(f"{q}{f' [{default}]' if default != '' else ''}: ").strip() or default
        try:
            return validate(v) if validate else v
        except ValueError as e:
            print(f"  ! {e}")


def choose(q: str, options: list[str], default: str, helps: dict | None = None) -> str:
    print(f"\n{q}")
    for i, o in enumerate(options, 1):
        print(f"  {i}) {o}" + (f"  - {helps[o]}" if helps and o in helps else "") + ("   (default)" if o == default else ""))

    def v(s):
        if s.isdigit() and 1 <= int(s) <= len(options):
            return options[int(s) - 1]
        if s in options:
            return s
        raise ValueError(f"enter a number 1-{len(options)} or one of: {', '.join(options)}")
    return ask("Choice", default, v)


def yes_no(q: str, default: bool = True) -> bool:
    def v(s):
        if s.lower() in ("y", "yes"):
            return True
        if s.lower() in ("n", "no"):
            return False
        raise ValueError("answer y or n")
    return ask(f"{q} (y/n)", "y" if default else "n", v)


def _number(kind, lo=None, hi=None):
    def v(s):
        try:
            x = kind(s)
        except ValueError:
            raise ValueError(f"enter a {kind.__name__}")
        if (lo is not None and x < lo) or (hi is not None and x > hi):
            raise ValueError(f"must be between {lo} and {hi}")
        return x
    return v


def peek(path: str) -> str:
    """One-line description of a table's columns so the user can sanity-check it before running."""
    import pandas as pd
    from src.preprocessing.data import LEVEL_COLUMN, normalize_columns
    if not path.lower().endswith((".csv", ".tsv", ".txt")):
        return "(not previewed)"
    df = normalize_columns(pd.read_csv(path, nrows=200, sep="\t" if path.lower().endswith((".tsv", ".txt")) else ","))
    labels = [lv for lv, c in LEVEL_COLUMN.items() if c in df.columns]
    return (f"{df.shape[1]} columns; labels: {', '.join(labels) or 'NONE (unsupervised only)'}; "
            f"spatial x/y: {'yes' if {'x', 'y'} <= set(df.columns) else 'no'}; image column: {'yes' if 'image' in df.columns else 'no'}")


def wizard() -> list[str]:
    from src.preprocessing.data import MODALITIES
    print("spCellEval - spatial cell-type annotation pipeline\n")
    top = choose("What would you like to do?",
                 ["Run pipeline (choose stages)", "Re-make analysis + figures from an existing results folder",
                  "Re-run a saved pipeline config", "List methods and availability", "Quit"],
                 "Run pipeline (choose stages)")
    if top == "Quit":
        raise SystemExit(0)
    if top == "List methods and availability":
        return ["methods"]
    if top.startswith("Re-make"):
        return ["visualize", "--results", ask("Results folder", "results/pipeline")]
    if top.startswith("Re-run"):
        def cfgv(s):
            if not Path(s).is_file():
                raise ValueError("file not found")
            return s
        return ["pipeline", "--config", ask("Path to pipeline_config.json", "results/pipeline/pipeline_config.json", cfgv)]

    print("\nStages:")
    for i, s in enumerate(STAGES, 1):
        print(f"  {i}) {s:<11} {STAGE_HELP[s]}")

    def stagesv(s):
        if s.lower() == "all":
            return DEFAULT_STAGES[:]                                   # convert is offered below when the data looks raw
        try:
            sel = [STAGES[int(x) - 1] if x.strip().isdigit() else x.strip() for x in s.split(",") if x.strip()]
        except IndexError:
            raise ValueError(f"stage numbers are 1-{len(STAGES)}")
        if not sel or any(x not in STAGES for x in sel):
            raise ValueError(f"choose numbers 1-{len(STAGES)}, names or 'all' (e.g. 2,4,5)")
        return [x for x in STAGES if x in sel]
    stages = ask("Which stages to run (comma-separated numbers, or 'all' = everything except convert)", "all", stagesv)
    argv = ["pipeline", "--stages", ",".join(stages)]
    need_data = any(s != "visualize" for s in stages)
    from src.preprocessing import convert as cv
    converted_names: list[str] = []

    def datav(s):
        files = expand_data(s)
        if not files:
            raise ValueError("enter at least one file or folder")
        return files
    files: list[str] = []
    if need_data:
        files = ask("\nDataset file(s) or folder(s), comma-separated", "", datav)
        for f in files:
            print(f"  {Path(f).name}: {peek(f)}")
        argv += ["--data", *files]
        # ---- raw table? offer the repo's process_crc_codex.py
        raw_like = [f for f in files if f.lower().endswith((".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet"))
                    and cv.looks_raw(cv.header(f))]
        do_convert = "convert" in stages
        if not do_convert and raw_like:
            print(f"\n  {', '.join(Path(f).name for f in raw_like)} looks like a RAW table "
                  f"({cv.n_marker_columns(cv.header(raw_like[0]))} columns match the '<marker> - ...:Cyc_<n>_ch_<n>' pattern).")
            print("  The pipeline needs the processed format (markers first, then image, cell_id, x, y, cell_type).")
            do_convert = yes_no("Convert it first with the repo's process_crc_codex.py?", True)
        cof = 0.0
        if do_convert:
            stages = [s for s in STAGES if s in set(stages) | {"convert"}]
            argv[2] = ",".join(stages)
            cols, sug = cv.header(files[0]), cv.suggest(cv.header(files[0]))
            if len(files) == 1:
                nm = ask("\nName for the converted dataset", cv.dataset_name(files[0]))
                argv += ["--dataset-name", nm]
                converted_names = [nm]
            else:
                converted_names = [cv.dataset_name(f) for f in files]
            print("\nMap the raw table's columns (Enter accepts the suggestion):")
            for opt, label in (("label_column", "cell-type label"), ("cell_id", "cell id"), ("image", "image / sample"),
                               ("patient", "patient"), ("region", "region"), ("x", "x coordinate"), ("y", "y coordinate")):
                def vcol(s):
                    if s not in cols:
                        raise ValueError(f"'{s}' is not a column of {Path(files[0]).name}")
                    return s
                argv += [f"--raw-{opt.replace('_', '-')}", ask(f"  {label} column", sug[opt] or "", vcol)]

            def regv(s):
                n = cv.n_marker_columns(cols, s)
                if n == 0:
                    raise ValueError("no column matches this pattern")
                print(f"    -> {n} marker columns")
                return s
            argv += ["--raw-marker-regex", ask("  regex identifying marker columns", cv.DEFAULT_REGEX, regv)]
            cof = ask("  arcsinh cofactor applied by the converter (0 = keep raw intensities)", "5", _number(float, 0))
            argv += ["--raw-cofactor", str(cof)]
            for flag, label in (("--raw-label-map", "label-map JSON {raw label: benchmark label}"),
                                ("--raw-hierarchy", "hierarchy JSON {cell_type: [level_2, level_1]}")):
                v = ask(f"  {label} (blank = none)", "", lambda s: s if (not s or Path(s).is_file()) else (_ for _ in ()).throw(ValueError("file not found")))
                if v:
                    argv += [flag, v]
        argv += ["--modality", choose("Data modality", MODALITIES, "codex"),
                 "--level", choose("Cell-type granularity to train/evaluate at", ["level3", "level2", "level1"], "level3",
                                   {"level3": "finest (subtypes)", "level1": "coarsest"})]
        if do_convert and cof > 0:
            tr = "none"
            print(f"\nTransform: none (the converter already applied arcsinh(x/{cof:g}), so it is not applied again)")
        else:
            tr = choose("Transform", ["auto", "arcsinh", "log1p", "none"], "auto",
                        {"auto": "arcsinh for protein, log1p for RNA"})
        argv += ["--transform", tr]
        if tr in ("auto", "arcsinh"):
            argv += ["--cofactor", str(ask("arcsinh cofactor", "5", _number(float, 1e-9)))]
        argv += ["--normalize", choose("Per-column normalization", ["none", "zscore", "minmax", "robust"], "none"),
                 "--batch-correct", choose("Batch correction", ["none", "median"], "none", {"median": "per-image median centring"})]
    if stages == ["preprocess"]:
        where = ask("\nSave the splits in (blank = next to your data file, i.e. its processed/ folder)", "")
        if where:
            argv += ["--out", where]
    else:
        if "preprocess" in stages:
            print("\nThe splits themselves go next to your data file (its processed/ folder); the folder below holds the run's other outputs.")
        argv += ["--out", ask("Export directory", "results/pipeline")]

    if "benchmark" in stages:
        mode = choose("Workflow mode", ["supervised", "unsupervised"], "supervised",
                      {"supervised": "train/val/test splits, 5-fold CV, progressive subsampling",
                       "unsupervised": "clustering + marker pseudo-labels, QC metrics without ground truth"})
        argv += ["--mode", mode]
        if mode == "supervised" and "preprocess" not in stages:
            sd = ask("Folder holding existing splits (blank = the folder of your data file, i.e. its processed/ folder)", "",
                     lambda s: s if (not s or Path(s).is_dir()) else (_ for _ in ()).throw(ValueError("folder not found")))
            if sd:
                argv += ["--splits-dir", sd]
        if mode == "supervised":
            split = choose("Split strategy", ["holdout", "cv", "progressive", "all"], "all",
                           {"holdout": "stratified 80/20 (= one fold of the 5-fold split)", "cv": "5-fold cross-validation",
                            "progressive": "train on 1%..80% of the data", "all": "all three"})
            argv += ["--split", split]
            argv += ["--kfold-method", choose("Fold method (repo DataSetHandler)", ["StratifiedKFold", "StratifiedGroupKFold"],
                                              "StratifiedKFold", {"StratifiedGroupKFold": "keeps each image in one fold"})]
            if split in ("cv", "all"):
                argv += ["--folds", str(ask("Number of CV folds", "5", _number(int, 2, 50)))]
            if split in ("progressive", "all"):
                def fracv(s):
                    try:
                        f = [float(x) for x in s.split(",")]
                    except ValueError:
                        raise ValueError("comma-separated numbers, e.g. 0.01,0.1,0.5")
                    if not f or any(not 0 < x <= 1 for x in f):
                        raise ValueError("fractions must be in (0, 1]")
                    return s
                argv += ["--fractions", ask("Training fractions", FRACTIONS, fracv)]
        else:
            from src.models.marker import find_marker_matrix
            names = converted_names or [Path(f).stem.replace("_quantification", "") for f in files]
            auto = [find_marker_matrix(nm, "level3") for nm in names]
            if any(auto):
                print("  bundled marker matrices found for: " + ", ".join(nm for nm, m in zip(names, auto) if m))

            def mmv(s):
                if s and not Path(s).is_file():
                    raise ValueError("file not found")
                return s
            mm = ask("Marker decision-matrix CSV for pseudo-labels (blank = bundled / none)", "", mmv)
            if mm:
                argv += ["--marker-matrix", mm]

        st = method_status()
        from src.models import REGISTRY, TIERS
        print("\nMethods:")
        for t in TIERS:
            names = sorted(n for n, m in REGISTRY.items() if m.tier == t)
            print(f"  tier{t} {TIERS[t]}: " + ", ".join(f"{n}{'' if st[n] == 'ready' else '*'}" for n in names))
        print("  (* = cannot run here: " + "; ".join(sorted({v for v in st.values() if v != 'ready'})) + ")")

        def methv(s):
            try:
                resolve_methods(s)
            except ValueError as e:
                raise ValueError(str(e)[:200])
            return s
        meth = ask("Methods: 'ready', 'all', tier1..tier4, 'trainable', or comma list", "ready", methv)
        if yes_no("Also run spatial majority-vote variants (name+vote) of the trainable methods?", False):
            meth += "," + ",".join(n + "+vote" for n in resolve_methods(meth) if REGISTRY[n].trainable and n not in ("most_frequent", "stratified"))
        argv += ["--methods", meth]
        argv += ["--device", choose("Hardware", ["cpu", "gpu", "auto"], "auto"),
                 "--timeout", str(ask("Per-method timeout in seconds (0 = none, no isolation)", "1800", _number(float, 0))),
                 "--jobs", str(ask("Methods to run in parallel", "1", _number(int, 1, 64))),
                 "--max-cells", str(ask("Subsample to N cells for a quick run (0 = all)", "0", _number(int, 0))),
                 "--script-runs", str(ask("Repeats for the repo's script methods (stability needs > 1)", "1", _number(int, 1, 50))),
                 "--seed", str(ask("Random seed", "0", _number(int)))]

    print("\n" + "=" * 60 + "\nSummary\n" + "=" * 60)
    print("  stages:", " -> ".join(stages))
    a = build_parser().parse_args(argv)
    for k, v in vars(a).items():
        if k in ("cmd", "config", "no_plots", "stages") or v in (None, []):
            continue
        print(f"  {k:<14} {str(len(v)) + ' file(s)' if k == 'data' else v}")
    print("\nEquivalent command:\n  python cli.py " + " ".join(f'"{x}"' if " " in x else x for x in argv))
    if not yes_no("\nRun now?", True):
        raise SystemExit(0)
    return argv


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    try:
        if not argv:
            argv = wizard()
        # --config: load saved settings as defaults, explicit flags still win
        if argv and argv[0] == "pipeline" and "--config" in argv:
            cfg_path = argv[argv.index("--config") + 1]
            cfg = json.loads(Path(cfg_path).read_text())
            parser._subs["pipeline"].set_defaults(**{k: v for k, v in cfg.items() if k != "cmd"})
        a = parser.parse_args(argv)
        if hasattr(a, "data") and a.cmd != "visualize":
            a.data = expand_data(a.data)               # ValueError -> "error: ..." and exit code 2 below
        if a.cmd is None:
            parser.print_help()
            return 0
        {"methods": cmd_methods, "preprocess": cmd_preprocess, "train": cmd_run, "benchmark": cmd_run,
         "visualize": cmd_visualize, "analyze": cmd_analyze, "pipeline": cmd_pipeline, "convert": cmd_convert}[a.cmd](a)
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return 130
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

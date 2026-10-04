"""spCellEval command line: guided full-pipeline wizard, or scripted sub-commands.

    python cli.py                       # interactive wizard (pick stages, answer prompts, confirm, run)
    python cli.py pipeline --data d.csv --stages analyze,preprocess,benchmark,visualize
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
STAGES = ["analyze", "preprocess", "benchmark", "visualize"]
STAGE_HELP = {"analyze": "dataset analysis (composition, most/least common types, marker profiles, neighbourhoods)",
              "preprocess": "preprocessing export (transformed table + 80/20, 5-fold CV and progressive split files)",
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

    sp = add("pipeline", help="chain stages: analyze -> preprocess -> benchmark -> visualize")
    common(sp, "all", data_required=False)
    sp.set_defaults(out="results/pipeline")
    sp.add_argument("--stages", default=",".join(STAGES), help=f"comma list from {', '.join(STAGES)}")
    sp.add_argument("--config", help="JSON file written by an earlier pipeline run; command-line flags override it")
    common(add("train", help="train/run selected methods on a single split"), "holdout")
    common(add("benchmark", help="multi-method benchmark (hold-out + CV + progressive)"), "all")

    sp = add("preprocess", help="load, transform, write the repo-layout table and create folds with the repo's run_kfold_creator")
    sp.add_argument("--data", nargs="+", required=True)
    sp.add_argument("--out", default="results/preprocessed")
    for a, kw in [("--modality", dict(choices=MODALITIES, default="codex")),
                  ("--transform", dict(choices=["auto", "arcsinh", "log1p", "none"], default="auto")),
                  ("--cofactor", dict(type=float, default=5.0)),
                  ("--normalize", dict(choices=["none", "zscore", "minmax", "robust"], default="none")),
                  ("--batch-correct", dict(choices=["none", "median"], default="none")),
                  ("--level", dict(choices=["level1", "level2", "level3"], default="level3")),
                  ("--folds", dict(type=int, default=5)),
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


def cmd_preprocess(a) -> None:
    from src.preprocessing import build_workspace, load_dataset, transform
    for path in a.data:
        ds = transform(load_dataset(path, modality=a.modality, level=a.level), a.transform, a.cofactor,
                       a.normalize, a.batch_correct)
        ws = build_workspace(ds, ds.y, Path(a.out) / ds.name / "workspace", a.level, a.kfold_method, a.seed, a.folds,
                             make_folds=ds.y is not None)
        print(f"{ds.name}: {len(ds)} cells x {len(ds.markers)} markers -> {ws.proc}"
              + (f"\n  folds + labels + validation sets (repo's run_kfold_creator): {ws.kdir.name}" if ws.has_folds else ""))


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
        device=a.device, seed=a.seed, k_neighbors=a.k_neighbors)
    res = run_benchmark(cfg)
    if res.empty:
        raise RuntimeError("no results produced - see benchmark.log")
    write_reports(a.out)
    print(f"\nstatus counts: {res.status.value_counts().to_dict()}")
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
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pipeline_config.json").write_text(json.dumps({k: v for k, v in vars(a).items() if k not in ("cmd", "config")}, indent=2))
    ns = lambda **kw: argparse.Namespace(**{**vars(a), **kw})
    runners = {"analyze": lambda: cmd_analyze(ns()),
               "preprocess": lambda: cmd_preprocess(ns(out=str(out / "preprocessed"))),
               "benchmark": lambda: cmd_run(ns(no_plots=True)),
               "visualize": lambda: cmd_visualize(ns(results=str(out)))}
    report, failed = [], set()
    for s in stages:
        print(f"\n===== stage: {s} - {STAGE_HELP[s]} =====")
        if s == "visualize" and "benchmark" in failed:
            report.append((s, "skipped (benchmark failed)", 0.0))
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
    print(f"outputs: {out}   (config saved to {out / 'pipeline_config.json'})")
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
            return STAGES[:]
        try:
            sel = [STAGES[int(x) - 1] if x.strip().isdigit() else x.strip() for x in s.split(",") if x.strip()]
        except IndexError:
            raise ValueError("stage numbers are 1-4")
        if not sel or any(x not in STAGES for x in sel):
            raise ValueError(f"choose numbers 1-4, names or 'all' (e.g. 1,3,4)")
        return [x for x in STAGES if x in sel]
    stages = ask("Which stages to run (comma-separated numbers, or 'all')", "all", stagesv)
    argv = ["pipeline", "--stages", ",".join(stages)]
    need_data = any(s != "visualize" for s in stages)

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
        argv += ["--modality", choose("Data modality", MODALITIES, "codex"),
                 "--level", choose("Cell-type granularity to train/evaluate at", ["level3", "level2", "level1"], "level3",
                                   {"level3": "finest (subtypes)", "level1": "coarsest"})]
        tr = choose("Transform", ["auto", "arcsinh", "log1p", "none"], "auto",
                    {"auto": "arcsinh for protein, log1p for RNA"})
        argv += ["--transform", tr]
        if tr in ("auto", "arcsinh"):
            argv += ["--cofactor", str(ask("arcsinh cofactor", "5", _number(float, 1e-9)))]
        argv += ["--normalize", choose("Per-column normalization", ["none", "zscore", "minmax", "robust"], "none"),
                 "--batch-correct", choose("Batch correction", ["none", "median"], "none", {"median": "per-image median centring"})]
    out_default = "results/pipeline"
    argv += ["--out", ask("\nExport directory", out_default)]

    if "benchmark" in stages:
        mode = choose("Workflow mode", ["supervised", "unsupervised"], "supervised",
                      {"supervised": "train/val/test splits, 5-fold CV, progressive subsampling",
                       "unsupervised": "clustering + marker pseudo-labels, QC metrics without ground truth"})
        argv += ["--mode", mode]
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
            auto = [find_marker_matrix(Path(f).stem.replace("_quantification", ""), "level3") for f in files]
            if any(auto):
                print("  bundled marker matrices found for: " + ", ".join(Path(f).name for f, m in zip(files, auto) if m))

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
         "visualize": cmd_visualize, "analyze": cmd_analyze, "pipeline": cmd_pipeline}[a.cmd](a)
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return 130
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

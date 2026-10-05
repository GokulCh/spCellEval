import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cli  # noqa: E402
from src.evaluation import repo_assets  # noqa: E402
from src.evaluation.metrics import per_class_table, supervised_metrics  # noqa: E402
from src.evaluation.repo_assets import category, hierarchical_f1_score, ancestors, scaling_score, to_level  # noqa: E402
from src.models import REGISTRY, Task, run_isolated  # noqa: E402
from src.models import scripts as scripts_mod  # noqa: E402
from src.models.base import register  # noqa: E402

HAS_XGB = importlib.util.find_spec("xgboost") is not None
HAS_SCANPY = importlib.util.find_spec("scanpy") is not None and importlib.util.find_spec("leidenalg") is not None
needs_xgb = pytest.mark.skipif(not HAS_XGB, reason="the repo's run_classic_ml_default.py imports xgboost")

# real labels from the repo hierarchy; M1_Macrophage is the most common and Endothelial the rarest
TYPES = {"M1_Macrophage": ("CD68", 300), "B_cell": ("CD20", 200), "CD4+_T_cell": ("CD4", 150),
         "Cancer": ("PanCK", 120), "Endothelial": ("CD31", 3)}
MARKERS = [m for m, _ in TYPES.values()] + ["Noise1", "Noise2"]
CLASSIC = "logistic_regression,random_forest,xgboost,most_frequent,stratified"


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("d")
    rng = np.random.default_rng(0)
    rows = []
    for img in ("i1", "i2"):
        for k, (t, (m, n)) in enumerate(TYPES.items()):
            X = rng.gamma(1.0, 1.0, (n, len(MARKERS))) * 2
            X[:, MARKERS.index(m)] += 60
            xy = rng.normal(k * 50, 8, (n, 2))
            df = pd.DataFrame(X, columns=MARKERS)
            df["image"], df["x"], df["y"], df["cell_type"] = img, xy[:, 0], xy[:, 1], t
            rows.append(df)
    p = d / "toy_quantification.csv"
    pd.concat(rows).to_csv(p, index=False)
    mm = pd.DataFrame(0, index=list(TYPES), columns=[m for m, _ in TYPES.values()])
    for t, (m, _) in TYPES.items():
        mm.loc[t, m] = 1
    mm.index.name = "Populations"
    mm.to_csv(d / "matrix.csv")
    return p, d / "matrix.csv"


@pytest.fixture(scope="module")
def full_run(data, tmp_path_factory):
    p, mm = data
    out = tmp_path_factory.mktemp("o")
    assert cli.main(["benchmark", "--data", str(p), "--out", str(out), "--marker-matrix", str(mm),
                     "--methods", f"{CLASSIC},svm,louvain,spade,singler,spatial_gnn,knn_smooth,marker_score,"
                                  "random_forest+vote,cellsighter",
                     "--split", "all", "--fractions", "0.05,0.5,0.8", "--jobs", "2"]) == 0
    return out


# ------------------------------------------------------------------ reuse of repo code
def test_metrics_match_sklearn_and_hand_values():
    yt = np.array(["a"] * 6 + ["b"] * 3 + ["c"])
    yp = np.array(["a"] * 5 + ["b"] + ["b"] * 2 + ["a"] + ["c"])
    m = supervised_metrics(yt, yp, rare_frac=0.15)
    assert m["f1_macro"] == pytest.approx(f1_score(yt, yp, average="macro"))
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["rare_accuracy"] == 1.0 and m["n_rare_classes"] == 1             # class c is 10% of cells
    pc = per_class_table(yt, yp).set_index("cell_type")
    assert pc.loc["a", "recall"] == pytest.approx(5 / 6) and pc.loc["a", "precision"] == pytest.approx(5 / 6)
    assert pc.loc["b", "specificity"] == pytest.approx(6 / 7)                  # b FP = 1 of 7 non-b cells
    perfect = supervised_metrics(yt, yt)
    assert perfect["jsd"] == 0 and perfect["jsd_scaled"] == 1 and perfect["mcc"] == 1


def test_repo_assets_and_notebook_functions_are_the_repos_own():
    assert scaling_score("xgboost") == 0.25 and scaling_score("tacit") == 0.75 and scaling_score("most_frequent") == 0
    assert scaling_score("leiden_res0_5") == 0.25
    assert category("random_forest+vote") == "Supervised" and category("leiden_res1_0") == "Unsupervised"
    assert to_level(["M1_Macrophage"], "level1")[0] == "Immune"
    assert hierarchical_f1_score(["B_cell"], ["B_cell"], ancestors()) == 1.0
    for fn in (repo_assets.hierarchical_f1_score, repo_assets.gmean_score, repo_assets.calculate_cell_type_distribution,
               repo_assets.calculate_r2_and_pearson, repo_assets.calculate_time):    # compiled from the notebook, not copied
        assert fn.__code__.co_filename.endswith("eval_mapping.ipynb")


def test_workspace_uses_repo_fold_creator(data, tmp_path):
    from src.preprocessing import build_workspace, load_dataset, transform
    ds = transform(load_dataset(data[0]))
    ws = build_workspace(ds, ds.y, tmp_path / "ws", seed=1)
    for f in ("fold_1_train.csv", "fold_1_validation.csv", "fold_1_test.csv", "fold_indices.json"):
        assert (ws.kdir / f).exists(), f                                      # written by run_kfold_creator / DataSetHandler
    assert ws.labels.name == "labels_StratifiedKFold_level3.csv" and ws.quant.exists()
    folds = ws.folds()
    assert len(folds) == 5
    tests = np.concatenate([f["test"] for f in folds])
    assert sorted(tests) == list(range(len(ds)))                              # every cell tested exactly once
    f0 = folds[0]
    assert abs(len(f0["test"]) / len(ds) - 0.2) < 0.01 and not set(f0["train"]) & set(f0["test"])
    assert not set(f0["train"]) & set(f0["validation"])
    cols = list(pd.read_csv(ws.quant, nrows=0).columns)
    assert cols[:len(ds.markers)] == ds.markers and cols[len(ds.markers)] == "spc_row"


# ------------------------------------------------------------------ end-to-end supervised
@needs_xgb
def test_supervised_outputs_and_analysis(full_run, data):
    out = full_run
    r = pd.read_csv(out / "benchmark_results.csv")
    ok = r[r.status == "ok"]
    assert set(ok[ok.method == "random_forest"].impl) == {"classic"} and set(ok[ok.method == "svm"].impl) == {"native"}
    assert ok[(ok.method == "random_forest") & (ok.split == "cv")].f1_macro.mean() > 0.9
    assert ok[(ok.method == "most_frequent") & (ok.split == "cv")].f1_macro.mean() < 0.3
    assert set(r[r.method == "cellsighter"].status) == {"skipped"}
    rep = (out / "summary" / "run_report.txt").read_text()                      # per-method outcome report
    assert rep.startswith("METHOD REPORT") and "cellsighter" in rep and "SKIPPED" in rep and "random_forest" in rep
    assert set(r.split) == {"cv", "holdout", "progressive", "all", "not_run"}     # --split all = 5-fold + 80/20 + progressive
    assert ok[(ok.split == "cv") & (ok.method == "xgboost")].fold.nunique() == 5
    ho = ok[(ok.split == "holdout") & (ok.method.isin(["xgboost", "random_forest", "svm"]))]
    assert set(ho.method) == {"xgboost", "random_forest", "svm"} and ho.n_test.between(300, 320).all()      # separate 80/20 runs
    bs = pd.read_csv(out / "analysis" / "by_split.csv")
    assert {"cv", "holdout", "progressive"} <= set(bs[bs.method == "random_forest"].split)
    assert not ok[ok.split == "progressive"].method.isin(["louvain", "spade", "marker_score"]).any()
    # classic ML results come straight from the repo script, in its own layout
    cdir = out / "toy" / "workspace" / "results" / "toy" / "random_forest_default_StratifiedKFold" / "level3"
    for f in ("predictions_fold_1.csv", "fold_times.txt", "average_rfc_results.json", "confusion_matrix_fold_1.csv"):
        assert (cdir / f).exists(), f
    fi = pd.read_csv(out / "feature_importance.csv")
    assert {"random_forest", "xgboost", "logistic_regression"} <= set(fi.method)     # read from the repo's pickled models
    # raw saves
    lm = pd.read_csv(out / "level_metrics.csv")
    assert set(lm.eval_level) == {"level3", "level2", "level1"} and lm.hierarchical_f1.notna().any()
    pc = pd.read_csv(out / "per_class_results.csv")
    assert {"precision", "recall", "specificity", "f1", "support"} <= set(pc.columns)
    pf = pd.read_csv(out / "toy" / "svm" / "level3" / "predictions_cv2.csv")
    assert {"true_phenotype", "predicted_phenotype", "x", "y", "image", "spc_row"} <= set(pf.columns)
    assert (out / "toy" / "_progressive" / "svm" / "predictions_0.05.csv").exists()
    # dataset report: most / least common
    rep = json.loads((out / "toy" / "dataset_report" / "summary.json").read_text())["levels"]["level3"]
    assert rep["most_common"]["cell_type"] == "M1_Macrophage" and rep["least_common"]["cell_type"] == "Endothelial"
    assert rep["rare_types"] == ["Endothelial"] and rep["n_types"] == 5
    # analysis tables + repo-format export
    a = out / "analysis"
    for f in ("final_results", "method_ranking", "class_difficulty", "top_confusions", "composition_recovery",
              "feature_importance_consensus", "data_scarcity", "efficiency", "spatial_gain", "level_comparison",
              "funky_heatmap_input", "failures"):
        assert (a / f"{f}.csv").exists(), f
    fr = pd.read_csv(out / "toy" / "final_results.csv", sep=";")
    assert {"level_3", "level_2", "level_1"} <= set(fr.level) and "overall_performance" in fr
    methods = set(fr.method)
    assert {"random_forest", "xgboost", "louvain", "marker_score", "svm"} <= methods     # 'all'-split methods are ranked too
    rf = fr[(fr.method == "random_forest") & (fr.level == "level_3")].iloc[0]
    assert rf.n_runs == 5 and 0 <= rf.stability <= 1
    ins = (a / "insights.md").read_text()
    assert "M1_Macrophage" in ins and "Endothelial" in ins and "Hardest cell types" in ins and "Best method" in ins
    for f in ("method_ranking", "per_class_f1_heatmap", "class_difficulty", "composition_recovery", "feature_importance",
              "abundance_level3", "composition_stacked_level3", "marker_zscores_by_celltype", "level_comparison"):
        assert (out / "toy" / "figures" / f"{f}.png").exists(), f
    assert list((out / "toy" / "figures" / "confusion").glob("*.png")) and list((out / "toy" / "figures" / "spatial").glob("*.png"))
    for f in ("overall_performance_heatmap", "performance_vs_runtime_scalability", "scaling_curves_f1_macro", "metric_correlation"):
        assert (out / "figures" / f"{f}.png").exists(), f
    assert cli.main(["visualize", "--results", str(out)]) == 0          # re-running analysis from saved files only


@needs_xgb
def test_holdout_is_one_stratified_fold(data, tmp_path):
    p, _ = data
    assert cli.main(["benchmark", "--data", str(p), "--out", str(tmp_path), "--split", "holdout",
                     "--methods", "random_forest,svm", "--no-plots"]) == 0
    r = pd.read_csv(tmp_path / "benchmark_results.csv")
    assert set(r.split) == {"holdout"} and (r.n_test.between(300, 320)).all()          # 20% of 1546 cells


# ------------------------------------------------------------------ repo script adapter
def _stub(tmp_path, name: str, body: str) -> Path:
    f = tmp_path / f"{name}.py"
    f.write_text("import sys, os, pandas as pd\nout = sys.argv[sys.argv.index('--output_path') + 1]\n"
                 "q = sys.argv[sys.argv.index('--dataset_path') + 1]\n" + body)
    return f


def _register_stub(monkeypatch, name, script, mode="all", nested=False):
    spec = scripts_mod.ScriptSpec(str(script), "python", lambda c: ["--dataset_path", str(c.ws.quant), "--output_path", str(c.out)],
                                  nested=nested, name=name)
    monkeypatch.setitem(scripts_mod.SPECS, name, (1, "cluster", spec, (), ()))
    monkeypatch.setitem(REGISTRY, name, REGISTRY["leiden"].__class__(name, None, 1, "cluster", (), False, "", "script", (), spec))


def test_script_adapter_ingests_variants_logs_failures_and_times_out(data, tmp_path, monkeypatch):
    p, _ = data
    ok = _stub(tmp_path, "ok", "import pandas as pd\nd = pd.read_csv(q)\nfor v in ('stub_a', 'stub_b'):\n"
                         "    for it in (1, 2):\n"
                         "        o = d.copy(); o['true_phenotype'] = d['cell_type']; o['predicted_phenotype'] = d['cell_type']\n"
                         "        os.makedirs(f'{out}/{v}/level3', exist_ok=True); o.to_csv(f'{out}/{v}/level3/predictions_{it}.csv', index=False)\n"
                         "    open(f'{out}/{v}/fold_times.txt', 'w').write('Fold 1 inference_time: 3.5\\nFold 2 inference_time: 4.5\\n')\n")
    bad = _stub(tmp_path, "bad", "sys.stderr.write('boom: bad input\\n'); sys.exit(3)\n")
    slow = _stub(tmp_path, "slow", "import time; time.sleep(60)\n")
    _register_stub(monkeypatch, "stub_ok", ok, nested=True)
    _register_stub(monkeypatch, "stub_bad", bad)
    _register_stub(monkeypatch, "stub_slow", slow)
    assert cli.main(["benchmark", "--data", str(p), "--out", str(tmp_path / "o"), "--split", "cv", "--no-plots", "--timeout", "20",
                     "--methods", "stub_ok,stub_bad,stub_slow,stub_ok+vote,svm"]) == 0
    r = pd.read_csv(tmp_path / "o" / "benchmark_results.csv")
    s = r[r.method == "stub_a"]
    assert set(s.status) == {"ok"} and len(s[s.method == "stub_a"]) == 2 and s.f1_macro.min() == 1.0
    assert s[(s.method == "stub_a") & (s.fold == 1)].runtime_s.iloc[0] == 3.5           # from the script's own fold_times.txt
    assert {"stub_a", "stub_b", "stub_a+vote", "stub_b+vote"} <= set(r.method)
    assert r[r.method == "stub_bad"].status.iloc[0] == "failed" and "boom" in r[r.method == "stub_bad"].error.iloc[0]
    assert r[r.method == "stub_slow"].status.iloc[0] == "timeout"
    assert (r[r.method == "svm"].status == "ok").all()                                   # batch carried on


@pytest.mark.skipif(not HAS_SCANPY, reason="scanpy + leidenalg not installed")
def test_real_leiden_script_runs_through_adapter(data, tmp_path):
    p, _ = data
    assert cli.main(["benchmark", "--data", str(p), "--out", str(tmp_path), "--split", "cv", "--methods", "leiden",
                     "--no-plots", "--timeout", "600"]) == 0
    r = pd.read_csv(tmp_path / "benchmark_results.csv")
    assert {"leiden_res0_5", "leiden_res0_8", "leiden_res1_0", "leiden_res2_0"} <= set(r.method)
    assert (r.status == "ok").all() and r.f1_macro.min() > 0.5 and r.split.eq("all").all()


# ------------------------------------------------------------------ unsupervised, unlabeled, levels
@needs_xgb
def test_unsupervised(data, tmp_path):
    p, mm = data
    assert cli.main(["benchmark", "--data", str(p), "--out", str(tmp_path), "--mode", "unsupervised",
                     "--marker-matrix", str(mm), "--methods", "random_forest,marker_score,louvain,spade"]) == 0
    r = pd.read_csv(tmp_path / "benchmark_results.csv")
    ok = r[r.status == "ok"].groupby("method").mean(numeric_only=True)
    assert {"random_forest", "marker_score", "louvain", "spade"} <= set(ok.index)
    assert ok.loc["marker_score", "pseudo_consistency"] > 0.8 and ok.loc["marker_score", "silhouette"] > 0.3
    assert (tmp_path / "toy" / "figures" / "unsupervised_qc.png").exists()


@needs_xgb
def test_unlabeled_table_and_derived_levels(data, tmp_path):
    p, mm = data
    nl = tmp_path / "nolabel_quantification.csv"
    pd.read_csv(p).drop(columns="cell_type").to_csv(nl, index=False)
    assert cli.main(["benchmark", "--data", str(nl), "--out", str(tmp_path / "u"), "--mode", "unsupervised",
                     "--marker-matrix", str(mm), "--methods", "marker_score,louvain", "--no-plots"]) == 0
    assert (tmp_path / "u" / "analysis" / "unsupervised_qc.csv").exists()
    # level2 requested but the table only has level3 labels -> derived through hierarchy_mappings.pkl
    assert cli.main(["benchmark", "--data", str(p), "--level", "level2", "--split", "holdout",
                     "--methods", "logistic_regression", "--out", str(tmp_path / "l2"), "--no-plots"]) == 0
    lm = pd.read_csv(tmp_path / "l2" / "level_metrics.csv")
    assert set(lm.eval_level) == {"level2", "level1"}


def test_dataset_only_analysis(data, tmp_path, capsys):
    assert cli.main(["analyze", "--data", str(data[0]), "--out", str(tmp_path)]) == 0
    assert "most common M1_Macrophage" in capsys.readouterr().out


def test_timeout_and_crash_do_not_stop_batch():
    @register("_boom", 1, "supervised")
    def boom(t):
        raise MemoryError

    t = Task(np.zeros((4, 2), np.float32), ["a", "b"], np.arange(2), np.arange(2, 4), np.array(["x", "y"]))
    try:
        r = run_isolated("svm", t, timeout=0.05)               # a child cannot even start in 50 ms
        assert r["status"] == "timeout" and r["runtime_s"] == 0.05
        assert run_isolated("_boom", t, timeout=0)["status"] == "oom"
    finally:
        REGISTRY.pop("_boom")


def test_isolated_ok(data):
    d = pd.read_csv(data[0])
    X = d[MARKERS].to_numpy(np.float32)
    idx = np.arange(len(d))
    t = Task(X, MARKERS, idx[::2], idx[1::2], d.cell_type.to_numpy()[::2])
    r = run_isolated("svm", t, timeout=120)
    assert r["status"] == "ok" and (r["result"].labels == d.cell_type.to_numpy()[1::2]).mean() > 0.9


# ------------------------------------------------------------------ CLI wizard / pipeline
def feed(monkeypatch, answers):
    it = iter(answers)

    def fake(prompt=""):
        try:
            a = next(it)
        except StopIteration:
            raise AssertionError(f"wizard asked more questions than scripted: {prompt!r}")
        print(f"{prompt}{a}")
        return a
    monkeypatch.setattr("builtins.input", fake)
    return it


def test_wizard_full_pipeline_with_reprompts(data, tmp_path, monkeypatch, capsys):
    p, _ = data
    out = tmp_path / "wiz"
    answers = ["1", "",                       # run pipeline, all stages
               "nope.csv", str(p),            # bad path is rejected, then valid
               "", "", "", "",                # modality, level, transform, cofactor
               "", "",                        # normalize, batch correction
               str(out),                      # export dir
               "", "1", "",                   # supervised, holdout, fold method
               "bogus", "logistic_regression,svm,most_frequent",     # bad method rejected, then valid
               "y",                           # add +vote variants
               "1", "0", "", "", "", "",      # cpu, no timeout, jobs, max-cells, script runs, seed
               ""]                            # confirm run
    feed(monkeypatch, answers)
    assert cli.main([]) == 0
    txt = capsys.readouterr().out
    assert "does not exist" in txt and "unknown method" in txt and "Equivalent command" in txt
    assert "pipeline summary" in txt and "FAILED" not in txt
    for f in ("pipeline_config.json", "benchmark_results.csv", "analysis/insights.md",
              "figures/performance_vs_runtime_scalability.png", "preprocessed/toy/workspace/datasets/toy/quantification/processed/labels_StratifiedKFold_level3.csv",
              "toy/dataset_report/summary.json", "toy/figures/abundance_level3.png"):
        assert (out / f).exists(), f
    r = pd.read_csv(out / "benchmark_results.csv")
    assert {"logistic_regression", "logistic_regression+vote", "svm", "svm+vote", "most_frequent"} == set(r.method)
    assert set(r.split) == {"holdout"}
    assert cli.main(["pipeline", "--config", str(out / "pipeline_config.json"), "--stages", "visualize"]) == 0


def test_pipeline_flags_stage_failure_continues_and_errors(data, tmp_path, capsys):
    p, mm = data
    with pytest.raises(SystemExit) as e:          # bad method -> benchmark fails, analyze still ran, visualize skipped
        cli.main(["pipeline", "--data", str(p), "--out", str(tmp_path), "--methods", "nonsense"])
    assert e.value.code == 1
    txt = capsys.readouterr().out
    assert "skipped (benchmark failed)" in txt and (tmp_path / "toy" / "dataset_report" / "summary.json").exists()
    assert cli.main(["pipeline", "--data", "missing.csv", "--out", str(tmp_path)]) == 2
    assert cli.main(["visualize", "--results", str(tmp_path / "empty")]) == 2


def test_wizard_menu_and_cancel(monkeypatch, capsys):
    feed(monkeypatch, ["4"])
    assert cli.main([]) == 0 and "Tier 1" in capsys.readouterr().out
    monkeypatch.setattr("builtins.input", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert cli.main([]) == 130


# ------------------------------------------------------------------ splits only / relative output paths
@needs_xgb
def test_splits_only_stage_and_relative_out_path(data, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["pipeline", "--data", str(data[0]), "--out", "rel/out", "--stages", "preprocess",
                     "--fractions", "0.05,0.5"]) == 0
    ws = tmp_path / "rel" / "out" / "preprocessed" / "toy" / "workspace"
    kd = "datasets/toy/quantification/processed/kfolds_StratifiedKFold_level3"
    assert (ws / kd / "fold_5_test.csv").exists()                                # 5-fold CV
    assert (ws / "v" / "h" / kd / "fold_1_test.csv").exists()                    # 80/20 hold-out
    small = pd.read_csv(ws / "v" / "p0.05" / kd / "fold_1_train.csv")
    big = pd.read_csv(ws / "v" / "p0.5" / kd / "fold_1_train.csv")
    assert 0 < len(small) < len(big) < len(pd.read_csv(ws / kd / "fold_1_train.csv")) + 1   # progressive subsets
    assert not (tmp_path / "rel" / "out" / "benchmark_results.csv").exists()     # no methods were run
    # a relative --out must work with the repo's classic-ML script (it runs from its own folder)
    assert cli.main(["benchmark", "--data", str(data[0]), "--out", "rel/o2", "--split", "holdout",
                     "--methods", "random_forest", "--no-plots"]) == 0
    r = pd.read_csv("rel/o2/benchmark_results.csv")
    assert (r.status == "ok").all() and r.f1_macro.iloc[0] > 0.9


@needs_xgb
def test_benchmark_reuses_existing_folds_and_rebuilds_when_settings_differ(data, tmp_path, capsys):
    common = ["--data", str(data[0]), "--modality", "codex", "--methods", "svm,random_forest", "--split", "cv"]
    # preprocess + benchmark in one pipeline: folds are created once, then linked
    assert cli.main(["pipeline", *common, "--stages", "preprocess,benchmark", "--out", str(tmp_path / "a")]) == 0
    txt = capsys.readouterr().out
    assert txt.count("5 folds created") == 1 and "reusing the folds already created" in txt
    k = tmp_path / "a" / "toy" / "workspace" / "datasets" / "toy" / "quantification" / "processed" / "kfolds_StratifiedKFold_level3"
    assert (k / "fold_3_test.csv").read_text() == (tmp_path / "a" / "preprocessed" / "toy" / "workspace" / "datasets" / "toy"
                                                    / "quantification" / "processed" / "kfolds_StratifiedKFold_level3" / "fold_3_test.csv").read_text()
    r = pd.read_csv(tmp_path / "a" / "benchmark_results.csv")
    assert (r.status == "ok").all() and set(r.method) == {"svm", "random_forest"}
    # a later, separate benchmark can point at the exported splits
    assert cli.main(["benchmark", *common, "--out", str(tmp_path / "b"), "--no-plots",
                     "--splits-dir", str(tmp_path / "a" / "preprocessed")]) == 0
    assert "reusing the folds already created" in capsys.readouterr().out
    # different seed -> fingerprint differs -> new folds, said out loud
    assert cli.main(["benchmark", *common, "--out", str(tmp_path / "c"), "--no-plots", "--seed", "7",
                     "--splits-dir", str(tmp_path / "a" / "preprocessed")]) == 0
    txt = capsys.readouterr().out
    assert "do not match this run (seed differ)" in txt and "5 folds created" in txt


# ------------------------------------------------------------------ raw table conversion (repo's process_crc_codex.py)
@pytest.fixture(scope="module")
def raw_table(data, tmp_path_factory):
    d = pd.read_csv(data[0])
    raw = pd.DataFrame({f"{m} - marker:Cyc_{i + 2}_ch_{i + 1}": d[m] * 10 for i, m in enumerate(MARKERS)})
    raw["HOECHST1:Cyc_1_ch_1"] = 5.0                                     # nuclear stain: must be dropped
    raw["CellID"], raw["File Name"] = np.arange(len(d)), d.image
    raw["patients"], raw["Region"] = d.image.map({"i1": "P1", "i2": "P2"}), "R1"
    raw["X"], raw["Y"], raw["ClusterName"] = d.x, d.y, d.cell_type
    raw["neighborhood"] = d.cell_type.astype("category").cat.codes       # label-derived: must not become a feature
    p = tmp_path_factory.mktemp("raw") / "TOY_expression.csv"
    raw.to_csv(p, index=False)
    return p


def test_convert_command_uses_repo_converter(raw_table, tmp_path):
    assert cli.main(["convert", "--data", str(raw_table), "--out", str(tmp_path)]) == 0
    out = tmp_path / "datasets" / "TOY" / "quantification" / "processed" / "TOY_quantification.csv"
    c = pd.read_csv(out)
    assert list(c.columns[:len(MARKERS)]) == MARKERS and {"image", "cell_id", "x", "y", "cell_type"} <= set(c.columns)
    assert "neighborhood" not in c and not any("HOECHST" in x for x in c.columns)      # leakage / stains dropped
    assert c.CD68.max() < 15                                                           # arcsinh(x / 5) applied by the converter
    assert (tmp_path / "datasets" / "TOY" / "quantification" / "processed" / "markers.txt").exists()
    assert cli.main(["convert", "--data", str(raw_table), "--out", str(tmp_path), "--raw-x", "nope"]) == 2   # bad column -> clear error


def test_pipeline_convert_stage_then_benchmark(raw_table, tmp_path, capsys):
    out = tmp_path / "p"
    assert cli.main(["pipeline", "--data", str(raw_table), "--out", str(out), "--stages", "convert,analyze,benchmark",
                     "--methods", "svm,most_frequent", "--split", "holdout"]) == 0
    txt = capsys.readouterr().out
    assert "stage: convert" in txt and "not applied twice" in txt and "FAILED" not in txt
    assert (out / "converted" / "datasets" / "TOY" / "quantification" / "processed" / "TOY_quantification.csv").exists()
    r = pd.read_csv(out / "benchmark_results.csv")
    assert set(r.dataset) == {"TOY"} and r[r.method == "svm"].f1_macro.iloc[0] > 0.9
    # an unconvertible raw table fails the convert stage and the dependent stages are skipped, not crashed
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(bad, index=False)
    with pytest.raises(SystemExit):
        cli.main(["pipeline", "--data", str(bad), "--out", str(tmp_path / "q"), "--stages", "convert,analyze"])
    assert "skipped (convert failed)" in capsys.readouterr().out


def test_wizard_offers_conversion_for_raw_table(raw_table, tmp_path, monkeypatch, capsys):
    out = tmp_path / "w"
    answers = ["1", "",                                   # run pipeline, all stages
               str(raw_table),
               "",                                        # 'Convert it first?' -> yes (default)
               "",                                        # dataset name
               "", "", "", "", "", "", "",                # label, cell id, image, patient, region, x, y (suggested)
               "", "",                                    # marker regex, cofactor
               "", "",                                    # label-map, hierarchy
               "", "",                                    # modality, level  (transform prompt is skipped)
               "", "",                                    # normalize, batch correction
               str(out),
               "", "1", "", "svm", "", "1", "0", "", "", "", "", "",    # mode, holdout, fold method, methods, vote, device, timeout, jobs, cells, runs, seed, confirm
               ]
    feed(monkeypatch, answers)
    assert cli.main([]) == 0
    txt = capsys.readouterr().out
    assert "looks like a RAW table" in txt and "Transform: none" in txt and "stage: convert" in txt and "FAILED" not in txt
    assert (out / "converted" / "datasets" / "TOY" / "quantification" / "processed" / "TOY_quantification.csv").exists()
    assert set(pd.read_csv(out / "benchmark_results.csv").method) == {"svm"}

# spCellEval unified framework

Nothing here is committed; all files are new (no existing file was modified).

The framework **orchestrates the repository's own code** and only adds what the repository does not have.

## What is reused vs. new

| Need | Repository code that is called | New code (only where nothing existed) |
|---|---|---|
| 5-fold CV, 80/20 hold-out, label encoding, validation split | `methods/utils/run_kfold_creator.py` -> `DataSetHandler` (hold-out = one fold of the stratified 5-fold split) | progressive 1-80 % subsampling (`preprocessing/splits.py`; repo only has the `training_amount/xgb_amount.ipynb` experiment) |
| Logistic regression, random forest, XGBoost, most-frequent / stratified baselines | `methods/classic_ml/run_classic_ml_default.py` (`ClassicMLDefault`) with the repo's `*_model_kwargs.json`; feature importance from the models it pickles | - |
| Leiden, FlowSOM | `methods/leiden/run_leiden_clustering.py`, `methods/FlowSOM/run_flowsom.R` (clusters mapped to labels inside the scripts with the repo's greedy mapping) | - |
| TACIT, Scyan, Astir, Tribus, Starling, MAPS | `run_TACIT.R`, `scyan/run_scyan.py`, `astir/run_astir.py`, `tribus/run_tribus.py`, `starling/run_starling.py`, `MAPS/run_maps.py` with the repo's bundled decision matrices / yml / xlsx | - |
| Metrics | the evaluation notebook's own functions, extracted at import time from `evaluation/eval_mapping.ipynb` (hierarchical F1, G-mean, composition distribution, R2/Pearson, timing parser); `hierarchy_mappings.pkl`, `cell_type_hierarchy.txt`; repo score weights, stability and scalability formulas; `scalability_score.json` | sensitivity / specificity / rare-vs-abundant accuracy, per-class table, unsupervised QC |
| Colours / figure styles | `plotting/method_colors.json`, `dataset_colors.json`, `funky_heatmap.R` input columns | the figures themselves (the repo's plots are notebook cells with hard-coded paths) |
| SVM, Louvain, SPADE, SingleR, scANVI, scArches, spatial GNN (SGC), kNN smoothing, spatial voting, marker-score pseudo-labelling | nothing in the repo | `src/models/{traditional,reference,spatial,marker}.py` |
| CellSighter, STELLAR, Nimbus, DeepCellTypes | scripts exist but are image-based or dataset-bound (cannot be driven from a quantification table) | registered, reported as `skipped` with the reason |
| RIBCA | no script in the repo | registered, `skipped` |

## Pipeline

```
load -> transform -> workspace (repo layout, folds from run_kfold_creator)
     -> classic ML scripts (cv / hold-out / progressive variants)   [split cv|holdout|progressive]
     -> repo method scripts (one run over all cells, as in the repo) [split 'all']
     -> native methods (only those with no repo counterpart)
     -> predictions_*.csv -> metrics -> raw tables -> analysis -> figures
```
`--timeout > 0` runs every script / native method in a killable process; timeouts, OOM and crashes become status rows and
the batch continues. Missing packages / Rscript / unavailable inputs become `skipped` with the reason.

## Usage

```bash
python cli.py                                   # guided wizard: pick stages, answer prompts, confirm, run
python cli.py pipeline --data d.csv --stages analyze,preprocess,benchmark,visualize --methods ready
python cli.py pipeline --config results/pipeline/pipeline_config.json   # re-run a saved configuration
python cli.py pipeline --data raw.csv --stages convert,analyze,preprocess,benchmark,visualize   # RAW table: converted first
python cli.py convert --data raw.csv --out results/converted    # only the conversion (src/preprocessing/datasets/process_crc_codex.py)
python cli.py methods                           # what is runnable here, and why not
python cli.py benchmark --data a.csv b.csv --split all --timeout 1800 --jobs 4
python cli.py benchmark --data d.csv --mode unsupervised --marker-matrix src/methods/scyan/cHL_CODEX_decision_matrix_level3.csv
python cli.py visualize --results results/benchmark
python -m pytest tests -q
```
**Raw tables.** The wizard detects a raw CODEX-style table (many `<marker> - ...:Cyc_<n>_ch_<n>` columns) and offers to run the
repo's `process_crc_codex.py` first (the `convert` stage): it suggests the script's column options from the file's header
(label, cell id, image, patient, region, x, y), you confirm or change them, and the processed
`datasets/<name>/quantification/processed/<name>_quantification.csv` feeds the later stages. The converter applies
arcsinh(x / cofactor) itself, so the pipeline switches its own transform to `none` to avoid transforming twice (use
`--raw-cofactor 0` to keep raw intensities and let the pipeline transform instead). The converter needs a label column; an
unlabeled raw table cannot be converted (run unsupervised mode on an already-processed table instead).

Interpreters per method (e.g. a conda env for scyan, a custom Rscript) go in `configs/script_methods.json`.
Keep `--out` short on Windows: the repo's classic-ML script writes files ~130 characters below it (260-character limit).

## Outputs under `--out`

Raw (written during the run; `visualize` re-runs the analysis from these alone): `benchmark_results.csv` (one row per
method x split x fold x fraction: status, runtime, metrics, `pred_file`), `level_metrics.csv` (level3/2/1 via the repo
hierarchy), `per_class_results.csv`, `feature_importance.csv`, `<dataset>/workspace/` (the repo's own
`datasets/<name>/quantification/processed/` layout, folds, labels and the classic-ML results incl. pickled models),
`<dataset>/<method>/<level>/predictions_*.csv`, `<dataset>/dataset_report/` (composition per level, rare types, marker
profiles, spatial neighbourhoods, `summary.json` with most/least common type).

Derived (`analysis/`): `final_results.csv` (+ `<dataset>/final_results.csv` in the repo's `;` format), `method_ranking`,
`class_difficulty`, `prevalence_vs_f1`, `top_confusions`, `composition_recovery`, `feature_importance_consensus`,
`data_scarcity`, `efficiency` (Pareto), `spatial_gain`, `level_comparison`, `unsupervised_qc`, `metric_correlation`,
`funky_heatmap_input.csv`, `failures`, `insights.md/json`. Figures (PNG + SVG) use the repo's colours.

Scores follow the repo: overall = 0.16 weighted F1 + 0.24 hierarchical F1 + 0.20 macro F1 + 0.20 MCC + 0.08 ARI + 0.12
(1-JSD); stability = 1 - std(weighted F1)/0.2 over folds; scalability = mean(`scalability_score.json` score, 1 - runtime/8 h).
The primary split per dataset is cv (else hold-out, else progressive); methods with split `all` are ranked alongside.

## Things found in the repo's own scripts (not changed)

* `MAPS/run_maps.py` sets `max_epochs = 2` and reads `fold_<i>_test.csv` as its *training* file and `fold_<i>_train.csv`
  as its test file. It is wired in as-is; check it before trusting MAPS numbers.
* Astir's `--separate_col` is the *last marker* (inclusive slice) while Scyan / TACIT / Starling take the first non-marker
  column; the adapter passes each what it expects.
* `calculate_time` (eval notebook) averages train+prediction times together for "inference"; per-run times here are taken
  from each script's own `fold_times.txt`.

## Verification status (honest)

Tested end to end on synthetic data (`tests/`, 15 tests): folds via the repo's fold creator, classic ML through the repo
script (CV, hold-out, progressive, pickled-model feature importance), the **real Leiden script** through the adapter, a stub
script (variants, failure, timeout), SVM / Louvain / SPADE / SingleR / spatial / marker-score natives, supervised and
unsupervised modes, unlabeled tables, level derivation, metrics vs hand values, the analysis tables, insights and figures,
and the CLI wizard.
**Not run here** (not installed): the R scripts (FlowSOM, TACIT - no Rscript), Scyan, Astir, Tribus, Starling, MAPS,
scANVI, scArches; the adapters pass them the arguments their own CLIs define and collect their `predictions_*.csv`, but
they are untested. Nothing has been run on the real benchmark datasets. `--device gpu` has never run.
Known limits: runtime for classic ML is the script's own fit+predict time; Harmony batch correction is not wired; the R
`funky_heatmap.R` is fed via `funky_heatmap_input.csv` but not run.

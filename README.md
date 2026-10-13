# <img alt="</> logotype" src="./website/plotly_figures/logo_example.png" style="height: 2em; vertical-align: middle;"> spCellEval - Benchmarking Cell Phenotyping Methods in Spatial Proteomics 

We present ["spCellEval"](https://huggingface.co/spaces/Arozhada/spcelleval), a quantitative comparison of automated/semi-automated cell phenotyping methods for Spatial Proteomics datasets on a diverse set of 13 curated public datasets. The methods are compared with a list of label transfer metrics divided into 4 categories: classification performance, distribution recovery, stability and scalability. This benchmark acts as a foundation to evaluate and improve automated cell phenotyping. 

![Alt text](img/fig_1.png "Title")

## Current Results Overview: 
![Alt text](img/fig_2.png "Title")

## Getting Started

In order to reproduce the results, the raw datasets currently need to be downloaded from public repositories. Please refer to the public registered Stage 1 manuscript. [IMMUCan](https://zenodo.org/records/12912567) is one example dataset. The downloaded datsets should be added as the following folder structure. 

```
spCellEval/
├── src/                          # Core benchmarking logic
│   ├── QC                        # Main script to QC datasets
│   ├── evaluation                # Main script to aggregate/collect results
│   ├── plotting                  # Main script to get plots
│   ├── preprocessing             # Main script to preprocess datasets
|   └── methods/                  # Each method gets its own subfolder
│       ├── method1/
│       │   ├── run_method.py     # Main script to run specified method     
│       │   └── ... (scripts, models, configs)
│       ├── method2/
│       └── ...
├── data/                        # To be downloaded from figshare
│   ├── raw/
│   └── processed/
├── results/                      # Results produced by methods or aggregation (See figshare)
│   ├── method1/
│   ├── method2/
│   └── summary/                  # to be downloaded from figshare
├── website/                      # Website code (HTML, CSS, JS, assets, deploy configs)
│   ├── index.html
│   ├── figures/
├── tests/                        # Automated tests for benchmarking and methods
├── environment.yml               # Dependencies
├── .gitignore
├── LICENSE
└── README.md

```

### Preprocessing

Preprocessing of each dataset can be found in `src/preprocessing/datasets/<process_dataset.ipynb>` Paths need to be adjusted.

For some datasets, multistack tiffs or channel_names have to be created. Please refer to `src/preprocessing/`

### Running methods

Scripts to run each method are provided in `src/<methods>/run_<method>.py`. For supervised method, create kfolds first using the `run_kfold_creator.py` file.

Datasets and parameter settings can be found in manuscript supplement.

For installation, and method specific details like runtime, please refer to each method's documentation.

The expected output from each method is a `predictions_*.csv` file for each fold chosen and a `fold_time.txt` recording running times if chosen.

#### Method parameters and automatic tuning

Every method's parameters are configurable per dataset in `configs/method_params.json` (the dataset key is the data file name without `_quantification`; a `"default"` block applies to all datasets; anything unset keeps the method's own default):

```json
{"default": {"leiden": {"resolutions": [0.8]}},
 "IMMUcan": {"xgboost": {"n_estimators": 250, "max_depth": 8}, "tacit": {"r": 50, "p": 20}}}
```

```bash
python cli.py params --dataset IMMUcan --init          # list all parameters/defaults/search values; write an editable IMMUcan block
python cli.py tune --data IMMUcan_quantification.csv --methods xgboost,leiden,tacit --trials 12   # search, then save the best values to the file
python cli.py benchmark --data IMMUcan_quantification.csv     # uses the saved values (--params FILE to use another file)
```

`tune` scores each candidate (grid in `src/models/params.py`) on the **validation** split of private folds (never the test split; `--tune-seed` makes them differ from the benchmark's), keeps the best `--metric` (default `f1_macro`) and writes it to the params file. Outputs: `tuning_log.csv` (every trial), `tuning_summary.csv` + `tuning_gain.png` (current vs tuned score). Speed-ups: `--jobs N` runs N trials at once, `--stage1-cells 20000` first scores all candidates on a 20k-cell subsample and re-scores only the best `--finalists` on the full data; the table is loaded once. `extra_args` passes raw CLI tokens to script-based methods for options not listed. `python cli.py params --schema` writes a JSON schema for editor checking of the params file.

Seed variability: `benchmark --repeats 5` repeats the whole run with 5 consecutive seeds and writes `summary/seed_variability.csv` (mean/std macro-F1); `summary/comparison_f1_macro.md` is a ready-to-paste methods x datasets table.

The shipped `configs/method_params.json` already holds best-judgement settings for every method (rationale per method under `_notes`). They are not tuned on your data: validate the file with `python cli.py params --check`, run `benchmark`, and refine with `tune`. Each run's report starts with a sanity-check section that flags methods with identical predictions, single-class predictions or mostly unassigned cells.

#### Results as text and as figures

Every benchmark run writes both. Nothing needs the images to be understood:

| file | what it is |
|---|---|
| `summary/AI_REPORT.md` | one markdown report with everything: run overview, the parameters every method ran with, leaderboards with ~20 metrics, per-level / per-class F1, hardest cell types, top confusions, composition bias, data scarcity, fold variability and paired tests, marker importance, text confusion matrices, and an index of all figures. Paste it into an AI model to review the results. |
| `summary/AI_REPORT_SHORT.md`, `AI_REPORT.json` | the leaderboards and insights only (a few screens) / the same leaderboards as records |
| `figures/**/NAME.txt` | a text twin of every figure `NAME.png/svg`: panel titles, axis labels and the plotted numbers (heatmap cells, bar heights, points, annotations) |
| `analysis/*.csv` | every table behind the figures (`metrics_long.csv` = every metric of every method in one tidy table) |
| `<dataset>/params_used.json` | the exact parameters each method used |

Metrics per run: accuracy, balanced accuracy, macro / weighted F1, macro / weighted precision and recall, MCC, Cohen's kappa, ARI, NMI, homogeneity, completeness, V-measure, Fowlkes-Mallows, G-mean, hierarchical F1, sensitivity / specificity, rare vs abundant accuracy and F1, unassigned fraction, classes never predicted, composition recovery (R², Pearson, JSD), spatial smoothness of the predictions, runtime, memory, stability, scalability. Figures include ranking, all-metrics and per-class heatmaps, metric ranks, fold variability with 95% CIs, precision vs recall, rare vs abundant, unassigned / missed classes, spatial smoothness, cluster agreement, paired tests against the best method, confusion matrices, spatial maps, composition recovery, data-scarcity curves, runtime / scalability, and cross-dataset ranks and spread. `python cli.py visualize --results <out>` rebuilds all of it from saved results.

### Evaluation Scripts
The notebooks in  `src/metrics_scripts` perform the evaluation over all methods. The code blocks withiin `eval_mapping.ipynb` goes through all the methods within a specified dataset and outputs a `final_results.csv` file that contains all of the metrics for different levels. 

## Adding your own method
To officially add your own method, please open an issue and provide us with the following to reproduce your method. 
1. GitHub repo for the method
2. List of Parameters used (if any)
3. OPTIONAL: Your predictions (this speeds up the evaluation process)

Folder Structure to add your predictions in 
```
results/
├── Dataset1/
│   ├── method1/
│   │    ├──predictions_*.csv
│   │    └──fold_times.txt
│   ├── method2/
│   ...
├── Dataset2/
└── Dataset3/
```


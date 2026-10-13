# spCellEval: technical report for this week

Dataset: CRC CODEX TMA (235,519 cells, 56 markers, 17 cell types, 140 images).
Results quoted here come from the partial run `crc_tma_v2` (its `summary/AI_REPORT.md`) and are compared with the first full run from last week (October 5).
"Verified" means I checked it in the code or in a report. "Inferred" means it is my explanation of a result that I have not tested directly. Both are marked where it matters.

---

## 1. The short version

1. **Last week** every method ran on fixed, partly hard-coded settings. Several scores were suspiciously low (MAPS 0.19, Scyan 0.05, Starling 0.09, Astir 0.17, TACIT 0.22) and nobody could say why.
2. **This week** I made every method configurable per dataset, added automatic tuning, many more metrics and figures (each with a plain-text version), a one-file AI-readable report, and a Slurm script that runs everything. I then audited the code and the data path.
3. **What the audit found:**
   - Real bugs or bad settings: MAPS (trained 2 epochs, on the wrong fold), Starling (stopped after about 3 epochs), Scyan (blank matrix cells passed as zero; CUDA crash on CPU runs), the drafted marker matrix (empty rows), scANVI (single-cell batch crash, random NaN).
   - Not bugs: preprocessing, labels, folds and scoring were all correct. The data matches the raw file cell by cell.
4. **Results this week:** MAPS fixed (0.194 to 0.792). XGBoost leads at 0.821 macro-F1. Supervised methods are 0.62 to 0.82. Label-free methods are 0.05 to 0.43. They stay low mainly because of the labels and the metric, not the code.
5. **Still open:** Scyan leaves 58% of cells "undefined" and I do not know why. Starling uses only 7 of 17 labels. scANVI and scArches have no results in this report (re-run pending, retry fix untested). Levels 2 and 1 have not been run with the new hierarchy. Tuning has not been run.

---

## 2. How a run works (so the rest makes sense)

```
quantification CSV  ->  preprocessing  ->  folds  ->  each method runs  ->  predictions  ->  metrics  ->  figures + reports
```

- **Data:** one table, one row per cell: 56 marker intensities, x/y position, image id, and the cell type label. The table in `data/datasets/crc_tma/quantification/processed/` is already transformed. I checked it against the raw file: arcsinh of the raw value divided by 5 matches, same 235,519 cells, same cell-type counts. The raw columns that could leak the answer (cluster ID, neighbourhood) are removed. No NaN or infinite values, minimum 0, maximum about 10.
- **Folds:** 5-fold, split by cell. The test fold is never used for training or tuning. See the caveats section: splitting by cell is optimistic.
- **Methods run in one of three ways:**
  - *Native* (`src/models/`): written in this repo (SVM, Louvain, SPADE, SingleR, scANVI, scArches, spatial GNN, kNN smooth, marker score).
  - *Script* (`src/methods/...`, wrapped by `src/models/scripts.py`): the repo's own scripts, such as XGBoost, random forest, logistic regression, Leiden, FlowSOM, TACIT, Scyan, Astir, Tribus, Starling, MAPS, STELLAR. They are launched as separate processes and the wrapper builds their command line from the parameters.
  - *Baselines*: most-frequent and stratified (random with the right proportions).
- **Supervised vs label-free** (this matters for every result below):
  - *Supervised*: learn from labelled training cells (XGBoost, random forest, logistic regression, SVM, scANVI, scArches, MAPS, STELLAR, spatial GNN, kNN smooth, SingleR). SingleR is grouped as prior-knowledge in the benchmark but builds its reference from the training labels, so it behaves like a supervised method.
  - *Unsupervised clustering* (Leiden, Louvain, FlowSOM, SPADE, Starling): they cluster cells without labels. To score them, each cluster is named by its most common true label (a greedy mapping). This is the benchmark's own rule. It makes their scores an upper bound.
  - *Prior-knowledge* (TACIT, Scyan, Astir, Tribus, marker score): they use a marker matrix saying which markers define each type. No training labels.
- **Marker matrices.** TACIT, Scyan and Astir each need a table of "cell type x marker".
  - Scyan format: +1 high, -1 low, blank = marker not relevant for that type.
  - TACIT format: 1 or blank (negatives are clipped to 0 in our wrapper).
  - Astir format: YAML listing the positive markers per type.
  - We have a hand-made matrix. We also have a "drafted" one built from the labels (circular, comparison only) and a "best 3 markers per type" one for the ceiling test.

---

## 3. What I built (and where)

### 3.1 Per-method, per-dataset configuration
- **File:** `configs/method_params.json`. Section `"default"` applies to every dataset. A section named after a dataset (its file name without `_quantification`) can override single parameters. Each method has a note explaining the value.
- **Precedence, lowest to highest:** defaults in code (`SPACE` in `src/models/params.py`) < the file's `"default"` < the dataset's section < per-run overrides.
- **`SPACE`** lists every parameter each method accepts, with its default and its tuning candidates. `python cli.py params` prints all of it. `python cli.py params --check` validates the file.
- **Reproducibility:** each run gets its own copy (`<out>/method_params.json`) and saves the values it used (`params_used.json`). The repo's file is never modified by a run.

### 3.2 Automatic tuning (`python cli.py tune`, code in `src/evaluation/tune.py`)
- For each dataset and method it generates candidate parameter sets from `SPACE` and scores each with the benchmark's own hold-out run on the **validation** split. The test split is never looked at, and tuning folds use a different seed (`tune_seed`), so a later benchmark number is honest.
- Two-stage search: every candidate is first scored on a subsample (`--stage1-cells`, default 20,000), and only the best 3 plus the current setting are re-scored on all cells.
- A tie keeps the current setting. A failed trial is logged and skipped. The best set is written back into the params file.
- Outputs: `tuning_log.csv`, `tuning_summary.csv` (current vs tuned) and `tuning_gain.png`.
- **Status: implemented, not run on the cluster.** So no value in the params file is tuned: they are best-judgement settings.

### 3.3 Metrics, tables, figures, reports (`src/evaluation/`)
- `metrics.py`: 20+ metrics including macro/weighted F1, accuracy, MCC, kappa, ARI, NMI, g-mean, balanced accuracy, precision/recall (macro and weighted), rare-type vs abundant-type F1, JSD and scaled JSD (composition distance), unassigned fraction, number of classes missed, homogeneity/completeness/V-measure, Fowlkes-Mallows.
- `extra_tables.py`, `plots_extra.py`: per-class table, top confusions, scarcity (training fraction) table, runtime/scalability, correlation of metrics.
- `figure_text.py`: writes a `.txt` twin for each figure (the numbers behind the picture) so it can be read without looking at it.
- `text_report.py`: writes `summary/AI_REPORT.md`, one plain-text file with the key tables, a "sanity checks" section and diagnostics, designed to be pasted into a chat.
- `slurm/run_pipeline_tuned.sbatch`: validates the params file, optionally tunes (`TUNE=1`), runs the benchmark and figures, then prints the sanity-check section. Options: `SMOKE=1` (about 20,000 cells quick check), `RESUME=1` (re-run only missing/failed methods), `METHODS=...`, `REPEATS=n` (several seeds). After a failure it still runs `visualize` so partial results are not lost.

### 3.4 Hierarchy for CRC
- `configs/hierarchies/crc_tma.json` maps the 17 labels to a level 2 (8 groups) and level 1 (3 groups). `hierarchy_mappings()` in `src/evaluation/repo_assets.py` loads the repo's pickle and then merges these JSON files (`setdefault`, so existing mappings win).
- Grouping (our own, not an official one): tumor; myeloid (macrophages, granulocytes, CD11c+ DCs), T cells (five T-cell types), B lineage (B cells, plasma cells), NK cells, unspecified immune; stromal (stroma, smooth muscle), endothelial (vasculature, lymphatics).
- **Why:** CRC has one set of 17 labels, so unknown labels were left unchanged when mapped to a coarser level, giving identical levels 1 to 3. Hierarchical F1 (weight 0.24 in "Overall") also could not be computed.
- **Status: untested.** `crc_tma_v2` was started before this code existed.

---

## 4. Method by method: configuration, result, why

All scores are macro-F1 (mean over 5 folds) unless stated. "Last week" is from October 5. Other metrics are in the leaderboard in section 5.

### 4.1 Supervised and learned methods

| Method | Key settings now | Last week | Now | What changed and why |
|---|---|---|---|---|
| **XGBoost** | up to 1000 trees, early stopping 30 rounds on the validation split, learning rate 0.1, depth 8, subsample 0.8 | 0.808 | **0.821** | Old run stopped at 100 trees. Small gain because it was already near its ceiling. |
| **MAPS** | up to 300 epochs (min 50, patience 50), batch 256, `standard_split` = true (trains on the 80% fold) | 0.194 | **0.792** | **Real bug.** The script hard-coded `max_epochs=2` and trained on the small 20% fold then predicted the other 80%. |
| **Random forest** | 500 trees, balanced class weights | 0.763 | 0.767 | More trees reduce variance; balanced weights help rare types. |
| **STELLAR** | 100 epochs, 256 hidden, graph radius 3 x median neighbour distance | 0.655 | 0.712 | Longer training and a wider graph radius; +0.057 (the largest supervised gain after MAPS). Cause not isolated further. |
| **kNN smooth** | 15 spatial neighbours, alpha 0.5, 300 trees | 0.656 | 0.670 | Neighbour smoothing of the classifier outputs. |
| **Spatial GNN** | 15 neighbours, 2 hops, C 10, 1000 iterations | 0.666 | 0.668 | No real change. |
| **Logistic regression** | C 10, max_iter 2000, balanced | 0.637 | 0.638 | 100 iterations did not converge on standardised markers; converging made no real difference. |
| **SVM** | RBF with C 10 up to 50,000 training cells, linear (LinearSVC) above | 0.618 | 0.618 | Unchanged. With 188k training cells (80% of 235k) it uses the linear fallback, which explains why it behaves like logistic regression. |
| **SingleR** | `n_top` 500 (uses every marker) | 0.428 | 0.428 | Builds one reference profile per type from training cells. Flat with more data (see the scarcity table). |
| **scANVI** | scvi defaults: 50 epochs, 1 layer, 128 hidden, 10 latent, batch 128 | 0.67 (partly failed) | **missing** | Re-run pending (see issues). A bigger network (2 layers, 256 hidden, 30 latent, batch 512) diverged to NaN at the first batch, so the defaults are used. |
| **scArches** | same network, 50 reference + 20 query epochs | 0.55 | **missing** | Re-run pending. |

**Why XGBoost wins:** it learns from labels and combines all 56 markers, with boosting handling the non-linear marker interactions. The labels themselves were made from all 56 markers plus tissue context, so a model that uses all markers can reproduce them. (Explanation is inferred but consistent with the label analysis in section 6.)

**Why STELLAR/MAPS are slightly below XGBoost:** they are neural/graph models trained on the same markers; on tabular data of this kind, gradient-boosted trees often match or beat them. (Inferred.)

### 4.2 Unsupervised clustering (clusters named by majority true label)

| Method | Key settings now | Last week | Now |
|---|---|---|---|
| **FlowSOM** | 15x15 self-organising map, 30 training rounds, 40 meta-clusters | 0.317 | 0.403 (13 of 17 labels used) |
| **Leiden** | resolutions 0.5, 1, 2, 3, 5; 15 neighbours; best shown | 0.305 (res 2) | 0.333 (res 5; 11 labels used) |
| **SPADE** | 50 clusters (more than types), ward linkage, 10,000 cells sampled | 0.200 | 0.291 |
| **Louvain** | resolution 1.5, 15 neighbours | 0.260 | 0.280 |
| **Starling** | up to 200 epochs, patience 20, 30 initial clusters | 0.089 | 0.111 (only 7 of 17 labels used) |

**Why they score about 0.3 to 0.4:** they can only output a type that wins at least one cluster. FlowSOM names 13 types, Leiden 11, so the rest score exactly 0, and macro-F1 counts every type equally. Over-clustering (more clusters than types) helps because naming is greedy, which is why the settings use more clusters than the 17 types. This is a structural limit, not a bug.

**Starling:** its early stopping (Lightning default patience 3) ended training almost immediately. Patience 20 and 200 epochs raised it only from 0.089 to 0.111, and it still outputs only 7 of 17 labels. **I do not know why.** I have not examined its clusters or per-cluster assignment.

### 4.3 Prior-knowledge methods (marker matrices)

| Method | Key settings now | Last week | Now | Notes |
|---|---|---|---|---|
| **TACIT** | r 200, p 20 | 0.216 | 0.262 (8% undefined) | Authors' own settings for this dataset |
| **Astir** | 100 epochs, 10 initialisations, `assignment_type = max` | 0.166 | 0.198 | Default 0.7 confidence cut made uncertain cells "Unknown" (counted wrong) |
| **Marker score** | `min_score` 0 | 0.277 | 0.277 | Simple z-score average per type |
| **Scyan** | preprocess on, `zeros_as_nan`, up to 200 epochs, min_delta 0.1 | 0.054 | 0.054 | Unchanged; 58% of cells "undefined" |
| **Tribus** | outlier quantile 1.0, clustering threshold 15000 | skipped | not in report | |

**TACIT, how it works (from the authors' code, which I read):** CLR normalisation, PCA/UMAP with `p` dimensions, Leiden with resolution `r` to make micro-clusters, then per-type thresholds from piecewise regression on cluster medians. Clusters are "clean" or "mixed"; mixed ones are resolved with kNN (k = 10); anything left is "Others", which we record as "undefined". Leiden `n.iter = 2` is hard-coded in the authors' function. Their CRC example uses r = 200 and p = 20, which is what we use. The package default r = 10 gives too few micro-clusters.

**Why TACIT is low here:** its quality depends on the signature matrix, and we do not have the authors' CRC matrix. With our drafted matrix, 104,000 of 235,519 cells (44%) went to "immune cells" (recall 0.87, precision 0.03) and CD3+ and CD4+ T cells, whose rows were identical, both scored 0. The authors' own matrix would be the fair test. The corresponding author is huynhk4@vcu.edu (from my earlier notes; check the TACIT README).

**Scyan, what I know and what I do not:**
- Scyan is a normalising flow trained with priors from the table. In its table, +1 means high, -1 low, NaN means the marker is not used for that population, and 0 means exactly average.
- Our matrix is about 90% blank. We passed the blanks as 0, which pinned each type to "average" on about 50 markers. I changed blanks to NaN (`zeros_as_nan`).
- **That did not change the score (0.054).** 58% of cells now come out as NaN, which we record as "undefined".
- **Hypothesis (unverified):** the predict step has a confidence cut-off that leaves cells unassigned. I asked for the output of `inspect.signature(scyan.Scyan.predict)` and its docstring to check; **I have not received it.** Until then the cause is open.

**Astir:** only 29% of its calls are right even though it assigns every cell. Astir only accepts positive markers per type, which loses the negative information of the matrix.

### 4.4 Baselines
- Most frequent 0.020 and stratified 0.058: identical to last week, as they should be. They define the floor.

---

## 5. Results

### 5.1 Leaderboard (mean over 5 folds)

| Method | Macro-F1 | Weighted-F1 | Accuracy | MCC | Overall |
|---|---|---|---|---|---|
| XGBoost | 0.821 | 0.879 | 0.878 | 0.862 | 0.658 |
| MAPS | 0.792 | 0.863 | 0.862 | 0.844 | 0.640 |
| Random forest | 0.767 | 0.845 | 0.848 | 0.827 | 0.627 |
| STELLAR | 0.712 | 0.826 | 0.821 | 0.800 | 0.601 |
| kNN smooth | 0.670 | 0.770 | 0.772 | 0.741 | 0.569 |
| Spatial GNN | 0.668 | 0.794 | 0.788 | 0.763 | 0.573 |
| Logistic regression | 0.638 | 0.783 | 0.775 | 0.749 | 0.560 |
| SVM | 0.618 | 0.762 | 0.760 | 0.732 | 0.546 |
| SingleR | 0.428 | 0.595 | 0.573 | 0.530 | 0.405 |
| FlowSOM | 0.403 | 0.565 | 0.583 | 0.524 | 0.407 |
| TACIT | 0.262 | 0.447 | 0.449 | 0.395 | 0.292 |

Other macro-F1: Leiden res 5 0.333, res 2 0.304, FlowSOM meta-clusters 0.302, res 3 0.300, SPADE 0.291, Louvain 0.280, marker score 0.277, Leiden res 1 0.270, res 0.5 0.247, Astir 0.198, Starling 0.111, stratified 0.058, Scyan 0.054, most frequent 0.020.

**Reading it:**
- Weighted-F1 and accuracy are higher than macro-F1 for every method because big types are easy and count more in them. The gap is largest for weak methods (TACIT 0.262 macro vs 0.447 weighted): they get common types roughly right and fail on rare ones.
- **"Overall" is the benchmark's weighted mix, not weighted-F1:** weighted-F1 0.16, hierarchical F1 0.24, macro-F1 0.20, MCC 0.20, ARI 0.08, composition similarity (scaled JSD) 0.12. Hierarchical F1 is not available in this run, so its 0.24 contributes nothing and every Overall is lower than its F1. Compare Overall between methods in this run only, not with published numbers.
- **Noise:** standard deviation of macro-F1 across folds is about 0.01 for the top methods (XGBoost 0.010, MAPS 0.008, random forest 0.009). Differences below about 0.02 between neighbours are within noise.
- The three levels (1, 2, 3) in the heatmap are identical because the run predates the hierarchy.

### 5.2 This week vs last week
Only MAPS changed a lot (+0.598). Other supervised methods moved 0.00 to 0.06 (STELLAR +0.057, XGBoost +0.013). Label-free methods moved +0.02 to +0.09 (FlowSOM +0.086, SPADE +0.091, TACIT +0.046, Astir +0.032, Leiden +0.028, Starling +0.022, Louvain +0.020). Scyan, baselines, marker score, SingleR and SVM were unchanged.

### 5.3 Which cell types are hard (F1 per type)

| Cell type | Share of cells | XGBoost | MAPS | Random forest |
|---|---|---|---|---|
| Tumor cells | 20.2% | 0.913 | 0.898 | 0.889 |
| Stroma | 8.6% | 0.735 | 0.708 | 0.666 |
| CD4+ T cells CD45RO+ | 7.1% | 0.821 | 0.798 | 0.760 |
| Immune cells | 1.3% | 0.858 | 0.831 | 0.833 |
| Tregs | 1.2% | 0.819 | 0.789 | 0.781 |
| CD4+ T cells | 1.0% | 0.737 | 0.725 | 0.676 |
| CD11c+ DCs | 0.17% | 0.685 | 0.613 | 0.534 |
| Lymphatics | 0.14% | 0.736 | 0.613 | 0.562 |
| NK cells | 0.14% | 0.838 | 0.822 | 0.796 |
| CD3+ T cells | 0.08% | 0.593 | 0.553 | 0.528 |

- Across all methods the rank correlation between a type's size and its mean F1 is 0.81: rarer types score lower.
- Hardest across 22 methods: CD3+ T cells (mean F1 0.14), CD11c+ DCs (0.17), lymphatics (0.18). Easiest: tumor cells (0.72), vasculature (0.63), B cells (0.63).
- NK cells are a surprise (0.838 for XGBoost with only 0.14% of cells), probably because their marker profile is distinctive. A batch or region effect is possible; I did not test it.

### 5.4 What XGBoost gets wrong
- CD3+ T cells: 53% correct (called CD4+ T cells 22%, B cells 12%).
- CD4+ T cells: 74% correct (called CD4+ CD45RO+ 11%, B cells 7%).
- CD11c+ DCs: 72% (called granulocytes 11%, macrophages 6%). Lymphatics: 73% (called stroma 11%, smooth muscle 5%). Stroma: 74% (smooth muscle 5%, tumor 5%). Tregs: 85% (CD4+ CD45RO+ 9%).
- The errors are between biologically close types (T-cell subtypes differing by one or two markers; stroma, smooth muscle and lymphatics sharing markers such as Vimentin and Podoplanin).
- Across all methods the most frequent error is stroma called macrophages (209,491 cells summed over 20 methods). This comes mostly from the clustering methods, where large macrophage-named clusters absorb cells of many types.

### 5.5 What the label-free methods actually predict

| Method | Macro-F1 | Labels used | Unassigned | Cells correct |
|---|---|---|---|---|
| SingleR | 0.428 | 17 of 17 | 0% | 57% |
| FlowSOM | 0.403 | 13 of 17 | 0% | 58% |
| Leiden (res 5) | 0.333 | 11 of 17 | 0% | 55% |
| SPADE | 0.291 | 11 of 17 | 0% | 52% |
| Louvain | 0.280 | 10 of 17 | 0% | 49% |
| Marker score | 0.277 | 18 | 0.1% | 39% |
| TACIT | 0.262 | 18 | 8% | 45% |
| Astir | 0.198 | 18 | 0% | 29% |
| Starling | 0.111 | 7 of 17 | 0% | 31% |
| Scyan | 0.054 | 18 | 58% | 5% |

"Labels used" counts distinct labels output; 18 means one extra label (Unknown/undefined) appeared.

### 5.6 How much training data is needed (macro-F1)

| Method | 1% | 5% | 10% | 25% | 50% | 80% |
|---|---|---|---|---|---|---|
| XGBoost | 0.663 | 0.748 | 0.765 | 0.799 | 0.825 | 0.830 |
| Random forest | 0.577 | 0.681 | 0.704 | 0.737 | 0.762 | 0.770 |
| kNN smooth | 0.511 | 0.601 | 0.624 | 0.657 | 0.677 | 0.681 |
| Spatial GNN | 0.552 | 0.649 | 0.661 | 0.672 | 0.668 | 0.666 |
| Logistic regression | 0.590 | 0.637 | 0.659 | 0.643 | 0.636 | 0.636 |
| SVM | 0.563 | 0.678 | 0.701 | 0.622 | 0.619 | 0.618 |
| SingleR | 0.419 | 0.429 | 0.432 | 0.433 | 0.432 | 0.430 |

- Methods improve quickly and then flatten. The median method reaches 95% of its best score with about 5% of the data. A few thousand labelled cells already give a usable classifier.
- Tree methods keep gaining to 50 to 80%; linear/spatial flatten at 10 to 25%. SingleR is flat because it averages the training cells into one profile per type.
- SVM drops from 0.70 (10%) to 0.62 (25%). This is consistent with it switching from an RBF to a linear model above 50,000 cells, but I have not checked the exact switch point.
- Only the cheaper methods have progressive runs; MAPS, STELLAR and the label-free methods are not in this table.

### 5.7 Runtime and scalability (rough)
Best methods are not the expensive ones: XGBoost and random forest run in about 1,000 seconds. Logistic regression, SVM, kNN smooth, spatial GNN, SingleR, marker score and the two baselines are on the performance-vs-runtime frontier. Scyan, Starling and Astir are slow and low. Runtimes are for one run on this cluster and depend on the node type. I have not checked the exact formula of the scalability score.

### 5.8 Metric correlation
F1, accuracy, MCC, kappa, ARI and NMI rank the methods almost identically (a dark block in the correlation figure), so macro-F1 is a fair single headline number. Composition metrics (R squared, Pearson), g-mean, completeness and unassigned fraction carry different information. This correlation is across about 25 entries; a few very poor methods (baselines, Scyan) drive a lot of it.

---

## 6. Why the label-free methods score low

1. **The labels are clusters.** The 17 types are the original authors' clusters, built from all 56 markers combined with tissue location, followed by manual curation (marker expression, morphology, location). A method that matches a few markers per type cannot reproduce that.
2. **A measured ceiling.** For each type I picked the 3 markers that best agree with the true labels (greedy, one at a time, each chosen to raise macro-F1, on up to 1,000 cells per type) and scored every cell with a simple rule (z-score every marker; a cell's score for a type = mean z-score of that type's markers; take the best type). Results over all 235,519 cells:

   | Matrix | Accuracy | Macro-F1 |
   |---|---|---|
   | Hand-made | 0.39 | 0.29 |
   | Hand-made without "immune cells" | 0.43 | 0.32 |
   | Drafted from the labels (top 2 to 3 markers by average z-score) | 0.53 | 0.38 |
   | Best 3 markers per type | 0.60 | 0.45 |

   The last two use the true labels, so they are best cases. Even they reach only 0.45 macro-F1. This is a stand-in for TACIT, not TACIT itself; real TACIT with the drafted matrix scored lower (macro-F1 0.26, accuracy 0.36, weighted-F1 0.44). "Immune cells" is defined by CD45 alone, which overlaps every immune type, so removing it helps.
3. **Weak, noisy markers.** A type's strongest marker is only 0.8 to 1.9 standard deviations above average. Stroma's strongest is 0.7. Cytokeratin is non-zero in only 64% of tumor cells, and the data has many exact zeros (the example granulocyte has CD45 = 0).
4. **Rare types and the metric.** Five types have under 1% of cells each (189 to 2,303 cells: CD3+ T, NK, CD11c+ DCs, lymphatics, CD4+ T). Almost every method scores about 0 on them, and macro-F1 counts every type equally.

**Conclusion:** the low label-free scores are mostly a property of the labels and the metric, not of the code. The exception is Scyan (and probably Starling), where the behaviour is still unexplained.

---

## 7. Problems found and how they were resolved

| # | Problem | Cause | Fix | Status |
|---|---|---|---|---|
| 1 | MAPS scored 0.19 | `max_epochs` hard-coded to 2; trained on the 20% fold and predicted the other 80% | epochs, patience, batch size and `standard_split` are now settings; up to 300 epochs on the 80% training fold | **Confirmed**: 0.792 |
| 2 | scANVI crashed on small training sets | last batch had a single cell; batch normalisation needs at least 2 ("Expected more than 1 value per channel") | incomplete last batch is dropped (`drop_last`) | Worked in later runs |
| 3 | scANVI / scArches sometimes NaN at the first batch | unknown; random; the input has no NaN, infinite or negative values | up to 3 attempts, each with a new seed and half the learning rate (`_retry`, `_lr`, `_check_trained` in `src/models/reference.py`) | **Untested** |
| 4 | Scyan near zero | blank matrix cells were passed as 0 ("average") | blanks passed as NaN (`zeros_as_nan`, in `_scyan` in `scripts.py`) | Applied, **score unchanged**; not the cause |
| 5 | Scyan: "No CUDA GPUs are available" on CPU runs | Lightning's `isolate_rng` calls the GPU RNG even for a CPU run | `run_scyan.py` sets `CUDA_VISIBLE_DEVICES=""` when `--accelerator cpu` | Worked |
| 6 | Drafted matrix had empty rows (stroma etc.) | kept only markers at least 1 SD above average | every type keeps at least its top 2 markers (`draft_matrix` in `marker.py`) | Worked |
| 7 | Levels 1 to 3 identical | CRC has no hierarchy | `configs/hierarchies/crc_tma.json` | **Untested** |
| 8 | Starling stopped almost at once | Lightning default patience 3 | patience 20, up to 200 epochs | Slight gain only (0.089 to 0.111) |
| 9 | Astir left uncertain cells "Unknown" | default 0.7 confidence threshold | `assignment_type = max` | +0.032 |
| 10 | TACIT too few micro-clusters | package default `r = 10` | r 200, p 20 (authors' own) | +0.046 |
| 11 | TACIT could not read negative matrix values | TACIT format is 1 or blank | negatives clipped to 0 in `_tacit` | Done |
| 12 | `figure_text` crashed on error-bar plots | `ErrorbarContainer` has no `patches` | skip containers without `patches` | Fixed (must be pulled on the cluster) |
| 13 | A failed pipeline lost the visualisation | `visualize` ran only at the very end | the Slurm script now still runs `visualize` after a non-zero exit | Done |
| 14 | `srun` "Memory required by task is not available" | the interactive job (15 CPUs, 15 GB) could not give a sub-step more memory | run python directly in the allocation, or use `sbatch --wrap` | Resolved |

**Hypotheses I checked and ruled out:** NaN or infinite values in the data; negative values / need for log1p; a preprocessing mistake; a bug in the fold split or scoring; a hierarchy bug (the levels were identical only because the run predates the hierarchy code).

---

## 8. Caveats of the evaluation (important when presenting)

1. **Folds are split by cell, not by image or patient.** A model can see a neighbour of a test cell in training, so supervised scores are optimistic. A group split exists in the repo but is not used.
2. **Clusters are named with the true labels** (the benchmark's own rule), so clustering scores are upper bounds and more clusters give a higher score.
3. **Macro-F1 has an extra class:** a prediction such as "undefined" counts as an 18th class with F1 0, which lowers macro-F1 by up to 1/18 (about 5.6%).
4. **Matrix quality:** the marker methods depend on our hand-made matrix. A matrix drafted from the labels is circular and is used only for comparison.
5. **One label granularity:** until the hierarchy is run, levels 1 to 3 are the same, and hierarchical F1 is missing from Overall.
6. **Several fixes are in the code but not yet confirmed on the cluster, and tuning has not been run.**
7. SVM falls back to LinearSVC above 50,000 training cells, so its curve is not a pure RBF model at large sizes.

---

## 9. What is still open

1. **Scyan: 58% undefined.** Get the output of this on the cluster and check whether `predict` has a confidence/threshold argument:
   ```bash
   python -c "import inspect, scyan; print(inspect.signature(scyan.Scyan.predict)); print(scyan.Scyan.predict.__doc__)"
   ```
2. **Starling: only 7 of 17 labels.** Needs a look at its cluster assignments.
3. **scANVI and scArches** are missing from the report. Re-run only those (after pulling the latest code):
   ```bash
   METHODS=scanvi,scarches OUT=data/results/crc_tma_v2_scvi sbatch slurm/run_pipeline_tuned.sbatch
   ```
4. **Hierarchy levels 2 and 1:** run a new (or resumed) run once the hierarchy code is on the cluster, and check that levels now differ and hierarchical F1 appears.
5. **Tuning:** `TUNE=1 TUNE_METHODS=... TRIALS=... sbatch slurm/run_pipeline_tuned.sbatch`. I expect small gains for label-free methods because of the ceiling.
6. **TACIT:** ask the authors for their CRC signature matrix (corresponding author: huynhk4@vcu.edu, check the README). It is the fairest test of the method.
7. **A stricter split** (by image or patient) for the supervised methods.
8. After re-runs, update the deck: the "this week" results and the "Not yet confirmed" statuses.

---

## 10. Glossary

- **Macro-F1:** F1 per cell type, then the plain average, so each type counts equally (rare types matter).
- **Weighted-F1:** F1 per type weighted by how many cells it has (big types dominate).
- **Accuracy:** fraction of cells labelled correctly.
- **MCC (Matthews correlation coefficient):** chance-corrected agreement, from -1 to 1, robust to class imbalance.
- **ARI / NMI:** agreement between two groupings, corrected for chance (ARI) or based on shared information (NMI).
- **JSD:** Jensen-Shannon distance between predicted and true cell-type proportions (lower is better; "scaled" flips it so higher is better).
- **Overall:** the benchmark's weighted combination of metrics (section 5.1).
- **Pareto-optimal:** no other method is both faster and better.
- **Greedy mapping:** to score clusters, each cluster is given the true label that most of its cells carry.
- **Patience / early stopping:** training stops when the validation score has not improved for this many epochs (or rounds).
- **Micro-cluster:** a small cluster made before deciding each cell's type (TACIT).
- **CLR / arcsinh:** normalisations; arcsinh (cells divided by a cofactor, here 5) is already applied to this table.

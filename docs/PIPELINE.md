# spCellEval Pipeline Guide

End-to-end benchmarking for high-plex spatial proteomics (IMC, MIBI, CycIF, CODEX).

## Quick start

```bash
# Full pipeline (preprocess → benchmark → spatial → evaluation)
python src/run_pipeline.py --dataset IMMUcan --recreate_kfolds

# Tabular methods only, skip preprocessing when quant CSV exists
python src/run_pipeline.py --dataset CRC_TMA --skip_preprocess --methods signature scyan
```

## Stages

| Stage | Module | Script | Output |
|-------|--------|--------|--------|
| 1. ETL | Preprocessing | `src/preprocessing/run_preprocess.py` | `data/processed/{dataset}/*_quantification.csv` |
| 2. Feature separation | Pseudo-labeling | `src/pseudo_labeling/run_pseudo_labeler.py` | `*_features_only.csv` (labels stripped) |
| 3. Benchmark | Methods | `src/benchmark/run_benchmark.py` | `results/{dataset}/{method}/level3/predictions_*.csv` |
| 4. Spatial | Post-process | `src/spatial/postprocess.py` | Smoothed predictions + graph summaries |
| 5. Evaluation | Metrics | `src/evaluation/run_evaluation.py` | `results/{dataset}/summary/final_results.csv` |

`run_pipeline.py` disables benchmark-internal spatial smoothing (`--no_spatial_smooth`) so smoothing runs once in Stage 4.

## Unlabeled clinical samples (no expert annotations)

When raw data has **no** `ClusterName` / ground-truth column:

```bash
python src/run_pipeline.py --dataset CRC_TMA --unlabeled
```

This will:

1. Run ETL with `--unlabeled` (markers + spatial only, no `cell_type`)
2. Pseudo-label via **signature** (writes `*_annotated.csv` + benchmark predictions)
3. Benchmark **signature, scyan, leiden** only (no k-folds, no RF/XGBoost)
4. Evaluate with **unsupervised metrics** (silhouette, marker purity, spatial entropy)

Use `--pseudo_method tacit` for R-based TACIT gating (requires R installed).

## Adding a dataset

1. Create `configs/<dataset>.yaml` (column mappings, markers, label remap).
2. Add an entry to `configs/benchmark.yaml` (methods, artifacts).
3. Place decision matrices under `configs/artifacts/<dataset>/`.
4. Add marker purity rules to `configs/marker_purity_rules.yaml`.
5. Place raw data under `data/raw/<dataset>/` (gitignored).

## Method requirements

| Category | Methods | Requirements |
|----------|---------|--------------|
| Tabular ML | random_forest, xgboost, logistic_regression | k-fold CSVs |
| Unsupervised | leiden, flowsom, phenograph | R for FlowSOM/PhenoGraph |
| Prior knowledge | signature, scyan, tacit, astir | Decision matrix in `configs/artifacts/`; R for TACIT |
| Image / deep | cellsighter, stellar, virtues_*, eva_*, kronos_* | **IMMUcan only**; raw images under `data/raw/IMMUcan/`; GPU recommended |

CRC_TMA and other datasets should omit image-based methods from `benchmark.yaml` unless wired explicitly.

## Configuration files

- `configs/benchmark.yaml` — default methods per dataset, artifact paths
- `configs/evaluation.yaml` — evaluation defaults
- `configs/marker_purity_rules.yaml` — biological purity rules per dataset
- `configs/artifacts/` — versioned decision matrices (canonical location)

## Tests

```bash
pytest tests/ -q
```

Integration coverage: `tests/test_pipeline_e2e.py` (signature → evaluation → spatial).

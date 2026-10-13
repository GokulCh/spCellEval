#!/bin/bash
# Full benchmark with the tuned method parameters (configs/method_params.json), 1 GPU. Based on run_pipeline_gpu.sbatch.
#
# What it does:   1. validates the params file   2. (optional) tunes parameters on validation splits   3. benchmark + figures
#                 + the plain-text report (summary/AI_REPORT.md)   4. prints the sanity-check section of that report.
# Everything is written to $OUT, including a copy of the exact parameters used ($OUT/method_params.json); the repo's
# configs/method_params.json is never modified.
#
# Submit with:    mkdir -p logs && sbatch slurm/run_pipeline_tuned.sbatch
#
# Quick check first (about 20k cells, hold-out only, the slow methods included; usually well under an hour):
#                 SMOKE=1 sbatch slurm/run_pipeline_tuned.sbatch
# Tune, then benchmark with the tuned values:
#                 TUNE=1 sbatch slurm/run_pipeline_tuned.sbatch
#                 TUNE=1 TUNE_METHODS=xgboost,louvain,tacit TRIALS=20 sbatch slurm/run_pipeline_tuned.sbatch
# Several seeds (mean / std of macro-F1 in summary/seed_variability.csv):   REPEATS=3 sbatch slurm/run_pipeline_tuned.sbatch
# Re-run only failed / missing methods of an earlier run:                    RESUME=1 OUT=<that run's folder> sbatch ...
#
# Other settings (environment variables, all optional):
#   DATA            quantification csv                        OUT          output folder
#   PARAMS          starting params file (default configs/method_params.json)
#   MARKER_MATRIX   decision-matrix csv for TACIT / Scyan / Astir / Tribus (set but empty = none -> drafted from the labels, circular)
#   MODALITY, TRANSFORM   data options (default crc_tma / none = the table is already transformed)
#   TIMEOUT         seconds per method run (default 43200 = 12 h: the longer training of Scyan / Astir / STARLING / MAPS needs it)
#   JOBS, THREADS, ORDER, SCRIPT_RUNS   as in run_pipeline_gpu.sbatch
#   TUNE=1, TRIALS (12), TUNE_METHODS, STAGE1_CELLS (20000), TUNE_MAX_CELLS (0 = all)   the tuning step
#   METHODS         methods of the full run (default all), e.g. METHODS=scyan,astir,starling,maps,tacit,flowsom,leiden to re-run only those
#   SMOKE=1, SMOKE_METHODS, SMOKE_CELLS (20000)             the quick check
# Other GPU / time limit: sbatch --partition=gpu-h100 --time=<limit> slurm/run_pipeline_tuned.sbatch
#SBATCH --job-name=spceleval_tuned
#SBATCH --output=logs/spceleval_tuned_%j.out
#SBATCH --error=logs/spceleval_tuned_%j.out
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=96:00:00
#SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=<your email>
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1

set -eo pipefail

# The environment's own python: no `conda activate` needed (batch jobs do not load your shell setup).
PYTHON=${PYTHON:-/lustre/home/chaluvadig/anaconda3/envs/cta/bin/python}
if [ ! -x "$PYTHON" ]; then
  echo "python not found: $PYTHON"
  echo "Run 'which python' inside your activated environment and pass it:  PYTHON=<that path> sbatch slurm/run_pipeline_tuned.sbatch"
  exit 1
fi
export PATH="$(dirname "$PYTHON"):$PATH"          # so the environment's Rscript (FlowSOM, TACIT) and other tools are found too

cd /lustre/home/chaluvadig/VCU_HEALTH/spCellEval

DATA=${DATA:-/lustre/home/chaluvadig/VCU_HEALTH/spCellEval/data/datasets/crc_tma/quantification/processed/crc_tma_quantification.csv}
SMOKE=${SMOKE:-}
TUNE=${TUNE:-}
if [ -n "$SMOKE" ]; then OUT=${OUT:-data/results/crc_tma_smoke}; else OUT=${OUT:-data/results/crc_tma_tuned}; fi
MODALITY=${MODALITY:-crc_tma}
TRANSFORM=${TRANSFORM:-none}
CPUS=${SLURM_CPUS_PER_TASK:-32}
JOBS=${JOBS:-$(( CPUS / 2 ))}                     # methods running at the same time
[ "$JOBS" -lt 1 ] && JOBS=1
THREADS=${THREADS:-$(( CPUS / 8 ))}               # threads inside the methods that can use them (xgboost, random forest, ...)
[ "$THREADS" -lt 1 ] && THREADS=1
if [ -n "$SMOKE" ]; then TIMEOUT=${TIMEOUT:-3600}; else TIMEOUT=${TIMEOUT:-43200}; fi   # a smoke test must not hang for hours
ORDER=${ORDER:-slow-first}                        # starts the slowest methods at once; usually the shortest total time
SCRIPT_RUNS=${SCRIPT_RUNS:-3}                     # repeats of the repo's script methods (stability needs > 1)
REPEATS=${REPEATS:-1}
TRIALS=${TRIALS:-12}
STAGE1_CELLS=${STAGE1_CELLS:-20000}
TUNE_MAX_CELLS=${TUNE_MAX_CELLS:-0}
TUNE_METHODS=${TUNE_METHODS:-xgboost,random_forest,logistic_regression,svm,louvain,leiden,flowsom,spade,tacit,spatial_gnn}
SMOKE_METHODS=${SMOKE_METHODS:-xgboost,random_forest,logistic_regression,louvain,leiden,flowsom,tacit,scyan,astir,tribus,starling,maps,stellar}
SMOKE_CELLS=${SMOKE_CELLS:-20000}
METHODS=${METHODS:-all}
PARAMS=${PARAMS:-configs/method_params.json}
MARKER_MATRIX=${MARKER_MATRIX-data/datasets/crc_tma/quantification/processed/crc_tma_marker_matrix.csv}   # unset -> default file; set but empty -> none

[ -f "$DATA" ] || { echo "data not found: $DATA"; exit 1; }
[ -f "$PARAMS" ] || { echo "params file not found: $PARAMS"; exit 1; }
mkdir -p "$OUT"
WORK_PARAMS="$OUT/method_params.json"             # this run's own copy; tuning writes here, the repo's file stays untouched
if [ -n "${RESUME:-}" ] && [ -f "$WORK_PARAMS" ]; then
  echo "resuming: keeping the parameters already in $WORK_PARAMS"
else
  cp "$PARAMS" "$WORK_PARAMS"
fi

EXTRA=()
[ -n "${RESUME:-}" ] && EXTRA+=(--resume)
if [ -n "$MARKER_MATRIX" ]; then
  [ -f "$MARKER_MATRIX" ] || { echo "marker matrix not found: $MARKER_MATRIX"; exit 1; }
  EXTRA+=(--marker-matrix "$MARKER_MATRIX")
fi
COMMON=(--data "$DATA" --modality "$MODALITY" --transform "$TRANSFORM" --params "$WORK_PARAMS" "${EXTRA[@]}")

echo "python=$PYTHON"
echo "data=$DATA"
echo "out=$OUT  params=$WORK_PARAMS (copy of $PARAMS)"
echo "cpus=$CPUS jobs=$JOBS threads=$THREADS order=$ORDER timeout=${TIMEOUT}s script-runs=$SCRIPT_RUNS repeats=$REPEATS"
echo "marker-matrix=${MARKER_MATRIX:-none (drafted from the labels: TACIT / Scyan / Astir / Tribus results are circular)}"
echo "mode: ${SMOKE:+SMOKE (hold-out, $SMOKE_CELLS cells) }${TUNE:+TUNE ($TRIALS trials) }$([ -z "$SMOKE$TUNE" ] && echo full benchmark)"
nvidia-smi -L 2>/dev/null || echo "no GPU visible (the GPU methods fall back to CPU)"

# 1. the params file must be valid before hours are spent
"$PYTHON" -u cli.py params --check --params "$WORK_PARAMS"

# 2. optional tuning on validation splits (never on the test split); writes the best values into $WORK_PARAMS
if [ -n "$TUNE" ] && [ -z "$SMOKE" ]; then
  "$PYTHON" -u cli.py tune "${COMMON[@]}" \
    --methods "$TUNE_METHODS" --trials "$TRIALS" --stage1-cells "$STAGE1_CELLS" --max-cells "$TUNE_MAX_CELLS" \
    --timeout "$TIMEOUT" --jobs "$JOBS" --model-threads "$THREADS" --out "$OUT/tune"
fi

# 3. benchmark + figures + text report
if [ -n "$SMOKE" ]; then
  RUN=(--stages benchmark,visualize --split holdout --methods "$SMOKE_METHODS" --max-cells "$SMOKE_CELLS" --script-runs 1)
else
  RUN=(--stages benchmark,visualize --split all --methods "$METHODS" --script-runs "$SCRIPT_RUNS" --repeats "$REPEATS")
fi
"$PYTHON" -u cli.py pipeline "${COMMON[@]}" "${RUN[@]}" \
  --timeout "$TIMEOUT" --jobs "$JOBS" --model-threads "$THREADS" --order "$ORDER" \
  --out "$OUT" || rc=$?

# every finished method is saved on its own, so figures can be rebuilt from a partial or failed run
if [ "${rc:-0}" -ne 0 ]; then
  echo "pipeline ended with an error (exit $rc): building the figures and reports from the results saved so far"
  "$PYTHON" -u cli.py visualize --results "$OUT" || true
fi

# 4. where to look
echo
echo "================ done ================"
echo "plain-text report (paste into an AI model): $OUT/summary/AI_REPORT.md   (short: AI_REPORT_SHORT.md)"
echo "figures: $OUT/figures and $OUT/<dataset>/figures (each figure has a .txt twin)"
echo "parameters used: $WORK_PARAMS and $OUT/<dataset>/params_used.json"
[ -n "$TUNE" ] && [ -z "$SMOKE" ] && echo "tuning: $OUT/tune/tuning_summary.csv (current vs tuned score)"
echo
echo "---- sanity checks on the predictions ----"
sed -n '/^## Sanity checks/,/^## /p' "$OUT/summary/AI_REPORT_SHORT.md" 2>/dev/null | head -40 || true

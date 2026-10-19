#!/bin/bash
# Parameter tuning for the light and medium methods only (no Scyan, Astir, TACIT, Starling, MAPS, Stellar, Tribus: each trial of those is a
# full multi-minute to multi-hour run, and the prior-knowledge ones are limited by the signature, not by their settings).
#
# Submit:   mkdir -p logs && sbatch slurm/tune_light.sbatch
# Resume:   RESUME=1 sbatch slurm/tune_light.sbatch          (keeps the parameters already tuned in $OUT/method_params.json)
# Options (environment variables):
#   TUNE_METHODS   methods to tune          TRIALS (12)   random candidates per method (plus the current setting)
#   REFINE_EVALS (8)   one-at-a-time refinement evaluations on all the data       BUDGETS (5000,20000)   successive-halving subsample sizes
#   JOBS (8)  trials at once   THREADS (4)  CPU threads inside each trial   (keep JOBS x THREADS <= CPUs)   TRIAL_TIMEOUT (7200 s)
# Output: $OUT/method_params.json (the tuned parameters), $OUT/tune/tuning_report.md (what changed and which parameters mattered),
#         tuning_summary.csv, tuning_log.csv, tuning_gain.png, tune.log.
# The repo's configs/method_params.json is never modified.
#SBATCH --job-name=spc_tune
#SBATCH --output=logs/spc_tune_%j.out
#SBATCH --error=logs/spc_tune_%j.out
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1

set -eo pipefail

PYTHON=${PYTHON:-/lustre/home/chaluvadig/anaconda3/envs/cta/bin/python}
[ -x "$PYTHON" ] || { echo "python not found: $PYTHON (pass PYTHON=<path>)"; exit 1; }
export PATH="$(dirname "$PYTHON"):$PATH"                 # the environment's Rscript etc.

cd /lustre/home/chaluvadig/VCU_HEALTH/spCellEval

DATA=${DATA:-/lustre/home/chaluvadig/VCU_HEALTH/spCellEval/data/datasets/crc_tma/quantification/processed/crc_tma_quantification.csv}
OUT=${OUT:-data/results/crc_tma_tuned}
MARKER_MATRIX=${MARKER_MATRIX:-src/methods/TACIT/crc_tma_decision_matrix_level3.csv}
TUNE_METHODS=${TUNE_METHODS:-xgboost,random_forest,logistic_regression,svm,singler,knn_smooth,spatial_gnn,marker_score,spade,louvain,leiden,flowsom}
TRIALS=${TRIALS:-12}
REFINE_EVALS=${REFINE_EVALS:-8}
BUDGETS=${BUDGETS:-5000,20000}
JOBS=${JOBS:-8}
THREADS=${THREADS:-4}
TRIAL_TIMEOUT=${TRIAL_TIMEOUT:-7200}

[ -f "$DATA" ] || { echo "data not found: $DATA"; exit 1; }
[ -f "$MARKER_MATRIX" ] || { echo "marker matrix not found: $MARKER_MATRIX"; exit 1; }
mkdir -p "$OUT/tune" logs
if [ -n "${RESUME:-}" ] && [ -f "$OUT/method_params.json" ]; then
  echo "resuming: keeping the parameters already in $OUT/method_params.json"
else
  cp configs/method_params.json "$OUT/method_params.json"
fi

echo "python=$PYTHON"; echo "data=$DATA"; echo "out=$OUT"
echo "methods=$TUNE_METHODS"
echo "trials=$TRIALS budgets=$BUDGETS refine-evals=$REFINE_EVALS jobs=$JOBS threads=$THREADS trial-timeout=${TRIAL_TIMEOUT}s"
nvidia-smi -L 2>/dev/null || echo "no GPU visible"

"$PYTHON" -u cli.py params --check --params "$OUT/method_params.json"

"$PYTHON" -u cli.py tune --data "$DATA" --modality crc_tma --transform none \
  --marker-matrix "$MARKER_MATRIX" --params "$OUT/method_params.json" \
  --methods "$TUNE_METHODS" --trials "$TRIALS" --budgets "$BUDGETS" --eta 3 --refine 1 --refine-evals "$REFINE_EVALS" --min-gain 0.002 \
  --jobs "$JOBS" --model-threads "$THREADS" --timeout "$TRIAL_TIMEOUT" --device auto --out "$OUT/tune"

echo
echo "================ tuning done ================"
echo "tuned parameters: $OUT/method_params.json"
echo "what changed and why: $OUT/tune/tuning_report.md"
echo "next: sbatch slurm/run_final.sbatch   (uses $OUT/method_params.json)"

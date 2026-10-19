#!/bin/bash
# WIDE variant of run_final.sbatch: all methods start at once on a bigger allocation (45 CPUs, 1 GPU). Edit the #SBATCH lines to what your
# node allows; keep JOBS x THREADS <= CPUs. The run then takes about as long as the slowest method (Starling) instead of the sum of many.
# The full benchmark on crc_tma: every method (heavy ones included) with the tuned parameters from slurm/tune_light.sbatch.
# Methods that were not tuned (Scyan, Astir, TACIT, Starling, MAPS, Stellar, Tribus) keep the shipped settings in the same file.
# 5-fold CV + hold-out + progressive training sizes (--split all), then figures and the plain-text report.
#
# Submit:   sbatch slurm/run_final_wide.sbatch
# Resume (re-runs only failed / missing methods):   RESUME=1 sbatch slurm/run_final.sbatch
# Options: PARAMS (default data/results/crc_tma_tuned/method_params.json; falls back to configs/method_params.json)
#          OUT (data/results/crc_tma_final)   METHODS (all)   SCRIPT_RUNS (1; use 3 for stability numbers of the script methods)
#          REPEATS (1; 3 = whole run over 3 seeds)   JOBS (20 = one per method)   THREADS (2)   TIMEOUT (43200 s per method run)
# Read first: $OUT/summary/AI_REPORT.md (sanity checks at the top), $OUT/benchmark.log, $OUT/crc_tma/matrix_agreement.csv
#SBATCH --job-name=spc_final_wide
#SBATCH --output=logs/spc_final_wide_%j.out
#SBATCH --error=logs/spc_final_wide_%j.out
#SBATCH --cpus-per-task=45
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1

set -eo pipefail

PYTHON=${PYTHON:-/lustre/home/chaluvadig/anaconda3/envs/cta/bin/python}
[ -x "$PYTHON" ] || { echo "python not found: $PYTHON (pass PYTHON=<path>)"; exit 1; }
export PATH="$(dirname "$PYTHON"):$PATH"

cd /lustre/home/chaluvadig/VCU_HEALTH/spCellEval

DATA=${DATA:-/lustre/home/chaluvadig/VCU_HEALTH/spCellEval/data/datasets/crc_tma/quantification/processed/crc_tma_quantification.csv}
OUT=${OUT:-data/results/crc_tma_final}
PARAMS=${PARAMS:-data/results/crc_tma_tuned/method_params.json}
[ -f "$PARAMS" ] || { echo "no tuned parameters at $PARAMS: using the shipped configs/method_params.json"; PARAMS=configs/method_params.json; }
MARKER_MATRIX=${MARKER_MATRIX:-src/methods/TACIT/crc_tma_decision_matrix_level3.csv}
METHODS=${METHODS:-all}
SCRIPT_RUNS=${SCRIPT_RUNS:-1}
REPEATS=${REPEATS:-1}
JOBS=${JOBS:-20}
THREADS=${THREADS:-2}
TIMEOUT=${TIMEOUT:-43200}

[ -f "$DATA" ] || { echo "data not found: $DATA"; exit 1; }
[ -f "$MARKER_MATRIX" ] || { echo "marker matrix not found: $MARKER_MATRIX"; exit 1; }
mkdir -p "$OUT" logs
if [ -n "${RESUME:-}" ] && [ -f "$OUT/method_params.json" ]; then
  echo "resuming: keeping $OUT/method_params.json"
else
  cp "$PARAMS" "$OUT/method_params.json"                  # this run's own copy; it is also what params_used.json reports
fi
EXTRA=(); [ -n "${RESUME:-}" ] && EXTRA+=(--resume)

echo "python=$PYTHON"; echo "data=$DATA"; echo "out=$OUT  params=$OUT/method_params.json (from $PARAMS)"
echo "methods=$METHODS script-runs=$SCRIPT_RUNS repeats=$REPEATS jobs=$JOBS threads=$THREADS timeout=${TIMEOUT}s"
nvidia-smi -L 2>/dev/null || echo "no GPU visible (the GPU methods fall back to CPU)"
export SPC_NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)      # >1: the script methods are spread over the GPUs in turn
echo "gpus=$SPC_NUM_GPUS cpus=${SLURM_CPUS_PER_TASK:-?} jobs=$JOBS threads=$THREADS"

"$PYTHON" -u cli.py params --check --params "$OUT/method_params.json"

"$PYTHON" -u cli.py pipeline --data "$DATA" --modality crc_tma --transform none \
  --marker-matrix "$MARKER_MATRIX" --params "$OUT/method_params.json" "${EXTRA[@]}" \
  --stages benchmark,visualize --split all --methods "$METHODS" --script-runs "$SCRIPT_RUNS" --repeats "$REPEATS" \
  --timeout "$TIMEOUT" --jobs "$JOBS" --model-threads "$THREADS" --order slow-first --device auto \
  --out "$OUT" || rc=$?

# every finished method is saved on its own, so figures can be rebuilt from a partial or failed run
if [ "${rc:-0}" -ne 0 ]; then
  echo "pipeline ended with an error (exit $rc): building the figures and reports from the results saved so far"
  "$PYTHON" -u cli.py visualize --results "$OUT" || true
fi

echo
echo "================ done ================"
echo "report (paste into an AI model): $OUT/summary/AI_REPORT.md   (short: AI_REPORT_SHORT.md)"
echo "figures: $OUT/figures and $OUT/<dataset>/figures"
echo "parameters used: $OUT/crc_tma/params_used.json"
echo
sed -n '/^## Sanity checks/,/^## /p' "$OUT/summary/AI_REPORT_SHORT.md" 2>/dev/null | head -40 || true

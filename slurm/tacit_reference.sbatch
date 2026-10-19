#!/bin/bash
# The TACIT developers' own CRC pipeline (TACIT/Code generate figure/runTACIT_paper.R) on the RAW expression file, scored against the original
# ClusterName labels. Compare its macro-F1 / accuracy with the benchmark's TACIT (0.468 / 0.653 in data/results/crc_tma_final).
#
# Submit:   mkdir -p logs && sbatch slurm/tacit_reference.sbatch
# Options:  R_THREADS (16)   OUT (data/results/tacit_reference)   R and P (200 / 20, the paper's CRC settings)
# Output:   $OUT/tacit_reference_predictions.csv, $OUT/tacit_reference_per_class.csv, and the summary printed at the end of the log.
# No GPU is needed: if your cluster has a CPU partition, change --partition and drop --gres.
#SBATCH --job-name=spc_tacit_ref
#SBATCH --output=logs/spc_tacit_ref_%j.out
#SBATCH --error=logs/spc_tacit_ref_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1

set -eo pipefail
cd /lustre/home/chaluvadig/VCU_HEALTH/spCellEval

RSCRIPT=${RSCRIPT:-/lustre/home/chaluvadig/anaconda3/envs/cta/bin/Rscript}
[ -x "$RSCRIPT" ] || { echo "Rscript not found: $RSCRIPT (pass RSCRIPT=<path>)"; exit 1; }
export PATH="$(dirname "$RSCRIPT"):$PATH"
OUT=${OUT:-data/results/tacit_reference}
RAW=${RAW:-data/datasets/crc_tma/quantification/raw/CRC_TMA_expression.csv}
SIG=${SIG:-src/methods/TACIT/crc_tma_decision_matrix_level3.csv}
export MC_CORES=${R_THREADS:-16} OMP_NUM_THREADS=${R_THREADS:-16}      # TACIT uses parallelly::availableCores() - 1

[ -f "$RAW" ] || { echo "raw file not found: $RAW"; exit 1; }
[ -f TACIT/R/TACIT_function.R ] || { echo "vendored TACIT code not found: TACIT/R/TACIT_function.R"; exit 1; }
mkdir -p "$OUT" logs
echo "rscript=$RSCRIPT raw=$RAW signature=$SIG out=$OUT threads=$MC_CORES r=${R:-200} p=${P:-20}"

"$RSCRIPT" scripts/tacit_paper_reference.R "$RAW" "$SIG" "$OUT" "${R:-200}" "${P:-20}"

echo
echo "================ done ================"
echo "benchmark TACIT for comparison: macro-F1 0.468, accuracy 0.653, unassigned 0.104 (data/results/crc_tma_final)"
echo "reference results: $OUT/tacit_reference_per_class.csv"

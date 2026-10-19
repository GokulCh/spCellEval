#!/bin/bash
# Annotate a NEW unlabelled dataset (python cli.py annotate).
# First pass:   mkdir -p logs && DATA=<table.csv> MARKER_MATRIX=<signature.csv> OUT=data/results/new_annot sbatch slurm/annotate.sbatch
# After expert review of $OUT/review_table.csv:   same command plus  REVIEW=$OUT/review_table.csv
# Other knobs:  RESOLUTION=10  TRANSFORM=auto|arcsinh|none  COFACTOR=5  MODALITY=codex  SEED=0  PYTHON=<env python>
#SBATCH --job-name=spc_annotate
#SBATCH --output=logs/annotate_%j.out
#SBATCH --error=logs/annotate_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
##SBATCH --partition=<your partition>

set -eo pipefail

PYTHON=${PYTHON:-/lustre/home/chaluvadig/anaconda3/envs/cta/bin/python}
[ -x "$PYTHON" ] || { echo "python not found: $PYTHON (pass PYTHON=<path of 'which python' in your env>)"; exit 1; }
export PATH="$(dirname "$PYTHON"):$PATH"
cd /lustre/home/chaluvadig/VCU_HEALTH/spCellEval

: "${DATA:?set DATA=<quantification table>}"
: "${MARKER_MATRIX:?set MARKER_MATRIX=<signature csv>}"
OUT=${OUT:-data/results/annotate}
CPUS=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS OPENBLAS_NUM_THREADS=$CPUS
for f in "$DATA" "$MARKER_MATRIX"; do [ -f "$f" ] || { echo "not found: $f"; exit 1; }; done
EXTRA=()
[ -n "${REVIEW:-}" ] && { [ -f "$REVIEW" ] || { echo "review file not found: $REVIEW"; exit 1; }; EXTRA+=(--review "$REVIEW"); }

echo "python=$PYTHON cpus=$CPUS data=$DATA matrix=$MARKER_MATRIX out=$OUT review=${REVIEW:-none}"
"$PYTHON" -u cli.py annotate --data "$DATA" --marker-matrix "$MARKER_MATRIX" --out "$OUT" \
  --resolution "${RESOLUTION:-10}" --modality "${MODALITY:-codex}" --transform "${TRANSFORM:-auto}" \
  --cofactor "${COFACTOR:-5}" --seed "${SEED:-0}" --jobs "$CPUS" "${EXTRA[@]}"

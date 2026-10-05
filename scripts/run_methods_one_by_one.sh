#!/usr/bin/env bash
# Run every method on all three splits (5-fold CV, 80/20 hold-out, progressive 1-80%), one method per run,
# then merge the runs so rankings, per-class results and figures cover all methods together.
#
#   bash scripts/run_methods_one_by_one.sh                 # sequential
#   SLURM=1 bash scripts/run_methods_one_by_one.sh         # one sbatch job per method (merge job runs after they finish)
#
# Edit the variables below. Methods that cannot run on the machine are recorded as "skipped" with the reason.
set -euo pipefail

DATA="${DATA:-/lustre/home/chaluvadig/VCU_HEALTH/spCellEval/data/datasets/crc_tma/quantification/processed/crc_tma_quantification.csv}"
OUT="${OUT:-data/results/crc_tma_by_method}"       # one sub-folder per method
MERGED="${MERGED:-data/results/crc_tma_merged}"
MODALITY="${MODALITY:-crc_tma}"
TRANSFORM="${TRANSFORM:-none}"                       # the converted table already has arcsinh applied
FRACTIONS="${FRACTIONS:-0.01,0.05,0.10,0.25,0.50,0.80}"
TIMEOUT="${TIMEOUT:-3600}"
MARKER_MATRIX="${MARKER_MATRIX:-}"                   # decision-matrix CSV, needed by marker_score / tacit / scyan / astir

METHODS=(logistic_regression random_forest xgboost most_frequent stratified svm singler spatial_gnn knn_smooth
         louvain spade leiden flowsom marker_score tacit scyan astir tribus starling maps scanvi scarches)

if [[ -n "${METHODS_LIST:-}" ]]; then read -ra METHODS <<< "$METHODS_LIST"; fi   # e.g. METHODS_LIST="svm random_forest"

run_one() {
  local m="$1" extra=()
  if [[ -n "$MARKER_MATRIX" ]]; then extra=(--marker-matrix "$MARKER_MATRIX"); fi
  python cli.py benchmark --data "$DATA" --modality "$MODALITY" --transform "$TRANSFORM" \
      --methods "$m" --split all --folds 5 --fractions "$FRACTIONS" --timeout "$TIMEOUT" --script-runs 3 \
      --no-plots "${extra[@]}" --out "$OUT/$m"
}

if [[ "${1:-}" == "--one" ]]; then run_one "$2"; exit 0; fi      # used by the sbatch jobs

if [[ "${SLURM:-0}" == "1" ]]; then
  ids=()
  for m in "${METHODS[@]}"; do
    ids+=("$(sbatch --parsable --job-name="spc_$m" --wrap "cd $PWD && METHODS_LIST='${METHODS_LIST:-}' DATA='$DATA' OUT='$OUT' MODALITY='$MODALITY' TRANSFORM='$TRANSFORM' FRACTIONS='$FRACTIONS' TIMEOUT='$TIMEOUT' MARKER_MATRIX='$MARKER_MATRIX' bash scripts/run_methods_one_by_one.sh --one $m")")
  done
  sbatch --dependency=afterany:"$(IFS=:; echo "${ids[*]}")" --job-name=spc_merge \
         --wrap "cd $PWD && python cli.py merge --inputs $OUT/* --out $MERGED"
  exit 0
fi

for m in "${METHODS[@]}"; do
  echo "================ $m ================"
  run_one "$m" || echo "!! $m exited with an error (continuing)"
done
python cli.py merge --inputs "$OUT"/* --out "$MERGED"

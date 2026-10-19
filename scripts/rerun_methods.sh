#!/bin/bash
# Re-run chosen methods of an existing benchmark folder with the current configs/method_params.json, keeping every other result.
#   bash scripts/rerun_methods.sh singler marker_score tribus svm logistic_regression
# Options: OUT (data/results/crc_tma_final)   PARAMS (configs/method_params.json)   METHODS (the full list below)
# What it does: copies the parameters into $OUT, deletes only the chosen methods' saved results, then submits the wide run with RESUME=1.
# The full method list is passed on purpose: the combined result files are rebuilt from the methods named in a run.
set -eo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-data/results/crc_tma_final}
PARAMS=${PARAMS:-configs/method_params.json}
ALL=${METHODS:-leiden,louvain,flowsom,spade,starling,xgboost,random_forest,logistic_regression,svm,knn_smooth,spatial_gnn,singler,most_frequent,stratified,marker_score,tacit,scyan,astir,tribus,maps,stellar}
[ "$#" -ge 1 ] || { echo "usage: bash scripts/rerun_methods.sh <method> [<method> ...]"; exit 1; }
[ -d "$OUT" ] || { echo "no results folder: $OUT"; exit 1; }
cp "$PARAMS" "$OUT/method_params.json"
for m in "$@"; do
  d=$(ls -d "$OUT"/*/_methods/"$m" 2>/dev/null || true)
  [ -n "$d" ] && { rm -rf $d; echo "cleared saved results of $m"; } || echo "$m had no saved results (it will just run)"
done
RESUME=1 METHODS="$ALL" PARAMS="$PARAMS" OUT="$OUT" sbatch slurm/run_final_wide.sbatch

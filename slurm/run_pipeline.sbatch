#!/bin/bash
# Submit with:   mkdir -p logs && sbatch slurm/run_pipeline.sbatch
# Override without editing:   JOBS=8 THREADS=6 ORDER=slow-first OUT=data/results/crc_tma_run2 sbatch slurm/run_pipeline.sbatch
# Marker matrix:              defaults to data/datasets/crc_tma/quantification/processed/crc_tma_marker_matrix.csv; another file: MARKER_MATRIX=<csv> sbatch ...; none (circular auto-draft): MARKER_MATRIX= sbatch ...
#                             (without one, TACIT / Scyan / Astir / marker_score run on a matrix drafted from the labels; flagged as circular)
# Different size:             sbatch --cpus-per-task=64 --mem=384G slurm/run_pipeline.sbatch     (JOBS and THREADS follow automatically)
# Different environment:      PYTHON=/path/to/env/bin/python sbatch slurm/run_pipeline.sbatch   (find it with `which python` inside the env)
#SBATCH --job-name=spceleval
#SBATCH --output=logs/spceleval_%j.out
#SBATCH --error=logs/spceleval_%j.out
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=<your email>
##SBATCH --partition=<your partition>

set -eo pipefail

# The environment's own python: no `conda activate` / conda.sh needed (batch jobs do not load your shell setup).
PYTHON=${PYTHON:-/lustre/home/chaluvadig/anaconda3/envs/cta/bin/python}
if [ ! -x "$PYTHON" ]; then
  echo "python not found: $PYTHON"
  echo "Run 'which python' inside your activated environment and pass it:  PYTHON=<that path> sbatch slurm/run_pipeline.sbatch"
  exit 1
fi
export PATH="$(dirname "$PYTHON"):$PATH"          # so the environment's Rscript (FlowSOM, TACIT) and other tools are found too

cd /lustre/home/chaluvadig/VCU_HEALTH/spCellEval

DATA=${DATA:-/lustre/home/chaluvadig/VCU_HEALTH/spCellEval/data/datasets/crc_tma/quantification/processed/crc_tma_quantification.csv}
OUT=${OUT:-data/results/crc_tma}
CPUS=${SLURM_CPUS_PER_TASK:-32}
JOBS=${JOBS:-$(( CPUS / 2 ))}                     # methods running at the same time; only ~15 methods can run here
[ "$JOBS" -lt 1 ] && JOBS=1
# Threads inside the methods that can use them (random forest, XGBoost, logistic regression, knn_smooth). Most methods
# (Leiden, Louvain, FlowSOM, SPADE, SVM ...) are single-threaded and ignore this, and only ~5 methods use it at the same
# time, so it can be much larger than CPUS/JOBS without oversubscribing the allocation in practice.
THREADS=${THREADS:-$(( CPUS / 8 ))}
[ "$THREADS" -lt 1 ] && THREADS=1
TIMEOUT=${TIMEOUT:-14400}                         # seconds per method run; Astir / Scyan / STARLING exceeded 3600 on ~200k cells
ORDER=${ORDER:-fast-first}                        # fast-first: quick results early | slow-first: usually the shortest total time | listed

MARKER_MATRIX=${MARKER_MATRIX-data/datasets/crc_tma/quantification/processed/crc_tma_marker_matrix.csv}   # unset -> the bundled file; set but empty -> none
EXTRA=()
[ -n "${RESUME:-}" ] && EXTRA+=(--resume)          # RESUME=1 sbatch ...: keep finished methods' results, re-run only failed / missing ones
[ -n "$MARKER_MATRIX" ] && { [ -f "$MARKER_MATRIX" ] || { echo "marker matrix not found: $MARKER_MATRIX"; exit 1; }; EXTRA+=(--marker-matrix "$MARKER_MATRIX"); }

echo "python=$PYTHON"
echo "cpus=$CPUS jobs=$JOBS threads-per-multithreaded-method=$THREADS order=$ORDER out=$OUT marker-matrix=${MARKER_MATRIX:-auto-draft (circular)}"

"$PYTHON" -u cli.py pipeline \
  --data "$DATA" \
  --modality crc_tma --transform none \
  --stages benchmark,visualize --split all --methods all \
  --timeout "$TIMEOUT" --jobs "$JOBS" --model-threads "$THREADS" --order "$ORDER" --script-runs 3 \
  "${EXTRA[@]}" \
  --out "$OUT"

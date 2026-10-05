#!/bin/bash
# Submit with:   mkdir -p logs && sbatch slurm/run_pipeline.sbatch
# Override without editing:   JOBS=8 ORDER=slow-first OUT=data/results/crc_tma_run2 sbatch slurm/run_pipeline.sbatch
# Different size:             sbatch --cpus-per-task=64 --mem=384G slurm/run_pipeline.sbatch     (JOBS and threads follow automatically)
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
JOBS=${JOBS:-$(( CPUS / 2 ))}                     # methods running at the same time (about 2 CPUs each); only ~15 methods can run here
[ "$JOBS" -lt 1 ] && JOBS=1
ORDER=${ORDER:-fast-first}                        # fast-first: quick results early | slow-first: usually the shortest total time | listed
THREADS=$(( CPUS / JOBS ))                        # threads inside each method (random forest, XGBoost, ...)
[ "$THREADS" -lt 1 ] && THREADS=1

echo "python=$PYTHON"
echo "cpus=$CPUS jobs=$JOBS threads-per-method=$THREADS order=$ORDER out=$OUT"

"$PYTHON" -u cli.py pipeline \
  --data "$DATA" \
  --modality crc_tma --transform none \
  --stages benchmark,visualize --split all --methods all \
  --timeout 3600 --jobs "$JOBS" --model-threads "$THREADS" --order "$ORDER" --script-runs 3 \
  --out "$OUT"

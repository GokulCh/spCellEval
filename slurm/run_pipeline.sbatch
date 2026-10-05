#!/bin/bash
# Submit with:   mkdir -p logs && sbatch slurm/run_pipeline.sbatch
# Override without editing:   JOBS=6 ORDER=slow-first OUT=data/results/crc_tma_run2 sbatch slurm/run_pipeline.sbatch
#SBATCH --job-name=spceleval
#SBATCH --output=logs/spceleval_%j.out
#SBATCH --error=logs/spceleval_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=<your email>
##SBATCH --partition=<your partition>

set -eo pipefail

source /lustre/home/chaluvadig/anaconda3/etc/profile.d/conda.sh
conda activate cta
cd /lustre/home/chaluvadig/VCU_HEALTH/spCellEval

DATA=${DATA:-/lustre/home/chaluvadig/VCU_HEALTH/spCellEval/data/datasets/crc_tma/quantification/processed/crc_tma_quantification.csv}
OUT=${OUT:-data/results/crc_tma}
JOBS=${JOBS:-8}                                   # methods running at the same time
ORDER=${ORDER:-fast-first}                        # fast-first: quick results early | slow-first: usually the shortest total time | listed
CPUS=${SLURM_CPUS_PER_TASK:-16}
THREADS=$(( CPUS / JOBS ))                        # threads inside each method (random forest, XGBoost, ...)
[ "$THREADS" -lt 1 ] && THREADS=1

echo "cpus=$CPUS jobs=$JOBS threads-per-method=$THREADS order=$ORDER out=$OUT"

python -u cli.py pipeline \
  --data "$DATA" \
  --modality crc_tma --transform none \
  --stages benchmark,visualize --split all --methods all \
  --timeout 3600 --jobs "$JOBS" --model-threads "$THREADS" --order "$ORDER" --script-runs 3 \
  --out "$OUT"

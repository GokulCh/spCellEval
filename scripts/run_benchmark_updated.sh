#!/bin/bash
#SBATCH --job-name=crc_benchmark_updated
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=50G

# Ensure logs directory exists
mkdir -p logs

# Activate your environment
source ~/.bashrc
conda activate spCellEval

# Run the benchmark command:
#  - --recreate_kfolds regenerates clean k-folds (marker-restricted, no
#    ClusterID/Profile_Homogeneity junk) under the new dataset_context markers
#  - the method set mirrors configs/benchmark.yaml (classic ML + reference
#    mapping + clustering + prior-knowledge methods)
python src/benchmark/run_benchmark.py --dataset CRC_TMA \
  --methods tacit xgboost random_forest svm leiden louvain flowsom astir \
             scyan ribca_adapted starling logistic_regression singler \
             scarches spade signature \
  --baseline_split --cross_validation --subsample_experiment --recreate_kfolds
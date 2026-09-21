#!/bin/bash
#SBATCH --job-name=crc_benchmark
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --mem=100G

# Ensure logs directory exists
mkdir -p logs

# Activate your environment
source ~/.bashrc
conda activate spCellEval

# Run the benchmark command
python src/benchmark/run_benchmark.py --dataset CRC_TMA \
  --methods xgboost random_forest svm logistic_regression \
  --baseline_split --cross_validation --subsample_experiment --recreate_kfolds

# python src/benchmark/run_benchmark.py --dataset CRC_TMA \
#   --baseline_split --cross_validation --subsample_experiment

# python src/benchmark/run_benchmark.py --dataset CRC_TMA --methods xgboost --parallel_jobs 1

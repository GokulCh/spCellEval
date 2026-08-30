"""SingleR-style reference mapping via Pearson correlation to mean profiles."""

import argparse
import os
import sys
from pathlib import Path

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_script_dir))
utils_dir = os.path.join(project_root, "methods", "utils")
sys.path.insert(0, utils_dir)

from reference_mapping import run_kfold_label_transfer, singler_predict  # noqa: E402

parser = argparse.ArgumentParser(description="SingleR-style label transfer on k-fold splits.")
parser.add_argument("kfold_dir", type=str, help="Directory with fold_*_{train,test}.csv files")
parser.add_argument("output_dir", type=str, help="Directory for predictions_*.csv outputs")
parser.add_argument("labels_path", type=str, help="labels_*.csv with label/phenotype columns")
parser.add_argument("-m", "--markers", dest="markers", nargs="+", required=True)
args = parser.parse_args()


def main():
    run_kfold_label_transfer(
        Path(args.kfold_dir),
        Path(args.labels_path),
        Path(args.output_dir),
        args.markers,
        singler_predict,
    )


if __name__ == "__main__":
    main()

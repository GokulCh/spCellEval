import argparse
import os
import sys
from pathlib import Path
from data_handler import DataSetHandler

# Ground-truth stripping (Stage 2)
_STAGE2_DIR = Path(__file__).resolve().parents[2] / "pseudo_labeling"
if str(_STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(_STAGE2_DIR))
from ground_truth import DEFAULT_EVAL_DROP_COLUMNS  # noqa: E402


def _resolve_drop_columns(drop_columns, strip_labels: bool, phenotype_column: str = "cell_type"):
    """Merge explicit drop_columns with automatic label stripping.

    The *phenotype_column* is kept in the DataFrame during preprocessing so it
    can be encoded as ``Y``; ``DataSetHandler.preprocess`` removes it from ``X``.
    """
    if not strip_labels:
        return drop_columns

    auto_drop = [
        c for c in DEFAULT_EVAL_DROP_COLUMNS
        if c not in ("Cell_ID", "x", "y", "batch_id", phenotype_column)
    ]
    if drop_columns is None:
        return auto_drop

    if isinstance(drop_columns, str):
        drop_columns_clean = drop_columns.strip()
        if "," in drop_columns_clean:
            explicit = [s.strip() for s in drop_columns_clean.split(",")]
        else:
            explicit = [drop_columns_clean]
    else:
        explicit = list(drop_columns)

    merged = list(dict.fromkeys(explicit + auto_drop))
    return merged


def run_fold_creation(
    main_dir,
    dataset_name,
    dropna,
    impute_value,
    phenotype_column,
    batch_identifier_column,
    drop_columns,
    drop_non_numerical,
    n_splits,
    method,
    group_shuffle_split_size,
    swap_train_test,
    random_state,
    percentage_validation,
    strip_labels=False,
    rare_fraction=0.01,
    common_fraction=0.05,
    methods=None,
):
    """
    Create k-fold splits. When *methods* is provided, preprocess the quant CSV once
    and emit every listed split strategy in a single pass.
    """
    fold_methods = list(methods) if methods else [method]

    if impute_value is not None:
        impute_value = float(impute_value)

    drop_columns = _resolve_drop_columns(drop_columns, strip_labels, phenotype_column)

    if drop_columns is not None and isinstance(drop_columns, str):
        drop_columns_clean = drop_columns.strip()
        if ',' in drop_columns_clean:
            drop_columns = [s.strip() for s in drop_columns_clean.split(',')]
        else:
            drop_columns = drop_columns_clean

    if strip_labels:
        print(f"strip_labels=True — auto-dropping label columns: {drop_columns}")

    if phenotype_column == 'cell_type':
        granularity_level = 'level3'
    elif phenotype_column == 'level_2_cell_type':
        granularity_level = 'level2'
    elif phenotype_column == 'level_1_cell_type':
        granularity_level = 'level1'

    def _create_for_dataset(ds_name: str, save_dir: str, dataset_path: str) -> None:
        data_handler = DataSetHandler(dataset_path, random_state=random_state)
        data_handler.preprocess(
            dropna,
            impute_value,
            phenotype_column,
            batch_identifier_column,
            drop_columns=drop_columns,
            drop_non_numerical=drop_non_numerical,
        )
        representation_saved = False
        for fold_method in fold_methods:
            kfold_path = os.path.join(save_dir, f'kfolds_{fold_method}_{granularity_level}')
            if os.path.isdir(kfold_path) and any(
                name.startswith("fold_") and name.endswith("_train.csv")
                for name in os.listdir(kfold_path)
            ):
                print(f"Skipping existing folds: {kfold_path}")
                continue

            print(f"Creating {fold_method} folds for {ds_name}")
            data_handler.createFolds(
                n_splits,
                fold_method,
                batch_identifier_column,
                group_shuffle_split_size,
                swap_train_test,
                rare_fraction=rare_fraction,
                common_fraction=common_fraction,
            )
            data_handler.save_labels(save_dir)
            data_handler.save_folds(save_dir)
            if not representation_saved:
                data_handler.save_cell_type_representation(save_dir)
                representation_saved = True
            data_handler.create_validation_set_from_fold(
                save_path=kfold_path,
                percentage_validation=percentage_validation,
            )

    # Loop through datasets
    if dataset_name is None:
        for dataset in os.listdir(os.path.join(main_dir, 'data', 'processed')):
            if dataset.startswith('.'):
                continue
            if not os.path.isdir(os.path.join(main_dir, 'data', 'processed', dataset)):
                continue

            print(f"Processing dataset {dataset}")
            dataset_path = os.path.join(main_dir, 'data', 'processed', dataset, f'{dataset}_quantification.csv')
            save_dir = os.path.join(main_dir, 'data', 'processed', dataset)
            _create_for_dataset(dataset, save_dir, dataset_path)
    else:
        if os.path.isdir(os.path.join(main_dir, 'data', 'processed', dataset_name)):
            print(f"Processing {dataset_name}")
            dataset_path = os.path.join(main_dir, 'data', 'processed', dataset_name, f'{dataset_name}_quantification.csv')
            save_dir = os.path.join(main_dir, 'data', 'processed', dataset_name)
            _create_for_dataset(dataset_name, save_dir, dataset_path)
        else:
            raise ValueError(f"{dataset_name} is not present among the datasets")
def main():
    
    parser = argparse.ArgumentParser(
        description = 'clean data, label target variable and create training, validation and test kfolds'
    )
    
    parser.add_argument(
        '--main_dir',
        type = str,
        help = """ Path to the main directory holding the folders 'datasets' and 'results'.
        . An explicit directory structure is required. See README for more information""",
        required = True,
    )
    parser.add_argument(
        '--dataset_name',
        type = str,
        default=None,
        help = "If method should be run on only 1 dataset, specify the name of the dataset. Default is None",
    )
    parser.add_argument(
        '--dropna',
        action='store_true',
        help='drop rows with missing values. Default is False',
    )
    parser.add_argument(
        '--impute_value',
        type=float,
        help='value to impute missing values. Default is None',
        default=None,
    )
    parser.add_argument(
        '--phenotype_column',
        type=str,
        help="""name of the column containing the target variable. Default is cell_type, which corresponds to level3 granularity.
        """,
        choices=['cell_type', 'level_2_cell_type', 'level_1_cell_type'],
        default='cell_type',
    )
    parser.add_argument(
        '--batch_identifier_column',
        type=str,
        help='Name of the column containing the batch identifier. Default is None',
        default=None,
    )
    parser.add_argument(
        '--drop_columns',
        type = str,
        help = 'columns to drop from the data. Comma-seperated strings. Example: column1,column2',
        default = None,
    )
    parser.add_argument(
        '--drop_non_numerical',
        action='store_true',
        help = 'drop non-numerical columns. Default is False',
    )
    parser.add_argument(
        '--n_splits',
        type = int,
        help = 'number of splits for the kfold. Default is 5',
        default = 5,
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        choices=['StratifiedKFold', 'ProgressiveKFold', 'StratifiedGroupKFold', 'GroupShuffleSplit'],
        default=None,
        help='Create multiple k-fold strategies in one preprocess pass.',
    )
    parser.add_argument(
        '--method',
        type = str,
        help = 'method to use for creating folds. Default is StratifiedKFold',
        choices=['StratifiedKFold', 'ProgressiveKFold', 'StratifiedGroupKFold', 'GroupShuffleSplit'],
        default = 'StratifiedKFold',
    )
    parser.add_argument(
        '--rare_fraction',
        type=float,
        default=0.01,
        help='Fraction threshold below which a cell type is labelled rare (default: 0.01).',
    )
    parser.add_argument(
        '--common_fraction',
        type=float,
        default=0.05,
        help='Fraction threshold at or above which a cell type is labelled common (default: 0.05).',
    )
    parser.add_argument(
        '--group_shuffle_split_size',
        type = float,
        help = 'size of the group shuffle split if GroupShuffleSplit was selected as method. Default is 0.5',
        default = 0.5,
    )
    parser.add_argument(
        '--swap_train_test',
        action='store_true',
        help = 'swap train and test sets. Default is False',
    )
    parser.add_argument(
        '--random_state',
        type = int,
        help = 'random state for reproducibility. Default is 42',
        default = 42,
    )
    parser.add_argument(
        '--strip_labels',
        action='store_true',
        help=(
            'Automatically drop ground-truth label columns (cell_type, level_1/2, '
            'cell_labels) from feature matrices to prevent data leakage. '
            'Stage 2 ground-truth management.'
        ),
    )
    parser.add_argument(
        '--percentage_validation',
        type = float,
        help = 'percentage of data to be used as validation set. Default is 0.15',
        default = 0.15,
    )

    args = parser.parse_args()
    run_fold_creation(
        args.main_dir, args.dataset_name, args.dropna, args.impute_value, args.phenotype_column,
        args.batch_identifier_column, args.drop_columns, args.drop_non_numerical, args.n_splits,
        args.method, args.group_shuffle_split_size, args.swap_train_test, args.random_state,
        args.percentage_validation, args.strip_labels, args.rare_fraction, args.common_fraction,
        methods=args.methods,
    )
    print("Done.")
if __name__ == '__main__':
    main()
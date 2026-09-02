import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split, GroupShuffleSplit, ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from typing import List, Optional

from kfold_strategies import (
    KFOLD_METHODS,
    build_cell_type_representation,
    fold_representation_table,
    progressive_kfold_splits,
)

class DataSetHandler:
    def __init__(self, path, random_state):
        self.file_path = path
        self.data = pd.read_csv(self.file_path)
        self.random_state = random_state
        self.X = None
        self.Y = None
        self.labels = None
        self.method = None
        self.granularity_level = None
        self.batch_identifier = None
        self.kfolds = None
        self.fold_indices = None
        self.fold_data = None
        print("DataSetHandler initialized successfully")
    
    
    def preprocess(self, dropna: bool, impute_value: float = None, phenotype_column: str = 'cell_type', batch_identifier_column: str = None, drop_columns: List[str] = None, drop_non_numerical: bool = False) -> None:
        """
        This function processes the laoded dataframe. If NA values are present, they can be dropped or imputed with a specified value. A group identifier can be
        specified so which will be kept even if non-numerical columns are dropped. The phenotype column is encoded and the data is split into X and Y. 
        """
        if phenotype_column == 'cell_type':
            self.granularity_level = 'level3'
        elif phenotype_column == 'level_2_cell_type':
            self.granularity_level = 'level2'
        elif phenotype_column == 'level_1_cell_type':
            self.granularity_level = 'level1'

        if not isinstance(dropna, bool):
            raise TypeError("dropna must be a boolean value")
        if not isinstance(impute_value, (float, int, type(None))):
            raise TypeError("impute_value must be a number")
        
        if dropna:
            self.data.dropna(inplace=True)
        if impute_value is not None:
            self.data.fillna(impute_value, inplace=True)
        if drop_columns is not None:
            if isinstance(drop_columns, str):
                drop_columns = [drop_columns]
            drop_columns = [c for c in drop_columns if c in self.data.columns]
            if drop_columns:
                self.data.drop(columns=drop_columns, inplace=True)

        label_encoder = LabelEncoder()
        self.Y = label_encoder.fit_transform(self.data[phenotype_column])
        self.labels = pd.DataFrame({
            'label': range(len(label_encoder.classes_)),
            'phenotype': label_encoder.classes_
            })
        
        # If group identifier is selected but is a non-numerical and user selects to drop non-numericals, it is stored in a separate variable and re-inserted to preserve it
        if drop_non_numerical:
            if batch_identifier_column is not None and self.data[batch_identifier_column].select_dtypes(include=[np.number]).empty:
                self.batch_identifier = self.data[batch_identifier_column]
                self.X = self.data.select_dtypes(include=[np.number])
                self.X.insert(len(self.X.columns), batch_identifier_column, self.batch_identifier)
            else:
                self.X  = self.data.select_dtypes(include=[np.number])
        else:
            self.X = self.data.drop(columns=[phenotype_column])

        print("Data successfully preprocessed")
    

    def createFolds(
        self,
        k: int,
        method: str,
        batch_identifier_column: str = None,
        group_shuffle_split_size: float = 0.5,
        swap_train_test: bool = False,
        *,
        rare_fraction: float = 0.01,
        common_fraction: float = 0.05,
    ) -> None:
        """
        Create k-fold splits.

        Supported methods: StratifiedKFold, ProgressiveKFold, StratifiedGroupKFold,
        GroupShuffleSplit (see ``kfold_strategies.KFOLD_METHODS``).
        """
        self.method = method
        self.rare_fraction = rare_fraction
        self.common_fraction = common_fraction
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        if method not in KFOLD_METHODS:
            raise ValueError(
                f"Invalid method '{method}'. Use one of: {', '.join(KFOLD_METHODS)}."
            )

        print(f'{method} method selected for creating folds')
        if method == 'ProgressiveKFold':
            self.kfolds = None
            self.fold_indices = progressive_kfold_splits(self.Y, k, random_state=self.random_state)
        elif method == 'StratifiedGroupKFold':
            if (batch_identifier_column is None) or (batch_identifier_column not in self.X.columns):
                print("batch_identifier_column not specified, reverting to StratifiedKFold")
                self.kfolds = StratifiedKFold(n_splits=k, random_state=self.random_state, shuffle=True)
                fold_generator = self.kfolds.split(self.X, self.Y)
            else:
                groups = self.X[batch_identifier_column]
                if groups.nunique() < k:
                    print(f"Only {groups.nunique()} unique group(s) in '{batch_identifier_column}', fewer than n_splits={k}, reverting to StratifiedKFold")
                    self.kfolds = StratifiedKFold(n_splits=k, random_state=self.random_state, shuffle=True)
                    fold_generator = self.kfolds.split(self.X, self.Y)
                else:
                    self.kfolds = StratifiedGroupKFold(n_splits=k, random_state=self.random_state, shuffle=True)
                    fold_generator = self.kfolds.split(self.X, self.Y, groups)
        elif method == 'StratifiedKFold':
            self.kfolds = StratifiedKFold(n_splits=k, random_state=self.random_state, shuffle=True)
            fold_generator = self.kfolds.split(self.X, self.Y)
        
        elif method == 'GroupShuffleSplit':
            if (batch_identifier_column is None) or (batch_identifier_column not in self.X.columns):
                print("batch_identifier_column not specified, reverting to ShuffleSplit")
                self.kfolds = ShuffleSplit(n_splits=k, test_size=group_shuffle_split_size, random_state=self.random_state)
                fold_generator = self.kfolds.split(self.X, self.Y)
            else:
                splitter = GroupShuffleSplit(n_splits=k, test_size=group_shuffle_split_size, random_state=self.random_state)
                groups = self.X[batch_identifier_column]
                fold_generator = splitter.split(self.X, self.Y, groups)
        else:
            raise ValueError(
                "Invalid method specified. Use one of: "
                + ", ".join(KFOLD_METHODS)
                + "."
            )

        if method != 'ProgressiveKFold':
            self.fold_indices = list(fold_generator)

        self.fold_data = []
        if swap_train_test:
            print("Swapping train and test sets in folds")
            self.fold_indices = [(test_index, train_index) for train_index, test_index in self.fold_indices]
        for train_index, test_index in self.fold_indices:
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            Y_train, Y_test = self.Y[train_index], self.Y[test_index]
            
            fold = {
                'X_train': X_train,
                'X_test': X_test,
                'Y_train': Y_train,
                'Y_test': Y_test
            }
            self.fold_data.append(fold)
        
        print(f"{k} folds created. To save the folds, call save_folds method.")

    def save_cell_type_representation(
        self,
        save_path: str,
        *,
        rare_fraction: Optional[float] = None,
        common_fraction: Optional[float] = None,
    ) -> pd.DataFrame:
        """Write global and per-fold cell-type abundance tables."""
        if self.labels is None or self.Y is None or self.fold_indices is None:
            raise ValueError("Call preprocess() and createFolds() before saving representation.")

        rare_fraction = self.rare_fraction if rare_fraction is None else rare_fraction
        common_fraction = self.common_fraction if common_fraction is None else common_fraction

        global_repr = build_cell_type_representation(
            self.labels,
            self.Y,
            rare_fraction=rare_fraction,
            common_fraction=common_fraction,
        )
        global_repr.to_csv(
            os.path.join(save_path, "cell_type_representation.csv"),
            index=False,
        )

        fold_tables = []
        for i, (train_idx, test_idx) in enumerate(self.fold_indices, start=1):
            fold_tables.append(
                fold_representation_table(
                    self.Y,
                    train_idx,
                    test_idx,
                    self.labels,
                    fold=i,
                    rare_fraction=rare_fraction,
                    common_fraction=common_fraction,
                )
            )
        fold_repr = pd.concat(fold_tables, ignore_index=True)
        kfolds_dir = os.path.join(save_path, f"kfolds_{self.method}_{self.granularity_level}")
        os.makedirs(kfolds_dir, exist_ok=True)
        fold_repr.to_csv(os.path.join(kfolds_dir, "fold_cell_type_representation.csv"), index=False)
        return global_repr

    def save_labels(self, save_path: str = None) -> None:
        if self.labels is None:
            raise ValueError("No labels have been created. Call preprocess first.")
        
        if save_path is None:
            save_path = os.getcwd()
        
        self.labels.to_csv(os.path.join(save_path, f'labels_{self.method}_{self.granularity_level}.csv'), index=False)
        print(f"Labels saved in: {save_path} as labels.csv")

    def save_folds(self, save_path: str = None) -> None:

        if self.fold_indices is None:
            raise ValueError("No folds have been created. Call createFolds first.")
        
        if save_path is None:
            save_path = os.getcwd()

        kfolds_dir = os.path.join(save_path, f'kfolds_{self.method}_{self.granularity_level}')
        os.makedirs(kfolds_dir, exist_ok=True)

        fold_meta = []
        for i, (train, test) in enumerate(self.fold_indices):
            entry = {'fold': i + 1, 'train': train.tolist(), 'test': test.tolist()}
            if self.method == 'ProgressiveKFold':
                entry['progressive_train_fraction'] = (i + 1) / len(self.fold_indices)
                entry['n_train'] = len(train)
                entry['n_test'] = len(test)
            fold_meta.append(entry)

        fold_data = {
            'random_state': self.random_state,
            'kfold_method': self.method,
            'folds': fold_meta,
        }


        with open(os.path.join(kfolds_dir, 'fold_indices.json'), 'w') as f:
            json.dump(fold_data, f)

        for i, fold in enumerate(self.fold_data):
            X_train, X_test = fold['X_train'], fold['X_test']
            Y_train, Y_test = fold['Y_train'], fold['Y_test']

            train_data = pd.concat([X_train.reset_index(drop=True), pd.Series(Y_train, name='encoded_phenotype').reset_index(drop=True)], axis=1)
            test_data = pd.concat([X_test.reset_index(drop=True), pd.Series(Y_test, name='encoded_phenotype').reset_index(drop=True)], axis=1)
            
            train_data.to_csv(os.path.join(kfolds_dir, f'fold_{i+1}_train.csv'), index=False)
            test_data.to_csv(os.path.join(kfolds_dir, f'fold_{i+1}_test.csv'), index=False)
        
        print(f"Folds saved in: {kfolds_dir}")

    def create_validation_set_from_fold(self, save_path: str, percentage_validation: float = 0.15):
        """
        Creates validation sets for each pre-existing training fold. The percentage of data to be used as validation can be specified. The folds have to be created before by calling createKfold
        and saved by calling save_folds. The pre-existing training folds will be overwritten.
        """
        # Load training folds and train test split
        if not os.path.exists(save_path):
            raise ValueError(f"The specified save_path {save_path} does not exist.")

        with open(os.path.join(save_path, 'fold_indices.json'), 'r') as f:
            fold_data = json.load(f)

        for i, fold in enumerate(fold_data['folds']):
            train_data_original = pd.read_csv(os.path.join(save_path, f'fold_{i+1}_train.csv'))
            try:
                train_data, validation_data = train_test_split(train_data_original, test_size=percentage_validation, random_state=self.random_state, stratify=train_data_original['encoded_phenotype'])
            except ValueError as e:
                print(f"Warning fold {i+1}: stratified split failed ({e}), falling back to random split")
                train_data, validation_data = train_test_split(train_data_original, test_size=percentage_validation, random_state=self.random_state)
            print(f"Overwriting 'fold_{i+1}_train.csv'... and saving validation")
            train_file_path = os.path.join(save_path, f'fold_{i+1}_train.csv')
            validation_file_path = train_file_path.replace('train.', 'validation.')
            validation_data.to_csv(validation_file_path, index=False)
            train_data.to_csv(train_file_path, index=False)

            train_indices = train_data.index.tolist()
            validation_indices = validation_data.index.tolist()

            fold['train'] = train_indices
            fold['validation'] = validation_indices

                # Save the updated fold indices back to the JSON file
        with open(os.path.join(save_path, 'fold_indices.json'), 'w') as f:
            json.dump(fold_data, f)

        print("Validation indices updated in fold_indices.json")

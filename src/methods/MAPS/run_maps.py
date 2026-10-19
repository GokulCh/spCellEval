# imports
import os
import numpy as np
import pandas as pd
from maps.cell_phenotyping import Trainer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import json
import glob
import re
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

def get_args():
    """
    ## Example usage:
    python run_maps.py ./benchmarking/data/processed_IMMUcan/kfolds_StratifiedGroupKFold_level3 ./benchmarking/results/IMMUcan/maps/level3 
    ./benchmarking/data/processed_IMMUcan/labels_StratifiedGroupKFold_level3.csv
    """
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Train k-folds on dataset')
    parser.add_argument('input_path', type=str, help='Path to the folder containing the folds data')
    parser.add_argument('results_dir', type=str, help='Path to the folder to save the results')
    parser.add_argument('label_path', type=str, help='Path to the file containing the labels')
    parser.add_argument('--max_epochs', type=int, default=2, help='Maximum training epochs (default: 2)')
    parser.add_argument('--min_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--patience', type=int, default=10, help='Early-stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.10)
    parser.add_argument('--standard_split', action='store_true', help='train on fold_N_train and predict fold_N_test. Default (as in the original script): train on fold_N_test and predict fold_N_train, i.e. the 20:80 protocol')
    args = parser.parse_args()

    return args



def main(args):

    # Define params
    batch_size, max_epochs, min_epochs, patience = args.batch_size, args.max_epochs, args.min_epochs, args.patience
    verbose = 0
    
    # Define the list of fold numbers
    fold_numbers = sorted(int(re.search(r"fold_(\d+)_", os.path.basename(f)).group(1)) for f in glob.glob(os.path.join(args.input_path, "fold_*_validation.csv")))   # any number of folds
    fold_accuracies, fold_f1_scores, fold_weighted_f1_scores, fold_precisions, fold_recalls = [], [], [], [], []

    # Initialize empty lists to store the train and test data
    train_data = []
    val_data = []
    test_data = []

    #Create folders
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    # Loop through each fold number
    for fold_number in fold_numbers:
        
        #get the time to calculate the time taken for each fold
        import time
        start_time = time.time()
        
        # Print the current fold numb
        print(f"Processing fold {fold_number}...")
        # Filter the files based on the fold number
        test_file = f"fold_{fold_number}_train.csv"
        val_file = f"fold_{fold_number}_validation.csv"
        train_file = f"fold_{fold_number}_test.csv"
        if args.standard_split:
            test_file, train_file = train_file, test_file

        # Read the train and test data files
        train_data_path = os.path.join(args.input_path, train_file)
        test_data_path = os.path.join(args.input_path, test_file)
        val_data_path = os.path.join(args.input_path, val_file)
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
        val_data = pd.read_csv(val_data_path)  

        print("Train data path: ", train_data_path)
        print("Test data path: ", test_data_path)
        print(train_data.shape)
        print(test_data.shape)

        drop_cols = [ 'x', 'y', 'csv', 'orig.ident','sample_name','tissue','donor','unique_region', 'File Name', 
                     'Point', 'Xcorr','Ycorr','array','SampleID','PatientROI','file_key', 'sample_id', 'cell_id',
                     'width_px', 'height_px', 'cell_labels', 'eccentricity', 'major_axis_length', 'image',
                        'minor_axis_length', 'area', 'perimeter', 'solidity', 'extent', 'orientation',
                      'cell_type', 'level_2_cell_type','level_1_cell_type']
        train_data = train_data.drop([x for x in drop_cols if x in train_data.columns], axis=1)
        test_data = test_data.drop([x for x in drop_cols if x in test_data.columns], axis=1)
        val_data = val_data.drop([x for x in drop_cols if x in val_data.columns], axis=1)
        print(train_data.shape)
        print(test_data.shape)
        
        num_features = train_data.shape[1] - 1 
        num_classes = int(max(d["encoded_phenotype"].max() for d in (train_data, val_data, test_data))) + 1   # labels are 0..K-1; counting only one split drops rare classes and breaks the output layer
        print("Num of features: ", num_features)
        print("Num of classes: ", num_classes)   
    
        model = Trainer(results_dir = args.results_dir, num_features= num_features, num_classes=num_classes, batch_size=batch_size, max_epochs=max_epochs, min_epochs=min_epochs, patience=patience, verbose=verbose,
                        learning_rate=args.learning_rate, dropout=args.dropout)
        # the MAPS Trainer takes CSV paths (fit/predict), not DataFrames: write the cleaned tables out first
        paths = {k: os.path.join(args.results_dir, f'_maps_{k}_fold{fold_number}.csv') for k in ('train', 'val', 'test')}
        for k, d in (('train', train_data), ('val', val_data), ('test', test_data)):
            d.rename(columns={'encoded_phenotype': 'cell_label'}).to_csv(paths[k], index=False)   # MAPS reads the label from 'cell_label'
        model.fit(paths['train'], paths['val'])

        #get the time taken for each fold
        end_time = time.time()
        elapsed_time = end_time - start_time
        #append time to the results
        with open(os.path.join(args.results_dir, 'fold_times.txt'), 'a') as f:
            f.write(f"Fold {fold_number} train_time: {elapsed_time:.2f} seconds\n")

        #get infernce time
        start_inference_time = time.time()
        # Make predictions on the test data
        pred_labels, pred_probs = model.predict(paths['test'])

        end_inference_time = time.time()
        inference_time = end_inference_time - start_inference_time
        #append inference time to the results
        with open(os.path.join(args.results_dir, 'fold_times.txt'), 'a') as f:
            f.write(f"Fold {fold_number} inference_time: {inference_time:.2f} seconds\n")

        
        Y_test = test_data["encoded_phenotype"]
        predictions = pred_labels

        # Calculate the metrics
        accuracy = accuracy_score(Y_test, predictions)
        f1 = f1_score(Y_test, predictions, average='macro')
        weighted_f1 = f1_score(Y_test, predictions, average='weighted')
        cm = confusion_matrix(Y_test, predictions)
        precision = precision_score(Y_test, predictions, average='weighted')
        recall = recall_score(Y_test, predictions, average='weighted')
        cr = classification_report(Y_test, predictions)

        print(f"Fold {fold_number} results:")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print("Accuracy:", accuracy)
        print("F1 Score:", f1)
        print("Weighted F1 Score:", weighted_f1)
        print("="*50)

        fold_accuracies.append(accuracy)
        fold_f1_scores.append(f1)
        fold_weighted_f1_scores.append(weighted_f1)
        fold_precisions.append(precision)
        fold_recalls.append(recall)

        # Load the labels
        labels = pd.read_csv(args.label_path)
        label_dict = dict(zip(labels['label'], labels['phenotype']))
        index = [label_dict[i] for i in range(0, len(label_dict))]

        labelled_Y_test = Y_test.map(label_dict)
        labelled_preds = pd.Series(predictions).map(label_dict)

        # predictions df output
        predictions_df = pd.DataFrame({
            'predictions': predictions,
            'predicted_phenotype': labelled_preds,
            'true_phenotype': labelled_Y_test
        })

        #merge the predictions with the test data
        predictions_df = pd.concat([test_data.reset_index(drop=True), predictions_df], axis=1)
        # Save the predictions to a CSV file
        predictions_df.to_csv(f"{args.results_dir}/predictions_fold_{fold_number}.csv", index=False)

        # Save the confusion matrix and classification report to separate files
        print("Saving results...")
        np.savetxt(f"{args.results_dir}/confusion_matrix_{fold_number}.csv", cm, delimiter=",", fmt="%d")
        with open(f"{args.results_dir}/classification_report_{fold_number}.txt", "w") as f:
            f.write(cr)
        print("Fold-{fold_number} results saved successfully!")    

    # Calculate the average metrics
    avg_accuracy = np.mean(fold_accuracies)
    avg_f1_score = np.mean(fold_f1_scores)
    avg_weighted_f1_score = np.mean(fold_weighted_f1_scores)
    avg_precision = np.mean(fold_precisions)
    avg_recall = np.mean(fold_recalls)

    print("Average Accuracy:", avg_accuracy)
    print("Average F1 Score:", avg_f1_score)
    print("Average Weighted F1 Score:", avg_weighted_f1_score)
    print("Average Precision:", avg_precision)
    print("Average Recall:", avg_recall)

    # This part saves the average results in a .json file
    avg_results = {
        'average_accuracy': avg_accuracy,
        'average_f1_score': avg_f1_score,
        'average_weighted_f1_score': avg_weighted_f1_score,
        'average_precision': avg_precision,
        'average_recall': avg_recall,
    }

    print("Saving average results...")
    with open(os.path.join(args.results_dir, 'average_maps_results.json'), 'w') as f:
        json.dump(avg_results, f, indent=4)
    print("Average results saved successfully!") 


if __name__ == '__main__':
    args = get_args()
    main(args)
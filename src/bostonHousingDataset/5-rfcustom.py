import os
import numpy as np
import pandas as pd
import argparse
import json
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from RandomForest import RandomForest

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_data(file_path):
    """
    Load Boston Housing data from a CSV file.
    Assumes the target variable is in the last column.
    """
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y


def main(args):
    create_dir_if_not_exists(args.log_dir)
    log_file = os.path.join(args.log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    # Load data
    X, y = load_data(args.data_path)

    # preprocessing
    X = X.fillna(X.median())
    X["DIS"] = np.log(X["DIS"])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Initialize K-Fold cross-validation
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    # Metrics storage
    fold_results = {
        "RMSE": [],
        "MAE": [],
        "R2": []
    }

    # Perform K-Fold CV
    for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Initialize customized RandomForest
        model = RandomForest(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=args.random_state,
            mode="regression"
        )

        # Train model
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Store metrics
        fold_results["RMSE"].append(rmse)
        fold_results["MAE"].append(mae)
        fold_results["R2"].append(r2)

        print(f"Fold {fold} Results:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R2: {r2:.4f}")

    # Summarize results
    results_summary = {
        "RMSE": f"{np.mean(fold_results['RMSE']):.4f} ± {np.std(fold_results['RMSE']):.4f}",
        "MAE": f"{np.mean(fold_results['MAE']):.4f} ± {np.std(fold_results['MAE']):.4f}",
        "R2": f"{np.mean(fold_results['R2']):.4f} ± {np.std(fold_results['R2']):.4f}",
    }

    print("\n=== Final Results (Cross-Validation) ===")
    print(f"RMSE: {results_summary['RMSE']}")
    print(f"MAE: {results_summary['MAE']}")
    print(f"R2: {results_summary['R2']}")

    # Save to log file
    with open(log_file, 'w') as f:
        json.dump({"args": vars(args), "results": results_summary, "fold_results": fold_results}, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boston Housing Regression with customized RandomForest.")
    parser.add_argument("--data_path", type=str, default="data/HousingData.csv", help="Path to the Boston Housing dataset.")
    parser.add_argument("--log_dir", type=str, default="results/BostonHousingDataset", help="Directory to save the log files.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of splits for K-Fold cross-validation.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility.")
    parser.add_argument("--n_estimators", type=int, default=200, help="Number of trees in RandomForest.")
    parser.add_argument("--min_samples_split", type=int, default=2, help="Minimum samples required to split an internal node in RandomForest.")
    parser.add_argument("--max_depth", type=int, default=6, help="Maximum depth of the tree.")

    args = parser.parse_args()
    main(args)
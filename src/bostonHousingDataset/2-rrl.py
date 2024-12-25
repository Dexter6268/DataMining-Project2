import os
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler


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


def save_rrl_format(X, y, data_path, info_path):
    """
    Save data and labels in RRL format.
    """
    # Save to *.data
    with open(data_path, "w") as data_file:
        for features, label in zip(X, y):
            row = ", ".join(map(str, features)) + f", {label}\n"
            data_file.write(row)

    # Save to *.info
    with open(info_path, "w") as info_file:
        for i in range(X.shape[1]):
            info_file.write(f"feature_{i+1}, continuous\n")
        info_file.write("class, continuous\n")  # For regression tasks, label is continuous
        info_file.write("LABEL_POS, -1\n")


def main(args):
    create_dir_if_not_exists(args.output_dir)

    # Load data
    X, y = load_data(args.data_path)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the entire dataset in RRL format
    save_rrl_format(
        X_scaled, y,
        data_path=os.path.join(args.output_dir, "bostonHousing.data"),
        info_path=os.path.join(args.output_dir, "bostonHousing.info")
    )

    print(f"Dataset has been saved in RRL format in {args.output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Boston Housing dataset to RRL format.")
    parser.add_argument("--data_path", type=str, default="data/HousingData.csv", help="Path to the Boston Housing dataset.")
    parser.add_argument("--output_dir", type=str, default="rrl/dataset", help="Directory to save the RRL formatted dataset.")
    
    args = parser.parse_args()
    main(args)
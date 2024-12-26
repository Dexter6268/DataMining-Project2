import os
import numpy as np
import xgboost as xgb
import argparse
import json
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def parse_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data_mode = 0
    labels = []
    datas = []

    for line in lines:
        if not data_mode:
            if "cases = (" in line:
                data_mode = 1
            continue
        cleaned_data = line.strip(" [] \n").split(", ")
        if len(cleaned_data) <= 1:
            continue
        if cleaned_data[0] == "relapse":
            labels.append(1)
        else:
            labels.append(0)
        cleaned_data = cleaned_data[1:]
        parsed_data = [None if item == "?" else float(item) for item in cleaned_data]
        datas.append(parsed_data)

    return datas, labels


def advanced_preprocessing(train_data, test_data, n_components=10):
    train_array = np.array(train_data, dtype=object)
    train_array = np.where(train_array == None, np.nan, train_array).astype(float)

    test_array = np.array(test_data, dtype=object)
    test_array = np.where(test_array == None, np.nan, test_array).astype(float)

    not_all_nan_mask = ~np.isnan(train_array).all(axis=0)
    train_array = train_array[:, not_all_nan_mask]
    test_array = test_array[:, not_all_nan_mask]

    var_train = np.nanvar(train_array, axis=0)
    not_zero_var_mask = (var_train != 0)
    train_array = train_array[:, not_zero_var_mask]
    test_array = test_array[:, not_zero_var_mask]

    imputer = SimpleImputer(strategy='median')
    imputer.fit(train_array)
    train_array = imputer.transform(train_array)
    test_array = imputer.transform(test_array)

    scaler = StandardScaler()
    scaler.fit(train_array)
    train_array = scaler.transform(train_array)
    test_array = scaler.transform(test_array)

    if n_components is not None and n_components > 0 and n_components < train_array.shape[1]:
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(train_array)
        train_array = pca.transform(train_array)
        test_array = pca.transform(test_array)

    return train_array, test_array


def main(args):
    create_dir_if_not_exists(args.log_dir)
    log_file = os.path.join(args.log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    train_data, train_label = parse_file(args.train_path)
    test_data, test_label = parse_file(args.test_path)

    train_label = np.array(train_label, dtype=int)
    test_label = np.array(test_label, dtype=int)

    train_data_processed, test_data_processed = advanced_preprocessing(
        train_data, 
        test_data, 
        n_components=args.pca_components
    )

    clf = xgb.XGBClassifier(
        random_state=args.random_state,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators
    )
    clf.fit(train_data_processed, train_label)

    y_pred = clf.predict(test_data_processed)
    y_prob = clf.predict_proba(test_data_processed)[:, 1]

    accuracy = accuracy_score(test_label, y_pred)
    precision = precision_score(test_label, y_pred, zero_division=0)
    recall = recall_score(test_label, y_pred, zero_division=0)
    f1 = f1_score(test_label, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(test_label, y_prob)
    except ValueError:
        roc_auc = float('nan')
    cm = confusion_matrix(test_label, y_pred)

    results = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "Confusion Matrix": cm.tolist(),
        "Classification Report": classification_report(test_label, y_pred, zero_division=0, output_dict=True)
    }

    with open(log_file, 'w') as f:
        json.dump({"args": vars(args), "results": results}, f, indent=4)

    print("=== 测试集指标 ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print("混淆矩阵:")
    print(cm)

    print("\n=== classification_report ===")
    print(classification_report(test_label, y_pred, zero_division=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost on Breast Cancer Dataset.")
    parser.add_argument("--train_path", type=str, default="data/breastCancer/breastCancer_Train.dbc", help="Path to the training dataset.")
    parser.add_argument("--test_path", type=str, default="data/breastCancer/breastCancer_Test.dbc", help="Path to the test dataset.")
    parser.add_argument("--log_dir", type=str, default="results/breastCancer/xgboost", help="Directory to save the log files.")
    parser.add_argument("--pca_components", type=int, default=0.99, help="Number of PCA components.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility.")
    parser.add_argument("--max_depth", type=int, default=6, help="Maximum tree depth in XGBoost.")
    parser.add_argument("--learning_rate", type=float, default=0.3, help="Learning rate for XGBoost.")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in XGBoost.")

    args = parser.parse_args()
    main(args)
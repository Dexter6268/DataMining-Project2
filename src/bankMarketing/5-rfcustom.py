import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import argparse
import os
from datetime import datetime
import json
from imblearn.over_sampling import SMOTE
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from RandomForest import RandomForest

# 创建目录函数
def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 主函数
def main(args):
    # 创建结果目录
    log_dir = "results/bankMarketing"
    create_dir_if_not_exists(log_dir)
    log_file = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    # 1. 读取数据
    data = pd.read_csv(args.data_path, sep=';')

    # 2. 数据预处理
    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    numeric_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    X = data.drop('y', axis=1)
    y = data['y']

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    # 初始化评估指标
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    roc_aucs = []
    confusion_matrices = []

    # 交叉验证过程
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        smote = SMOTE(sampling_strategy='auto', random_state=args.random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        # RandomForest 模型
        model = RandomForest(
            random_state=args.random_state,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            n_estimators=args.n_estimators,
            mode="classification"
        )
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # 评估指标
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        roc_aucs.append(roc_auc_score(y_test, y_prob))
        confusion_matrices.append(confusion_matrix(y_test, y_pred))

    # 汇总评估结果
    results = {
        "Accuracy": f"{np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}",
        "Precision": f"{np.mean(precisions):.4f} ± {np.std(precisions):.4f}",
        "Recall": f"{np.mean(recalls):.4f} ± {np.std(recalls):.4f}",
        "F1-Score": f"{np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}",
        "ROC-AUC": f"{np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}",
        "Confusion Matrices": [cm.tolist() for cm in confusion_matrices]
    }

    # 保存日志
    with open(log_file, 'w') as f:
        json.dump({"args": vars(args), "results": results}, f, indent=4)

    # 打印评估结果
    print("5-Fold Cross-Validation Results:")
    print(f"Accuracy: {results['Accuracy']}")
    print(f"Precision: {results['Precision']}")
    print(f"Recall: {results['Recall']}")
    print(f"F1-Score: {results['F1-Score']}")
    print(f"ROC-AUC: {results['ROC-AUC']}")

    print("\nConfusion Matrices for each fold:")
    for i, cm in enumerate(confusion_matrices, 1):
        print(f"Fold {i}:\n{cm}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RandomForest on Bank Marketing Dataset.")
    parser.add_argument("--data_path", type=str, default="data/bank+marketing/bank/bank-full.csv", help="Path to the dataset.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of splits for Stratified K-Fold.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility.")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum tree depth in RandomForest.")
    parser.add_argument("--min_samples_split", type=int, default=2, help="Minimum samples required to split an internal node in RandomForest.")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in RandomForest.")
    
    args = parser.parse_args()
    main(args)
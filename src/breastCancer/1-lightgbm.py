import os
import numpy as np
import lightgbm as lgb

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

data_dir = "data/breastCancer"
train_path = os.path.join(data_dir, "breastCancer_Train.dbc")
test_path = os.path.join(data_dir, "breastCancer_Test.dbc")


def parse_file(file_path):
    """
    读取 dbc 文件，返回 data, label 两个列表。
    data 的元素为特征向量(list)，label 为二分类标签(0或1)。
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data_mode = 0
    labels = []
    datas = []

    for line in lines:
        if not data_mode:
            # 当出现 "cases = (" 时，开始读取数据
            if "cases = (" in line:
                data_mode = 1
            continue
        cleaned_data = line.strip(" [] \n").split(", ")
        if len(cleaned_data) <= 1:
            continue
        # 第一个元素表示标签
        if cleaned_data[0] == "relapse":
            labels.append(1)
        else:
            labels.append(0)
        # 后面的元素是特征
        cleaned_data = cleaned_data[1:]
        # 将 "?" 替换为 None
        parsed_data = [None if item == "?" else float(item) for item in cleaned_data]
        datas.append(parsed_data)

    return datas, labels


def advanced_preprocessing(train_data, test_data, n_components=10):
    """
    高级数据预处理，包括：
      1) 删除无效列 (整列 NaN 或者 无方差列)
      2) 缺失值填充 (中位数)
      3) 标准化
      4) PCA 降维

    参数:
      - train_data: shape (n_train, n_features)
      - test_data : shape (n_test,  n_features)
      - n_components: PCA 降到多少维 (可自行调参)

    返回:
      - train_data_pca: 预处理后的训练集特征
      - test_data_pca : 预处理后的测试集特征
    """
    # 转成 numpy 数组，并将 None->np.nan
    train_array = np.array(train_data, dtype=object)
    train_array = np.where(train_array == None, np.nan, train_array).astype(float)

    test_array = np.array(test_data, dtype=object)
    test_array = np.where(test_array == None, np.nan, test_array).astype(float)

    # 1) 删除无效列
    #    1.1) 整列都是 NaN
    not_all_nan_mask = ~np.isnan(train_array).all(axis=0)
    train_array = train_array[:, not_all_nan_mask]
    test_array = test_array[:, not_all_nan_mask]

    #    1.2) 整列无方差 (对训练集而言)
    #         注意测试集也要删除对应列，以保持一致
    var_train = np.nanvar(train_array, axis=0)  # 计算每列的方差（忽略 NaN）
    not_zero_var_mask = (var_train != 0)
    train_array = train_array[:, not_zero_var_mask]
    test_array = test_array[:, not_zero_var_mask]

    # 2) 缺失值填充 (中位数)
    imputer = SimpleImputer(strategy='median')
    # 在训练集上 fit
    imputer.fit(train_array)
    train_array = imputer.transform(train_array)
    test_array = imputer.transform(test_array)

    # 3) 标准化
    scaler = StandardScaler()
    # 在训练集上 fit
    scaler.fit(train_array)
    train_array = scaler.transform(train_array)
    test_array = scaler.transform(test_array)

    # 4) PCA 降维
    #    如果维度非常大，可以根据实际数据设置 n_components
    if n_components is not None and n_components > 0 and n_components < train_array.shape[1]:
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(train_array)           # 在训练集上 fit
        train_array = pca.transform(train_array)
        test_array = pca.transform(test_array)

    return train_array, test_array


if __name__ == "__main__":
    # 1. 读取数据
    train_data, train_label = parse_file(train_path)
    test_data, test_label = parse_file(test_path)

    print("训练集样本数:", len(train_data), "；测试集样本数:", len(test_data))
    if len(train_data) > 0:
        print("单条训练数据长度:", len(train_data[0]))
    if len(test_data) > 0:
        print("单条测试数据长度:", len(test_data[0]))

    # 将标签转为 numpy array
    train_label = np.array(train_label, dtype=int)
    test_label = np.array(test_label, dtype=int)

    # 2. 数据预处理：删除无效列、缺失值填充、标准化、PCA
    #    这里可根据需要修改 PCA 的维度
    pca_components = 50  # 可以尝试不同值，如 None, 5, 10, 20, ...
    train_data_processed, test_data_processed = advanced_preprocessing(
        train_data, 
        test_data,
        n_components=pca_components
    )

    # 3. 使用 LightGBM 进行训练
    clf = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.2,
        min_child_samples=10,
        min_split_gain=0.0,
        random_state=42
    )
    clf.fit(train_data_processed, train_label)

    # 4. 在测试集上进行预测
    y_pred = clf.predict(test_data_processed)
    y_prob = clf.predict_proba(test_data_processed)[:, 1]  # 得到预测概率(正类)

    # 5. 计算评价指标
    accuracy = accuracy_score(test_label, y_pred)
    precision = precision_score(test_label, y_pred, zero_division=0)
    recall = recall_score(test_label, y_pred, zero_division=0)
    f1 = f1_score(test_label, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(test_label, y_prob)
    except ValueError:
        roc_auc = float('nan')
    cm = confusion_matrix(test_label, y_pred)

    # 6. 输出结果
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
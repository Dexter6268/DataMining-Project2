import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 输入文件路径
input_path = "data/bank+marketing/bank/bank-full.csv"

# 输出 RRL 格式文件路径
output_data_path = "rrl/dataset/bankMarketing.data"
output_info_path = "rrl/dataset/bankMarketing.info"


def load_and_process_csv(file_path):
    """
    加载 CSV 数据并进行初步处理。
    返回 data (特征) 和 labels (目标变量)。
    """
    # 读取 CSV 文件
    data = pd.read_csv(file_path, sep=";")

    # 提取目标变量
    labels = data['y'].apply(lambda x: 1 if x == "yes" else 0)

    # 编码分类变量
    label_encoders = {}
    for col in data.select_dtypes(include=['object']).columns:
        if col != 'y':  # 排除目标变量
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

    # 返回特征和目标变量
    return data.drop('y', axis=1), labels


def preprocess_data(data):
    """
    数据预处理，包括：
      1) 缺失值填充 (中位数)
      2) 标准化
    """
    # 缺失值填充 (中位数)
    imputer = SimpleImputer(strategy='median')
    data = imputer.fit_transform(data)

    # 标准化
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    return data


def generate_rrl_format(preprocessed_data, labels, output_data, output_info):
    """
    生成 RRL 格式的 *.data 和 *.info 文件。
    """
    # 写入 *.data 文件
    with open(output_data, "w") as f:
        for data, label in zip(preprocessed_data, labels):
            row = ", ".join(map(str, data)) + f", {label}\n"
            f.write(row)

    # 生成 *.info 文件
    with open(output_info, "w") as f:
        for i in range(preprocessed_data.shape[1]):
            f.write(f"feature_{i+1}, continuous\n")
        f.write("class, discrete\n")
        f.write("LABEL_POS, -1\n")


if __name__ == "__main__":
    # 1. 加载并处理数据
    features, labels = load_and_process_csv(input_path)

    # 2. 数据预处理
    preprocessed_features = preprocess_data(features)

    # 3. 转换为 RRL 格式
    generate_rrl_format(
        preprocessed_features, labels,
        output_data=output_data_path,
        output_info=output_info_path
    )

    print(f"RRL 数据集已生成：\n*.data 文件路径: {output_data_path}\n*.info 文件路径: {output_info_path}")
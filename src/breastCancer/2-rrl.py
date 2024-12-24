import os
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 输入文件路径
train_path = "data/breastCancer/breastCancer_Train.dbc"
test_path = "data/breastCancer/breastCancer_Test.dbc"

# 输出 RRL 格式文件路径
output_data_path = "rrl/dataset/breastCancer.data"
output_info_path = "rrl/dataset/breastCancer.info"


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
    """
    高级数据预处理，包括：
      1) 删除全空列
      2) 缺失值填充 (中位数)
      3) 标准化
      4) PCA 降维
    """
    # 转成 numpy 数组，并将 None->np.nan
    train_array = np.array(train_data, dtype=object)
    train_array = np.where(train_array == None, np.nan, train_array).astype(float)

    test_array = np.array(test_data, dtype=object)
    test_array = np.where(test_array == None, np.nan, test_array).astype(float)

    # 1) 删除全空列
    not_all_nan_mask = ~np.isnan(train_array).all(axis=0)
    train_array = train_array[:, not_all_nan_mask]
    test_array = test_array[:, not_all_nan_mask]

    # 2) 缺失值填充 (中位数)
    imputer = SimpleImputer(strategy='median')
    imputer.fit(train_array)  # 在训练集上 fit
    train_array = imputer.transform(train_array)
    test_array = imputer.transform(test_array)

    # 3) 标准化
    scaler = StandardScaler()
    scaler.fit(train_array)  # 在训练集上 fit
    train_array = scaler.transform(train_array)
    test_array = scaler.transform(test_array)

    # 4) PCA 降维
    if n_components is not None and n_components > 0 and n_components < train_array.shape[1]:
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(train_array)  # 在训练集上 fit
        train_array = pca.transform(train_array)
        test_array = pca.transform(test_array)

    return train_array, test_array


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
    # 1. 读取数据
    train_data, train_label = parse_file(train_path)
    test_data, test_label = parse_file(test_path)

    # 2. 数据预处理
    pca_components = 50  # 设置 PCA 的降维数目
    preprocessed_train, preprocessed_test = advanced_preprocessing(
        train_data, test_data, n_components=pca_components
    )

    # 合并训练集和测试集数据
    combined_data = np.vstack((preprocessed_train, preprocessed_test))
    combined_labels = np.hstack((train_label, test_label))

    # 3. 转换为 RRL 格式
    generate_rrl_format(
        combined_data, combined_labels,
        output_data=output_data_path,
        output_info=output_info_path
    )

    print(f"RRL 数据集已生成：\n*.data 文件路径: {output_data_path}\n*.info 文件路径: {output_info_path}")
import numpy as np
from scipy import stats
from joblib import Parallel, delayed

from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error


class CART:
    def __init__(self, max_depth=None, min_samples_split=2, mode="regression"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.mode = mode
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict_row(row, self.tree) for row in X])
    
    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_values = np.unique(y)
        if len(unique_values) == 1 or (self.max_depth and depth == self.max_depth) or num_samples < self.min_samples_split:
            if self.mode == 'regression':
                return np.mean(y)
            else:
                return stats.mode(y, keepdims=False).mode
        
        best_split = None
        best_score = float('inf')

        for feature_idx in range(num_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                left_y = y[left_mask]
                right_y = y[~left_mask]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                if self.mode == 'regression':
                    left_rss = np.sum((left_y - left_y.mean()) ** 2)
                    right_rss = np.sum((right_y - right_y.mean()) ** 2)
                    score = (left_rss + right_rss) / num_samples
                else:
                    left_gini = self._gini(left_y)
                    right_gini = self._gini(right_y)
                    score = (len(left_y) * left_gini + len(right_y) * right_gini) / num_samples

                if score < best_score:
                    best_score = score
                    best_left_mask = left_mask
                    best_split = (feature_idx, threshold)

        if best_split is None:
            if self.mode == 'regression':
                return np.mean(y)
            else:
                return stats.mode(y, keepdims=False).mode

        left_tree = self._build_tree(X[best_left_mask], y[best_left_mask], depth + 1)
        right_tree = self._build_tree(X[~best_left_mask], y[~best_left_mask], depth + 1)
        
        return {
            'best_split': best_split,
            'left': left_tree,
            'right': right_tree
        }

    def _predict_row(self, row, tree):
        if isinstance(tree, dict):
            feature, threshold = tree['best_split']
            if row[feature] <= threshold:
                return self._predict_row(row, tree['left'])
            else:
                return self._predict_row(row, tree['right'])
        else:
            return tree
    
    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1 - np.sum(p ** 2)

class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, max_features='sqrt', oob_score=False, mode="regression", random_state=0, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.oob_score = oob_score
        self.oob_score_ = None  # 用于存储OOB得分
        self.mode = mode
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.estimators = None

    def fit(self, X, y):
        # np.random.seed(self.random_state)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        # # 并行训练每棵树
        # self.estimators = Parallel(n_jobs=self.n_jobs)(
        #     delayed(self._train_tree)(X, y, i)
        #     for i in range(self.n_estimators)
        # )
        self.estimators = [self._train_tree(X, y, i) for i in range(self.n_estimators)]
        
        # 计算袋外得分
        if self.oob_score:
            self._cal_oob_score(X, y)
    
    def _train_tree(self, X, y, i):
        np.random.seed(self.random_state + i)
        num_samples, num_features = X.shape
        
        # boostrap
        sample_idx = np.random.choice(num_samples, size=num_samples, replace=True)
        X_k, y_k = X[sample_idx], y[sample_idx]
        
        if self.max_features == 'sqrt':
            feature_idx = np.random.choice(num_features, size=int(np.sqrt(num_features)), replace=False)
        elif self.max_features == 'log2':
            feature_idx = np.random.choice(num_features, size=int(np.log2(num_features)), replace=False)
        else:
            feature_idx = np.random.choice(num_features, size=num_features, replace=False)
        
        estimator = CART(self.max_depth, self.min_samples_split, self.mode)
        estimator.fit(X_k[:, feature_idx], y_k)
        return (estimator, feature_idx, sample_idx)
    
    def predict(self, X):
        y_pred = []
        for estimator, feature_idx, _ in self.estimators:
            y_pred.append(estimator.predict(X[:, feature_idx]))
        y_pred = np.array(y_pred).T
        self._y_pred = y_pred
        if self.mode == 'classification':
            y_pred = stats.mode(y_pred, axis=1, keepdims=False).mode
        elif self.mode == 'regression':
            y_pred = np.mean(y_pred, axis=1)
        return y_pred
    
    def predict_proba(self, X):
        y_pred = []
        for estimator, feature_idx, _ in self.estimators:
            y_pred.append(estimator.predict(X[:, feature_idx]))
        y_pred = np.array(y_pred).T
        y_prob = np.stack([1 - np.mean(y_pred, axis=1), np.mean(y_pred, axis=1)], axis=1)
        return y_prob
    
    def _cal_oob_score(self, X, y):
        num_samples, num_features = X.shape
        if self.mode == "regression":
            oob_predictions = np.zeros(num_samples)  # 存储袋外样本的预测结果
        else:
            oob_predictions = [[] for _ in range(num_samples)]
        oob_count = np.zeros(num_samples)  # 记录每个样本被多少棵树预测过
        for estimator, feature_idx, sample_idx in self.estimators:
            oob_idx = np.setdiff1d(np.arange(num_samples), sample_idx)
            # 对袋外样本进行预测
            for idx in oob_idx:
                oob_count[idx] += 1
                y_pred_oob = estimator._predict_row(X[idx, feature_idx], estimator.tree)

                if self.mode == "regression":
                    oob_predictions[idx] += y_pred_oob  # 回归任务是累加
                else:
                    oob_predictions[idx].append(y_pred_oob)  # 分类时采用投票
        mask = oob_count > 0
        if self.mode == "regression":
            # 计算R方
            oob_predictions = oob_predictions[mask] / oob_count[mask]  # 对每个样本的袋外预测结果进行平均
            self.oob_score_ = 1 - np.sum((oob_predictions - y[mask]) ** 2) / np.sum((y[mask] - y[mask].mean()) ** 2)
        else:
            # 计算分类准确率
            oob_predictions = np.array([stats.mode(predictions, keepdims=False).mode for predictions in oob_predictions if predictions])
            self.oob_score_ = np.mean(oob_predictions == y[mask])  # 计算准确率

def compare_against_sklearn(random_state=42):
    # 加载Iris数据集
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # 使用自实现的RandomForest进行训练
    rf_custom = RandomForest(n_estimators=100, max_depth=3, mode="classification", random_state=random_state, oob_score=True)
    rf_custom.fit(X_train, y_train)
    y_pred_custom = rf_custom.predict(X_test)

    # 使用sklearn的RandomForestClassifier进行训练
    rf_sklearn = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=random_state, oob_score=True)
    rf_sklearn.fit(X_train, y_train)
    y_pred_sklearn = rf_sklearn.predict(X_test)

    # 对比预测结果
    accuracy_custom = accuracy_score(y_test, y_pred_custom)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)

    print("自实现 RandomForest 分类准确率:", accuracy_custom)
    print('oob score:', rf_custom.oob_score_)
    print("sklearn RandomForest 分类准确率:", accuracy_sklearn)
    print('sklearn oob score:', rf_sklearn.oob_score_)


    # 生成回归数据集
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # 使用自实现的RandomForest进行训练
    rf_custom = RandomForest(n_estimators=100, 
                            max_depth=3, 
                            mode="regression", 
                            random_state=random_state, 
                            max_features='sqrt',
                            oob_score=True)
    rf_custom.fit(X_train, y_train)
    y_pred_custom = rf_custom.predict(X_test)

    # 使用sklearn的RandomForestRegressor进行训练
    rf_sklearn = RandomForestRegressor(n_estimators=100, 
                                    max_depth=3, 
                                    random_state=random_state,
                                    max_features='sqrt',
                                    oob_score=True)
    rf_sklearn.fit(X_train, y_train)
    y_pred_sklearn = rf_sklearn.predict(X_test)

    # 对比预测结果
    mse_custom = mean_squared_error(y_test, y_pred_custom)
    mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)

    print("自实现 RandomForest 回归 MSE:", mse_custom)
    print('自实现 oob score:', rf_custom.oob_score_)
    print("sklearn RandomForest 回归 MSE:", mse_sklearn)
    print('sklearn oob score:', rf_sklearn.oob_score_)

# compare_against_sklearn(random_state=0)

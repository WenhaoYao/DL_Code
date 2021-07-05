import numpy as np

def pca(data: np.ndarray, k):
    # 零均值化，去中心
    data_norm = data - data.mean()
    # 计算协方差矩阵的特征值和特征向量
    eig_values, eig_vectors = np.linalg.eig(np.cov(data_norm))
    index = np.argsort((-eig_values)[:k-1]) # 前K个维度的索引值
    # 将前k个特征向量的转置与数据叉乘构建新的空间
    return np.dot(eig_vectors[:,index].T, data_norm)

matrix = np.array([[-1, -1, 0, 2, 0],[-2, 0, 0, 1, 1]])
print(pca(matrix, 2))
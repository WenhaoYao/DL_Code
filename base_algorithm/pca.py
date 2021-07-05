import numpy as np

def pca(data: np.ndarray, k):
    data_norm = data - data.mean()
    eig_values, eig_vectors = np.linalg.eig(np.cov(data_norm))
    index = np.argsort((-eig_values)[:k-1])
    vector = eig_vectors[:,index]
    return np.dot(vector.T, data_norm)

matrix = np.array([[-1, -1, 0, 2, 0],[-2, 0, 0, 1, 1]])
print(pca(matrix, 2))
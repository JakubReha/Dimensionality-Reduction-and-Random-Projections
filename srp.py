import numpy as np
from DimRed import DimRed
from sklearn.random_projection import SparseRandomProjection as SRP2

class SRP(DimRed):
    def __init__(self, X, k):
        """
        :param d: Original dimensions
        :param k: Reduced space dimensions
        """
        self.k = k
        self.N = X.shape[1]
        self.d = X.shape[0]
        self.X = X
        self.scale = 1#np.sqrt(self.d / self.k)
        # R = np.sqrt(3) * np.random.choice([1, 0, -1], size=self.k * self.d, p=[1 / 6, 2 / 3, 1 / 6])
        # self.R = R.reshape((self.k, self.d))

    def fit(self):
        """
        :param X: data matrix, shape(d, N)
        :return: reduced data matrix
        """
        # self.X_k = self.R @ self.X
        transformer = SRP2(n_components=self.k)
        self.X_k = transformer.fit_transform(self.X.T).T
        return self.X_k





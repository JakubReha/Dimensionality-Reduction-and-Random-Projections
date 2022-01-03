import numpy as np
from DimRed import DimRed

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
        self.scale = np.sqrt(self.d / self.k)
        self.R = np.sqrt(3) * np.random.choice([1, 0, -1], size=((self.d, self.k)), p=[1 / 6, 2 / 3, 1 / 6])
        self.R = (self.R / np.sqrt((self.R ** 2).sum(axis=0))).T

    def fit(self):
        """
        :param X: data matrix, shape(d, N)
        :return: reduced data matrix
        """
        self.X_k = self.R @ self.X
        return self.X_k





import numpy as np
from DimRed import DimRed

class SRP(DimRed):
    def __init__(self, d, k):
        """
        :param d: Original dimensions
        :param k: Reduced space dimensions
        """
        self.k = k
        self.d = d
        R = np.sqrt(3) * np.random.choice([1, 0, -1], size=self.k * self.d, p=[1 / 6, 2 / 3, 1 / 6])
        self.R = R.reshape((self.k, self.d))

    def project(self, X):
        """
        :param X: data matrix, shape(d, N)
        :return: reduced data matrix
        """
        self.N = X.shape[1]
        self.X = X
        self.X_k = self.R @ X
        return self.X_k





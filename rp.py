import numpy as np
from DimRed import DimRed

class RP(DimRed):
    def __init__(self, X, k):
        """
        :param d: Original dimensions
        :param k: Reduced space dimensions
        """
        self.k = k
        self.N = X.shape[1]
        self.d = X.shape[0]
        self.X = X
        self.R = np.random.normal(loc=0, scale=1/self.k, size=((self.k, self.d)))

    def fit(self):
        """
        :param X: data matrix, shape(d, N)
        :return: reduced data matrix
        """
        self.X_k = self.R @ self.X
        return self.X_k





import numpy as np
from DimRed import DimRed
from sklearn.random_projection import GaussianRandomProjection as GRP

class RP(DimRed):
    def __init__(self, X, k):
        """
        :param d: Original dimensions
        :param k: Reduced space dimensions
        """
        self.k = k
        self.N = X.shape[1]
        self.d = X.shape[0]
        self.scale = np.sqrt(self.d / self.k)
        self.X = X
        self.R = np.random.normal(loc=0.0, scale=1, size=((self.d, self.k)))
        self.R = (self.R / np.sqrt((self.R**2).sum(axis=0))).T

    def fit(self):
        """
        :param X: data matrix, shape(d, N)
        :return: reduced data matrix
        """
        self.X_k = self.R @ self.X
        return self.X_k





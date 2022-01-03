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
        self.scale = np.sqrt(self.d / self.k)
        self.X = X
        self.R = np.random.normal(loc=0.0, scale=1.0, size=((self.d, self.k)))
        self.R = (self.R / np.sqrt((self.R**2).sum(axis=0))).T

    def fit(self):
        """
        :param X: data matrix, shape(d, N)
        :return: reduced data matrix
        """
        self.X_k = self.R @ self.X
        return self.X_k

    def distortions_inner_product(self, m):
        """
        for text data
        :param m: number of pair comparisons
        :return: distortions measured by inner product
        """
        ids1 = np.random.choice(range(self.N), size=m)
        ids2 = np.random.choice(list(set(range(self.N))-set(ids1)), size=m)
        emd_prod = self.scale**2 * np.sum(np.multiply(self.X_k.T[ids1], self.X_k.T[ids2]), axis=1)
        prod = np.sum(np.multiply(self.X.T[ids1], self.X.T[ids2]), axis=1)
        return emd_prod - prod



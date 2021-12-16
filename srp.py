import numpy as np
import matplotlib.pyplot as plt


class SRP:
    def __init__(self, d, k):
        self.k = k
        self.d = d
        R = np.sqrt(3) * np.random.choice([1, 0, -1], size=self.k * self.d, p=[1 / 6, 2 / 3, 1 / 6])
        self.R = R.reshape((self.k, self.d))

    def project(self, X):
        self.N = X.shape[1]
        self.X = X
        self.X_k = (self.R @ X).T
        return self.X_k.T


    def distortions(self, m):
        m = 90
        ids1 = np.random.choice(range(self.N), size=m)
        ids2 = np.random.choice(list(set(range(self.N))-set(ids1)), size=m)
        emd_dist = np.sqrt(self.d/self.k) * np.sqrt(np.sum((self.X_k[ids1] - self.X_k[ids2])**2, axis=1))
        dist = np.sqrt(np.sum((self.X[ids1] - self.X[ids2]).power(2), axis=1))
        return emd_dist - dist



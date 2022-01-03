import numpy as np
from DimRed import DimRed
from sklearn.decomposition import TruncatedSVD

class PCA(DimRed):

    def __init__(self, X, components):
        self.N = X.shape[1]

        self.X = X
        self.scale = 1
        self.X_mean = self.X - np.mean(self.X, axis=0)
        self.components = components
        self.svd = TruncatedSVD(n_components=self.components)
        self.svd.fit(X.T)


    def eigen(self):

        cov = np.cov(self.X_mean, rowvar=True)
        # each column represents a variable, with observations in the rows.

        eig_val, eig_vect = np.linalg.eigh(cov)

        # sort in descending order
        sort_index = np.argsort(eig_val)[::-1]

        eig_val = eig_val[sort_index]
        eig_vect = eig_vect[:, sort_index]  # column v[:, i] is eig_vect corresponding to eig_val w[i]

        return eig_val, eig_vect

    def fit(self, k):
        #eig_val, eig_vect = self.eigen()

        self.X_k = self.svd.components_[:k, :] @ self.X
        #eig_vect_subset = eig_vect[:, 0: self.components]

        #self.X_k = np.dot(eig_vect_subset.transpose(), self.X_mean.transpose()).transpose()

        return self.X_k

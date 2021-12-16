import numpy as np

class PCA:

    def __init__(self, X, components):

        self.X = X
        self.X_mean = self.X - np.mean(self.X, axis=0)
        self.components = components

    def eigen(self):

        cov = np.cov(self.X_mean, rowvar=False)
        # each column represents a variable, with observations in the rows.

        eig_val, eig_vect = np.linalg.eigh(cov)

        # sort in descending order
        sort_index = np.argsort(eig_val)[::-1]

        eig_val = eig_val[sort_index]
        eig_vect = eig_vect[:, sort_index]  # column v[:, i] is eig_vect corresponding to eig_val w[i]

        return eig_val, eig_vect

    def fit(self):
        eig_val, eig_vect = self.eigen()
        eig_vect_subset = eig_vect[:, 0: self.components]

        X_emb = np.dot(eig_vect_subset.transpose(), self.X_mean.transpose()).transpose()

        return X_emb

import numpy as np
class DimRed:

    def distortions_euclidean(self, m):
        """
        for image data
        :param m: number of pair comparisons
        :return: distortions measured by euclidean distance
        """
        ids1 = np.random.choice(range(self.N), size=m)
        ids2 = np.random.choice(list(set(range(self.N))-set(ids1)), size=m)
        emd_dist = self.scale * np.sqrt(np.sum((self.X_k.T[ids1] - self.X_k.T[ids2])**2, axis=1))
        dist = np.sqrt(np.sum((self.X.T[ids1] - self.X.T[ids2])**2, axis=1))
        return emd_dist - dist

    def distortions_inner_product(self, m):
        """
        for text data
        :param m: number of pair comparisons
        :return: distortions measured by inner product
        """
        ids1 = np.random.choice(range(self.N), size=m)
        ids2 = np.random.choice(list(set(range(self.N))-set(ids1)), size=m)
        emd_prod = np.sum(np.multiply(self.X_k.T[ids1], self.X_k.T[ids2]), axis=1)
        prod = np.sum(np.multiply(self.X.T[ids1], self.X.T[ids2]), axis=1)
        return emd_prod - prod
import numpy as np

class RP:
    # actually the R matrix is the one used in "very sparse random projection":
    def __init__(self, d, k, N):
        self.d = d
        self.k = k
        self.N = N
        s = np.sqrt(N)
        R = np.sqrt(s) * np.random.choice([1, 0, -1], size = d*k, p=[1/(2*s), 1 - 1/s, 1/(2*s)])
        self.R = R.reshape((k, d))

    def project(self, X):
        self.X_trans = X.T
        self.X_proj_trans = np.matmul(self.R, X).T
        return np.matmul(self.R, X)

    def distortion(self):
        N = self.X_trans.shape(0)
        mynum = min(100, np.sqrt(N))
        idx1 = np.random.choice(range(self.N),size=mynum)
        idx2 = np.random.choice(list(set(range(self.N)) - set(idx1)), size=mynum)
        ori_dist = np.sqrt(np.sum((self.X_trans[idx1] - self.X_trans[idx2])**2))
        embed_dist = np.sqrt(self.d / self.k) * np.sqrt(np.sum((self.X_proj_trans[idx1] - self.X_proj_trans[idx2])**2))
        return ori_dist - embed_dist




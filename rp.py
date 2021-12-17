import numpy as np
from DimRed import DimRed

class RP(DimRed):
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





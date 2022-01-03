import numpy as np
from DimRed import DimRed

class DCT(DimRed):
    def __init__(self, X):
        self.X = X
        self.N = X.shape[1]
        self.d = X.shape[0]
        self.scale = 1
        self.closest_power = 1


    def fft(self, samples):
        N = len(samples)
        if N == 1:
            return samples
        else:
            even_samples = self.fft(samples[::2])
            odd_samples = self.fft(samples[1::2])
            multiplicative_term = np.exp(-2j * np.pi * np.arange(N) / N)
            X = np.concatenate([even_samples + multiplicative_term[:int(N / 2)] * odd_samples,even_samples + multiplicative_term[int(N / 2):] * odd_samples])
            return X

    def dctf(self, samples):
        N = self.d
        x = np.zeros(N * 2)
        x[:N] = samples
        x[N:] = samples[::-1]
        dct = np.fft.fft(x)[:N]
        dct[0] = dct[0] * 1/(2*np.sqrt(N))
        i = np.arange(1, len(dct))
        dct[1:] = dct[1:] * 1/(np.sqrt(2*N))*np.exp(-1j * np.pi * i / (2 * N))
        return dct.real

    def fit(self, k, m):
        self.ids1 = np.random.choice(range(self.N), size=m)
        self.ids2 = np.random.choice(list(set(range(self.N)) - set(self.ids1)), size=m)
        self.y = np.empty_like(self.X, dtype=float)

        for i in self.ids1:
            self.y[:, i] = self.dctf(self.X[:, i])
        for i in self.ids2:
            self.y[:, i] = self.dctf(self.X[:, i])
        self.k = k
        self.X_k = self.y[:self.k]
        return self.X_k

    def distortions_euclidean(self):
        """
        for image data
        :param m: number of pair comparisons
        :return: distortions measured by euclidean distance
        """
        emd_dist = self.scale * np.sqrt(np.sum((self.X_k.T[self.ids1] - self.X_k.T[self.ids2])**2, axis=1))
        dist = np.sqrt(np.sum((self.X.T[self.ids1] - self.X.T[self.ids2])**2, axis=1))
        return emd_dist - dist


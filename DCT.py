import numpy as np
from DimRed import DimRed

class DCT(DimRed):
    def __init__(self, X, k):
        self.X = X
        self.d = np.shape(X)[0]
        self.k = k
        self.closest_power = 1
        while self.closest_power < self.d:
            self.closest_power *= 2


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

    def dct(self, samples, type):
        dct = []
        N = self.d
        if type == "numpy":
            x = np.zeros(N * 2)
            x[:N] = samples
            x[N:] = samples[::-1]
            dct = np.fft.fft(x)[:N]
        elif type == "classic":
            N = self.closest_power
            N_zeros = N - self.d
            samples = np.pad(samples, (0, N_zeros), 'constant')
            x = np.zeros(N*2)
            x[:N] = samples
            x[N:] = samples[::-1]
            dct = self.fft(x)[:N]
        else:
            raise Exception("specify the the type of fft")

        for i, f in enumerate(dct):
            if i == 0:
                dct[i] *= 1/(2*np.sqrt(N))*np.exp(-1j * np.pi * i / (2 * N))
            else:
                dct[i] *= 1/(np.sqrt(2*N))*np.exp(-1j * np.pi * i / (2 * N))
            i += 1

        return dct.real

    def fit(self, type="numpy"):
        self.X_k = np.empty_like(self.X, dtype=float)
        for i in range(len(self.X.T)):
            self.X_k[:, i] = self.dct(self.X_k[:, i], type)
        return self.X_k[:self.k]

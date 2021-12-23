import numpy as np
from scipy.io import wavfile
import glob
import os
from DCT import DCT
from scipy.fftpack import dct, idct
from scipy.fft import fft, ifft

files = glob.glob('data/audio/**/*.wav')
for file in files:
    if "_new.wav" in file:
        os.remove(file)
        continue
    #rate, data = wavfile.read(file)
    #data_f = fft(data, norm='ortho')
    #data_f[7500:] = 0
    #data_inv = ifft(data_f, norm='ortho')

    #wavfile.write(file.split(".")[0]+"_new.wav", rate, data_inv.real)
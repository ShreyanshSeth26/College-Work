import numpy as np
from scipy.fft import fft, ifft

h = np.array([1,-2,3,-2,1])
M = len(h)

K = 40
n = np.arange(K)
x = np.zeros(K)
x[:K] = ((-1) ** n[:K]) / (3 ** n[:K])

N = 8  # Length of each DFT block (8-point DFT)
L = N - M + 1  # Length of each segment to be convolved with h

y = np.zeros(K + M - 1)
h= np.concatenate((h, np.zeros(L-1)))
H = fft(np.array(h), N)

for i in range(0,K,L):
    x_segment = x[i : i + L]
    x_padded= np.concatenate((x_segment, np.zeros(M-1)))
    X_block = fft(x_padded, N)
    Y_block = X_block * H
    y_block = np.real(ifft(Y_block))
    y[i : i + N] += y_block

for i, value in enumerate(y):
    print(f"y[{i:02}] = {value:+.10e}")
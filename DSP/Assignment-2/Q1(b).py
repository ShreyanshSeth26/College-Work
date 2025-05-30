import numpy as np
from scipy.fft import fft, ifft

h = np.array([1, -1, -1, 1])
M = len(h)

K = 52
n = np.arange(K)
x = np.zeros(K)
x[:K] = np.array(3 / (2 ** n[:K]))

N = 16# Length of each DFT block (16-point DFT)
L = N - M + 1  # Length of each segment to be convolved with h

y = []
h= np.concatenate((h, np.zeros(L-1)))
H = fft(np.array(h), N)

for i in range(0,(K+M-1),L):
    x_segment = x[i : i + L]
    if i == 0:
        x_padded = np.concatenate((np.zeros(M - 1), x_segment))
    else:
        x_padded = np.array(x[i - (M - 1):i + L])
    X_block = fft(x_padded, N)
    Y_block = X_block * H
    y_block = np.real(ifft(Y_block, N))
    y = np.concatenate((y,y_block[M - 1:]))
    
y= y[0:K+M-1]
for i, value in enumerate(y):
    print(f"y[{i:02}] = {value:+.10e}")
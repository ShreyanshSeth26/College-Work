import numpy as np
import scipy as sc
import math
import cmath
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = 'false'

a = [1, -5, 8, -4, 0]
b = [1, 1, 1, 0, -4]

zeroes, poles, k = sc.signal.tf2zpk(b, a)

plt.scatter(np.real(poles), np.imag(poles), c='r', marker ='x', label='Poles')
plt.scatter(np.real(zeroes), np.imag(zeroes), c='b', marker = 'o', label='Zeros')

unitcircle = plt.Circle((0,0), radius=1, ls='--', ec='k', fc='none')
plt.grid(linestyle='dashed')
plt.gca().add_patch(unitcircle)
plt.gca().set_aspect(1)
plt.axis([-5,5,-5,5])
plt.legend()

poles_str = ', '.join([f"({pole.real:+.3f}{pole.imag:+.3f}j)" for pole in poles])
zeros_str = ', '.join([f"({zero.real:+.3f}{zero.imag:+.3f}j)" for zero in zeroes])
print(f"Poles: {poles_str}")
print(f"Zeros: {zeros_str}")

plt.title(r'Poles and zeros of $\frac{z^{4}+z^{3}+z^{2}-4}{z^{4}+5z^{3}+8z^{2}-4z}$')
plt.show()
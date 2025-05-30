import math
import matplotlib.pyplot as plt
import numpy as np

signal_x = [1 if -10 <= idx <= 10 else 0 for idx in range(-10, 11)]   # Define the input signal x[n] where the signal is 1 within the range -10 to 10, and 0 otherwise
response_h = [math.exp(-0.1 * idx) for idx in range(0,41)]    # Define the impulse response h[n] with an exponentially decaying formula
length_x, length_h = len(signal_x), len(response_h)   # Determine the lengths of the input signal and the impulse response
conv_length = length_x + length_h - 1   # The length of the convolution result will be the sum of the two lengths minus 1
conv_result_manual = [0] * conv_length  # Initialize the convolution result list with zeros

for n in range(conv_length):    # Perform the manual convolution using two nested loops
    for k in range(length_h):
        if 0 <= n - k < length_x:
            conv_result_manual[n] += response_h[k] * signal_x[n - k]
conv_result_numpy = np.convolve(signal_x, response_h)   # Perform convolution using numpy for verificatio
x_range = list(range(-10, conv_length - 10))    # Define the x-axis range (shifted by -10)

# Output the convolution results to ensure both methods give similar results and Plot the results of manual convolution vs numpy convolution
print(f"Manual Convolution: {[f'{val}' for val in conv_result_manual]}\n\nNumpy Convolution: {[f'{val}' for val in conv_result_numpy]}\n\nConvolution Length: {len(conv_result_manual)}")
plt.figure(figsize=(10, 6))
plt.plot(x_range, conv_result_manual, label='Manual Convolution', marker='o')
plt.plot(x_range, conv_result_numpy, label='Numpy Convolution', linestyle='--', marker='x')
plt.title('Comparison of Manual Convolution and Numpy Convolution')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
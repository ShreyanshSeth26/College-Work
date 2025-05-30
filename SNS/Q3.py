import numpy as np
import matplotlib.pyplot as plt

signal_length = 64   # Number of data points in the signal
sample_indices = np.arange(signal_length)   # Create an array of sample indices
signal = np.cos(2 * np.pi * 5 * sample_indices / signal_length) + \
                0.5 * np.cos(2 * np.pi * 12 * sample_indices / signal_length)   # Generating the signal

signal_fft = np.fft.fft(signal)     # Compute the FFT of the signal
fft_magnitude = np.abs(signal_fft)  # Get the magnitude of the FFT
fft_frequencies = np.fft.fftfreq(signal_length)/signal_length     # Compute the corresponding frequencies for the FFT components

# Print the FFT results with corresponding frequencies and magnitudes
print(f"FFT of the signal (signal_fft):\n{signal_fft}\n\nFrequencies and Magnitudes:")
for i, magnitude in enumerate(fft_magnitude[:signal_length]):
    print(f"Frequency: {fft_frequencies[i] * signal_length:.2f} Hz, Magnitude: {magnitude:.2f}")

# Plotting the magnitude spectrum of the signal
plt.figure(figsize=(8, 6))  
plt.stem(fft_frequencies[:signal_length] * signal_length, fft_magnitude[:signal_length])
plt.title('Magnitude Spectrum of the Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

pulse_width = 0.1   # Gaussian function's width
time_vals = np.linspace(-1, 1, 500)     # Generate time array from -1 to 1 with 500 points
gaussian_pulse = np.exp(-((time_vals**2) / (2 * pulse_width**2)))   # Compute Gaussian pulse in the time domain

transformed_pulse = np.fft.fftshift(np.fft.fft(gaussian_pulse))     # Apply Fourier Transform and shift it to center zero frequency
frequency_vals = np.fft.fftshift(np.fft.fftfreq(len(time_vals), time_vals[1] - time_vals[0]))   # Generate corresponding frequency axis for the FFT result
magnitude_spectrum = np.abs(transformed_pulse)  # Get the magnitude spectrum from the FFT result
print(f"FFT Result of Gaussian Pulse:\n {transformed_pulse}")   # Display FFT result for debugging purposes

# Plot the magnitude spectrum of the FFT
plt.figure(figsize=(10, 6))
plt.plot(frequency_vals, magnitude_spectrum)
plt.title('Magnitude Spectrum of FFT of Gaussian Pulse')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()
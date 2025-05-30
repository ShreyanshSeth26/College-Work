import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz

# Set up parameters for signal generation
sampling_rate_hz = 1000  # Samples per second
signal_duration_secs = 2  # Total signal duration in seconds
time_indices = np.linspace(0, signal_duration_secs, sampling_rate_hz * signal_duration_secs, endpoint=False)

low_freq_signal, high_freq_signal = np.cos(2 * np.pi * 10 * time_indices), 0.5 * np.cos(2 * np.pi * 100 * time_indices)  # 10 and 100 Hz components
combined_signal = low_freq_signal + high_freq_signal# Create a composite signal with two different frequencies
fir_coefficients = [0.1, 0.15, 0.5, 0.15, 0.1] # Design an FIR filter (low-pass filter)

low_pass_cutoff = 50  # Cutoff frequency for low-pass filter
iir_numerator, iir_denominator = butter(2, low_pass_cutoff / (sampling_rate_hz / 2), btype='low')   # Design an IIR filter (Butterworth low-pass filter)

# Plot the original signal
plt.figure()
plt.plot(time_indices, combined_signal)
plt.title("Original Signal with Low and High Frequency Components")
plt.xlabel("Time [seconds]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

fir_freq_response_w, fir_freq_response_h = freqz(fir_coefficients, worN=8000)   # Calculate frequency response for FIR filter
iir_freq_response_w, iir_freq_response_h = freqz(iir_numerator, iir_denominator, worN=8000) # Calculate frequency response IIR filter

# Create full spectrum frequency range (negative and positive frequencies)
fir_freq_response_w_full, fir_freq_response_h_full = np.concatenate((-fir_freq_response_w[::-1], fir_freq_response_w)), np.concatenate((np.abs(fir_freq_response_h[::-1]), np.abs(fir_freq_response_h)))
iir_freq_response_w_full, iir_freq_response_h_full = np.concatenate((-iir_freq_response_w[::-1], iir_freq_response_w)), np.concatenate((np.abs(iir_freq_response_h[::-1]), np.abs(iir_freq_response_h)))

# Plot the frequency responses(Full Spectrum)
plt.figure()
plt.plot(0.5 * sampling_rate_hz * fir_freq_response_w_full / np.pi, fir_freq_response_h_full, 'r', label='FIR Filter Response')
plt.plot(0.5 * sampling_rate_hz * iir_freq_response_w_full / np.pi, iir_freq_response_h_full, 'b', label='IIR Filter Response')
plt.title("FIR and IIR Filter Frequency Responses")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)
plt.show()

fir_filtered_signal = np.convolve(combined_signal, fir_coefficients, mode='same')# Apply the FIR filter to the signal using convolution
iir_filtered_signal = lfilter(iir_numerator, iir_denominator, combined_signal)# Apply the IIR filter to the signal using lfilter

# Plot the original and filtered signals
plt.figure()
plt.plot(time_indices, combined_signal, label='Original Signal')
plt.plot(time_indices, fir_filtered_signal, label='FIR Filtered Signal')
plt.plot(time_indices, iir_filtered_signal, label='IIR Filtered Signal')
plt.title("Comparison of Original and Filtered Signals")
plt.xlabel("Time [seconds]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()

original_signal_fft = np.fft.fft(combined_signal)   # Perform FFT on original signal
fir_signal_fft = np.fft.fft(fir_filtered_signal)    # Perform FFT on FIR filtered signal
iir_signal_fft = np.fft.fft(iir_filtered_signal)    # Perform FFT on IIR filtered signal
fft_frequencies = np.fft.fftfreq(len(original_signal_fft), 1 / sampling_rate_hz)    # Compute the FFT frequencies (both positive and negative frequencies)

# Plot the Fourier Transforms of the original and filtered signals (full spectrum)
plt.figure()
plt.plot(fft_frequencies, np.abs(original_signal_fft), label='Original Signal FFT')
plt.plot(fft_frequencies, np.abs(fir_signal_fft), label='FIR Filtered FFT')
plt.plot(fft_frequencies, np.abs(iir_signal_fft), label='IIR Filtered FFT')
plt.title("Fourier Transforms of Original and Filtered Signals (Full Spectrum)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)
plt.show()
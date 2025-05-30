question = 1
while question in [1, 2, 3, 4, 5]:
    question = int(input("Please enter a Question between 1 and 5: "))  #enter question number to get the answer for that question
    print()
    
    if  question == 1:
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
        plt.stem(x_range, conv_result_manual, linefmt=None, markerfmt='o', label='Manual Convolution')
        plt.stem(x_range, conv_result_numpy, linefmt=None, markerfmt='x', label='Numpy Convolution')
        plt.title('Comparison of Manual Convolution and Numpy Convolution')
        plt.xlabel('n')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()
        print()
        
    elif  question == 2:
        import numpy as np
        import matplotlib.pyplot as plt

        # Function to compute Fourier series coefficients
        def compute_fourier_terms(time_period: float, base_time: int, max_terms: int = 32) -> tuple[np.ndarray, np.ndarray]:
            angular_frequency = 2 * np.pi / time_period
            term_indices = np.arange(0, max_terms + 1)
            # Calculate coefficients based on sine function
            fourier_coeffs = 2 * np.sin(term_indices * angular_frequency * base_time) / (term_indices * angular_frequency * base_time)
            fourier_coeffs[0] = 2 * base_time / time_period
            print(f"C_k For T = {time_period // base_time} * T1:\n{fourier_coeffs}\n")
            return term_indices, fourier_coeffs

        # Function to plot the Fourier coefficients
        def display_fourier_coeffs(time_period: float, base_time: int, max_terms: int = 32) -> None:
            terms, coeffs = compute_fourier_terms(time_period, base_time, max_terms)
            magnitudes = np.abs(coeffs)  # Magnitudes of Fourier coefficients
            phases = np.angle(coeffs)   # Phases of Fourier coefficients
            plt.figure(figsize=(12, 6))
            # Plot Magnitude
            plt.subplot(1, 2, 1)
            plt.stem(terms, magnitudes, basefmt=" ", linefmt="b-", markerfmt="bo", label="|C_k|")
            plt.xlabel("Harmonic Index (k)")
            plt.ylabel("Magnitude |C_k|")
            plt.title(f"Magnitude of Fourier Coefficients (T = {time_period // base_time} * T1)")
            plt.grid(True)
            plt.legend()
            # Plot Phase
            plt.subplot(1, 2, 2)
            plt.stem(terms, phases, basefmt=" ", linefmt="r-", markerfmt="ro", label="∠C_k")
            plt.xlabel("Harmonic Index (k)")
            plt.ylabel("Phase ∠C_k (radians)")
            plt.title(f"Phase of Fourier Coefficients (T = {time_period // base_time} * T1)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        base_time = 2   # Base time period
        time_periods = [4 * base_time, 8 * base_time, 16 * base_time]   # List of different time periods to analyze
        for period in time_periods:   # Generate plots for each time period
            display_fourier_coeffs(period, base_time, max_terms=32)
        print()
            
    elif  question == 3:
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
        print()
        
    elif  question == 4:
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
        print()
        
    elif  question == 5:
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
        print()
        
    else:
        print("Thank You")
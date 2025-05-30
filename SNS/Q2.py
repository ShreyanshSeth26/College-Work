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
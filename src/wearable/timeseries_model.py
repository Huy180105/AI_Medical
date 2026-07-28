import numpy as np
from typing import Any


class WearableTimeSeriesModel:
    """
    Time-Series Analysis Engine that extracts spectral features (via FFT) from raw ECG,
    and runs predictive autoregressive forecasts for heart rate trends.
    """

    @staticmethod
    def compute_ecg_frequency_features(ecg_samples: list[float], sampling_rate: float = 100.0) -> dict[str, float]:
        """
        Applies FFT to raw ECG samples to compute dominant frequency peaks and spectral power entropy.
        """
        if len(ecg_samples) < 8:
            return {"dominant_frequency": 0.0, "spectral_power_entropy": 0.0}

        # Convert to numpy array and apply FFT
        y = np.array(ecg_samples)
        n = len(y)
        
        # Fast Fourier Transform
        fft_values = np.fft.fft(y)
        power_spectrum = np.abs(fft_values) ** 2
        
        # Half of spectrum due to symmetry
        half_n = n // 2
        pos_power = power_spectrum[1:half_n]  # exclude DC offset
        
        if len(pos_power) == 0 or np.sum(pos_power) == 0:
            return {"dominant_frequency": 0.0, "spectral_power_entropy": 0.0}
            
        # Dominant frequency peak
        max_idx = int(np.argmax(pos_power)) + 1
        freqs = np.fft.fftfreq(n, d=1.0/sampling_rate)
        dominant_freq = abs(freqs[max_idx])
        
        # Spectral Entropy
        probs = pos_power / np.sum(pos_power)
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        
        return {
            "dominant_frequency": float(round(dominant_freq, 2)),
            "spectral_power_entropy": float(round(entropy, 2))
        }

    @staticmethod
    def forecast_heart_rate(hr_history: list[float], seconds_ahead: int = 10) -> list[float]:
        """
        Forecasts future heart rate readings using an Autoregressive (AR-1)
        and Moving Average (MA) trend extrapolation model.
        """
        if not hr_history:
            return []
            
        if len(hr_history) < 3:
            return [hr_history[-1]] * seconds_ahead

        # Extrapolate using AR-1 coefficient
        mean_hr = sum(hr_history) / len(hr_history)
        last_hr = hr_history[-1]
        
        # Compute dynamic AR factor based on recent delta
        recent_delta = hr_history[-1] - hr_history[-2]
        ar_coeff = 0.82
        
        forecast = []
        current = last_hr
        
        for i in range(seconds_ahead):
            # Future point converges gradually to historical mean, adjusted by momentum
            momentum = recent_delta * (0.5 ** i)
            next_val = ar_coeff * current + (1.0 - ar_coeff) * mean_hr + momentum
            forecast.append(float(round(next_val, 1)))
            current = next_val
            
        return forecast

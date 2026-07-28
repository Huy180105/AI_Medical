from collections import deque
import math
from typing import Any
from src.wearable.patient_state import HealthPacket


class WearableFeatureEngineer:
    """
    Maintains a rolling buffer of physiological packets and performs
    feature extraction including time-domain statistics, HRV estimation, and trend analysis.
    """

    def __init__(self, window_size: int = 60) -> None:
        self.window_size = window_size
        self.buffer: deque[HealthPacket] = deque(maxlen=window_size)

    def add_packet(self, packet: HealthPacket) -> None:
        """Appends a new packet to the sliding window buffer."""
        self.buffer.append(packet)

    def extract_features(self) -> dict[str, Any]:
        """
        Extracts derived statistical and physiological features from the current window.
        """
        if not self.buffer:
            return {}

        hr_vals = [p.heart_rate for p in self.buffer]
        spo2_vals = [p.spo2 for p in self.buffer]
        temp_vals = [p.skin_temp for p in self.buffer]
        stress_vals = [p.stress for p in self.buffer]

        features = {
            "hr_mean": self._mean(hr_vals),
            "hr_std": self._std(hr_vals),
            "hr_max": max(hr_vals),
            "hr_min": min(hr_vals),
            
            "spo2_mean": self._mean(spo2_vals),
            "spo2_min": min(spo2_vals),
            
            "temp_mean": self._mean(temp_vals),
            "temp_trend": self._detect_trend(temp_vals, threshold=0.15),  # 0.15 deg C threshold
            
            "stress_mean": self._mean(stress_vals),
            "stress_max": max(stress_vals),
            
            "hrv_rmssd_ms": self._calculate_hrv_rmssd(hr_vals),
            "hr_trend": self._detect_trend(hr_vals, threshold=5.0),  # 5 bpm threshold
            "spo2_trend": self._detect_trend(spo2_vals, threshold=1.0),  # 1% threshold
        }
        
        return features

    def _mean(self, values: list[float]) -> float:
        if not values:
            return 0.0
        return round(sum(values) / len(values), 2)

    def _std(self, values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return round(math.sqrt(variance), 2)

    def _calculate_hrv_rmssd(self, hr_values: list[float]) -> float:
        """
        Estimates HRV RMSSD (Root Mean Square of Successive Differences) in milliseconds
        from a list of heart rate values by converting them to equivalent RR intervals.
        """
        if len(hr_values) < 2:
            return 0.0

        # Convert HR (bpm) to RR intervals (ms): RR = 60000 / HR
        rr_intervals = [60000.0 / hr for hr in hr_values]
        
        # Calculate successive differences
        diffs_sq = []
        for i in range(len(rr_intervals) - 1):
            diff = rr_intervals[i+1] - rr_intervals[i]
            diffs_sq.append(diff ** 2)

        mean_diff_sq = sum(diffs_sq) / len(diffs_sq)
        return round(math.sqrt(mean_diff_sq), 2)

    def _detect_trend(self, values: list[float], threshold: float) -> str:
        """
        Detects trend direction (Rising, Falling, Stable) using the difference
        between the averages of the first and second halves of the window.
        """
        if len(values) < 4:
            return "Stable"
            
        half = len(values) // 2
        first_half_avg = sum(values[:half]) / half
        second_half_avg = sum(values[half:]) / (len(values) - half)
        diff = second_half_avg - first_half_avg

        if diff > threshold:
            return "Rising"
        elif diff < -threshold:
            return "Falling"
        return "Stable"

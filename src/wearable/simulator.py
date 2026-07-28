import math
import random
import time
from typing import Any
from src.wearable.patient_state import PatientState, HealthPacket


class SamsungGalaxyWatchSimulator:
    """
    Simulates real-time physiological signal streaming from a Samsung Galaxy Watch.
    Generates realistic heart rate, SpO2, skin temperature, stress, steps,
    and high-frequency ECG waveforms based on configurable patient states.
    """

    def __init__(self) -> None:
        self.step_count = 5000
        self.calories = 150.0
        self.ecg_phase = 0.0
        self.last_timestamp = time.time()

    def generate_packet(self, state: PatientState) -> HealthPacket:
        """
        Generates a single HealthPacket containing physiological metrics
        and 100 Hz ECG samples representing the last 1-second window.
        """
        now = time.time()
        dt = now - self.last_timestamp
        self.last_timestamp = now

        # Base values per state
        if state == PatientState.NORMAL:
            base_hr = 70.0
            base_spo2 = 98.0
            base_temp = 36.6
            base_stress = 20.0
            sleep_stage = "Wake"
            activity = "Resting"
            step_inc = random.randint(0, 2)
            cal_inc = 0.01 * step_inc + 0.02
            
        elif state == PatientState.EXERCISE:
            base_hr = 135.0
            base_spo2 = 96.5
            base_temp = 37.5
            base_stress = 45.0
            sleep_stage = "Wake"
            activity = "Running"
            step_inc = random.randint(120, 160)  # fast steps
            cal_inc = 0.05 * step_inc + 0.15

        elif state == PatientState.SLEEP:
            base_hr = 54.0
            base_spo2 = 97.5
            base_temp = 36.1
            base_stress = 8.0
            # Cycle sleep stages based on time
            stages = ["Wake", "Light", "Deep", "REM"]
            sleep_stage = stages[int(now / 30) % 4]
            activity = "Sleeping"
            step_inc = 0
            cal_inc = 0.01

        elif state == PatientState.STRESS:
            base_hr = 98.0
            base_spo2 = 97.0
            base_temp = 36.8
            base_stress = 82.0
            sleep_stage = "Wake"
            activity = "Working"
            step_inc = random.randint(1, 5)
            cal_inc = 0.03

        elif state == PatientState.ARRHYTHMIA:
            base_hr = 110.0  # Eratic base
            base_spo2 = 95.5
            base_temp = 36.6
            base_stress = 65.0
            sleep_stage = "Wake"
            activity = "Resting"
            step_inc = 0
            cal_inc = 0.02

        elif state == PatientState.HYPOXIA:
            base_hr = 104.0  # Reflex tachycardia
            base_spo2 = 85.0   # Hypoxia (low SpO2)
            base_temp = 36.6
            base_stress = 72.0
            sleep_stage = "Wake"
            activity = "Restless"
            step_inc = random.randint(2, 8)
            cal_inc = 0.04

        elif state == PatientState.FEVER:
            base_hr = 105.0  # Elevated due to fever
            base_spo2 = 96.0
            base_temp = 39.2  # High fever
            base_stress = 55.0
            sleep_stage = "Wake"
            activity = "Resting"
            step_inc = 0
            cal_inc = 0.03

        elif state == PatientState.RECOVERY:
            base_hr = 85.0
            base_spo2 = 97.8
            base_temp = 36.9
            base_stress = 28.0
            sleep_stage = "Wake"
            activity = "Walking"
            step_inc = random.randint(30, 50)
            cal_inc = 0.02 * step_inc + 0.05
            
        else:
            base_hr = 70.0
            base_spo2 = 98.0
            base_temp = 36.6
            base_stress = 20.0
            sleep_stage = "Wake"
            activity = "Resting"
            step_inc = 0
            cal_inc = 0.02

        # Add noise/fluctuations
        heart_rate = round(base_hr + random.normalvariate(0.0, 1.5), 1)
        spo2 = round(max(50.0, min(100.0, base_spo2 + random.normalvariate(0.0, 0.5))), 1)
        skin_temp = round(base_temp + random.normalvariate(0.0, 0.1), 2)
        stress = round(max(0.0, min(100.0, base_stress + random.normalvariate(0.0, 2.0))), 1)

        # Update steps & calories
        self.step_count += step_inc
        self.calories = round(self.calories + cal_inc, 2)

        # Generate 1-second ECG signal buffer at 100 Hz (100 samples)
        ecg_samples = []
        sampling_rate = 100.0  # Hz
        
        # Calculate RR interval in seconds
        rr_interval = 60.0 / heart_rate

        for _ in range(100):
            # Advance phase
            # For Arrhythmia, add random rhythm perturbations (R-R variability)
            if state == PatientState.ARRHYTHMIA:
                rhythm_error = random.normalvariate(0.0, 0.05)
                phase_step = (2 * math.pi / (rr_interval + rhythm_error)) / sampling_rate
            else:
                phase_step = (2 * math.pi / rr_interval) / sampling_rate

            self.ecg_phase = (self.ecg_phase + phase_step) % (2 * math.pi)
            point = self._generate_ecg_point(self.ecg_phase)
            ecg_samples.append(point)

        return HealthPacket(
            heart_rate=heart_rate,
            spo2=spo2,
            ecg=ecg_samples,
            skin_temp=skin_temp,
            stress=stress,
            sleep_stage=sleep_stage,
            step_count=self.step_count,
            calories=self.calories,
            activity=activity,
            timestamp=now,
        )

    def _generate_ecg_point(self, phase: float) -> float:
        """
        Synthesizes a single point of a typical cardiac cycle heartbeat using Gaussian curves.
        Waves: P, Q, R, S, T.
        """
        # (Amplitude, peak_center_phase, width_sigma)
        waves = {
            "P": (0.08, 0.15 * 2 * math.pi, 0.04),
            "Q": (-0.12, 0.38 * 2 * math.pi, 0.015),
            "R": (1.00, 0.40 * 2 * math.pi, 0.01),
            "S": (-0.22, 0.42 * 2 * math.pi, 0.015),
            "T": (0.28, 0.65 * 2 * math.pi, 0.07),
        }

        val = 0.0
        for amp, center, width in waves.values():
            val += amp * math.exp(-((phase - center) ** 2) / (2 * (width ** 2)))

        # Baseline wander (low frequency noise)
        val += 0.02 * math.sin(phase / 2)
        # High frequency thermal noise
        val += random.normalvariate(0.0, 0.015)
        
        return round(val, 4)

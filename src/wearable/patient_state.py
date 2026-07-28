from dataclasses import dataclass, field
from enum import Enum


class PatientState(str, Enum):
    NORMAL = "Normal"
    EXERCISE = "Exercise"
    SLEEP = "Sleep"
    STRESS = "Stress"
    ARRHYTHMIA = "Arrhythmia"
    HYPOXIA = "Hypoxia"
    FEVER = "Fever"
    RECOVERY = "Recovery"


@dataclass
class HealthPacket:
    """Represents a single physiological state reading from the wearable simulator."""
    heart_rate: float
    spo2: float
    ecg: list[float]
    skin_temp: float
    stress: float
    sleep_stage: str  # Wake, Light, Deep, REM
    step_count: int
    calories: float
    activity: str
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "heart_rate": self.heart_rate,
            "spo2": self.spo2,
            "ecg": self.ecg,
            "skin_temp": self.skin_temp,
            "stress": self.stress,
            "sleep_stage": self.sleep_stage,
            "step_count": self.step_count,
            "calories": self.calories,
            "activity": self.activity,
            "timestamp": self.timestamp,
        }

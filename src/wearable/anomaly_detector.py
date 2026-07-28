from typing import Any
from src.wearable.patient_state import PatientState, HealthPacket


class WearableAnomalyDetector:
    """
    Checks physiological thresholds and signal features to detect clinical anomalies.
    """

    @staticmethod
    def detect_anomalies(packet: HealthPacket, features: dict[str, Any], state: PatientState) -> list[dict[str, Any]]:
        """
        Runs rules to identify clinical anomalies like Hypoxia, Tachycardia, AFib, and Fevers.
        """
        anomalies = []
        timestamp = packet.timestamp

        # 1. Hypoxia Check
        if packet.spo2 < 90.0:
            severity = "Emergency" if packet.spo2 < 85.0 else "High"
            anomalies.append({
                "type": "Hypoxia",
                "severity": severity,
                "description": f"Dangerously low blood oxygen levels: {packet.spo2}%. Ref: >= 95%.",
                "timestamp": timestamp
            })

        # 2. Heart Rate Checks
        # Resting tachycardia
        if packet.heart_rate > 100.0 and state != PatientState.EXERCISE:
            severity = "High" if packet.heart_rate > 120.0 else "Medium"
            anomalies.append({
                "type": "Tachycardia",
                "severity": severity,
                "description": f"Elevated resting heart rate detected: {packet.heart_rate} bpm.",
                "timestamp": timestamp
            })
            
        # Resting bradycardia
        elif packet.heart_rate < 50.0 and state != PatientState.SLEEP:
            severity = "High" if packet.heart_rate < 42.0 else "Medium"
            anomalies.append({
                "type": "Bradycardia",
                "severity": severity,
                "description": f"Abnormally low resting heart rate: {packet.heart_rate} bpm.",
                "timestamp": timestamp
            })

        # 3. Fever Check
        if packet.skin_temp > 38.0:
            severity = "High" if packet.skin_temp > 39.0 else "Medium"
            anomalies.append({
                "type": "Fever Spike",
                "severity": severity,
                "description": f"High skin temperature: {packet.skin_temp}°C. Ref: <= 37.5°C.",
                "timestamp": timestamp
            })

        # 4. Irregular Rhythm (ECG / HRV) Checks
        # High HRV RMSSD or explicit Arrhythmia state suggests AFib / Arrhythmia
        hrv_rmssd = features.get("hrv_rmssd_ms", 0.0)
        if state == PatientState.ARRHYTHMIA or hrv_rmssd > 180.0:
            anomalies.append({
                "type": "Arrhythmia (AFib)",
                "severity": "High",
                "description": f"Irregular heartbeat interval spacing detected. Estimated HRV RMSSD: {hrv_rmssd}ms.",
                "timestamp": timestamp
            })

        # 5. Stress Check
        if packet.stress > 80.0:
            anomalies.append({
                "type": "Stress Spike",
                "severity": "Medium",
                "description": f"High stress index: {packet.stress}/100.",
                "timestamp": timestamp
            })

        # 6. Sleep Apnea Check
        if state == PatientState.SLEEP and packet.spo2 < 92.0:
            anomalies.append({
                "type": "Sleep Apnea Indicator",
                "severity": "High",
                "description": f"Blood oxygen desaturation ({packet.spo2}%) during sleep stage '{packet.sleep_stage}'.",
                "timestamp": timestamp
            })

        return anomalies

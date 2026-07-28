import asyncio
import time
from typing import Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from src.wearable.patient_state import PatientState, HealthPacket
from src.wearable.simulator import SamsungGalaxyWatchSimulator
from src.wearable.feature_engineering import WearableFeatureEngineer
from src.wearable.timeseries_model import WearableTimeSeriesModel
from src.wearable.anomaly_detector import WearableAnomalyDetector
from src.wearable.risk_predictor import WearableRiskPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WearableStreamManager:
    """
    Manages active WebSocket connections and runs the background loop
    that simulates, feature engineers, and analyzes Galaxy Watch metrics.
    """

    def __init__(self, decision_engine: Optional[Any] = None) -> None:
        self.active_connections: list[WebSocket] = []
        self.simulator = SamsungGalaxyWatchSimulator()
        self.feature_engineer = WearableFeatureEngineer()
        self.risk_predictor = WearableRiskPredictor(decision_engine)
        
        self.current_state = PatientState.NORMAL
        self.sampling_rate = 1.0  # Hz
        
        self.is_running = False
        self._loop_task: Optional[asyncio.Task] = None

    async def connect(self, websocket: WebSocket) -> None:
        """Accepts a WebSocket connection and registers it."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("New wearable WebSocket client connected. Active: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        """Removes a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("Wearable WebSocket client disconnected. Active: %d", len(self.active_connections))

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcasts a JSON message to all connected clients."""
        disconnected_clients = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as exc:
                logger.warning("Error sending message to WebSocket client: %s", exc)
                disconnected_clients.append(connection)

        for client in disconnected_clients:
            self.disconnect(client)

    def start_simulation(self, state: PatientState = PatientState.NORMAL, sampling_rate: float = 1.0) -> bool:
        """
        Starts the background simulation loop task if not already running.
        """
        self.current_state = state
        self.sampling_rate = sampling_rate

        if self.is_running:
            logger.info("Simulator already running. Updated state to '%s'.", state.value)
            return True

        self.is_running = True
        self._loop_task = asyncio.create_task(self._simulation_loop())
        logger.info("Started Wearable Simulator loop task (State: %s, Rate: %.1fHz).", state.value, sampling_rate)
        return True

    def stop_simulation(self) -> bool:
        """
        Stops the background simulation loop task.
        """
        if not self.is_running:
            return False

        self.is_running = False
        if self._loop_task:
            self._loop_task.cancel()
            self._loop_task = None
            
        logger.info("Stopped Wearable Simulator loop task.")
        return True

    async def _simulation_loop(self) -> None:
        """
        Asynchronous background task that ticks periodically, generating,
        processing, and broadcasting physiological data.
        """
        try:
            while self.is_running:
                start_tick = time.perf_counter()
                
                # 1. Generate new packet
                packet = self.simulator.generate_packet(self.current_state)
                
                # 2. Add to rolling buffer
                self.feature_engineer.add_packet(packet)
                
                # 3. Compute features
                features = self.feature_engineer.extract_features()
                
                # 4. Extract time-series model projections
                hr_history = [p.heart_rate for p in self.feature_engineer.buffer]
                forecast = WearableTimeSeriesModel.forecast_heart_rate(hr_history, seconds_ahead=10)
                freq_features = WearableTimeSeriesModel.compute_ecg_frequency_features(packet.ecg)
                
                # Merge advanced features
                features.update(freq_features)
                features["hr_forecast_10s"] = forecast

                # 5. Detect anomalies
                anomalies = WearableAnomalyDetector.detect_anomalies(packet, features, self.current_state)

                # 6. Predict risk and check for clinical interventions
                risk_level = self.risk_predictor.assess_risk(anomalies)
                cdss_alert = self.risk_predictor.trigger_clinical_decision(packet, anomalies, risk_level)

                # 7. Assemble unified payload
                payload = {
                    "patient_state": self.current_state.value,
                    "telemetry": packet.to_dict(),
                    "features": features,
                    "anomalies": anomalies,
                    "risk_level": risk_level,
                    "cdss_alert": cdss_alert,
                    "timestamp": packet.timestamp
                }

                # 8. Broadcast to WebSocket subscribers
                await self.broadcast(payload)

                # Calculate sleep duration to hit target frequency
                elapsed = time.perf_counter() - start_tick
                sleep_time = max(0.01, (1.0 / self.sampling_rate) - elapsed)
                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            logger.info("Simulation loop task cancelled.")
        except Exception as exc:
            logger.error("Error in simulation loop task: %s", exc, exc_info=True)
            self.is_running = False

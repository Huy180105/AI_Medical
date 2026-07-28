from typing import Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Request
from pydantic import BaseModel, Field
from src.wearable.patient_state import PatientState
from src.wearable.stream import WearableStreamManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/wearable", tags=["Wearable AI"])


class StartSimulationRequest(BaseModel):
    state: PatientState = Field(default=PatientState.NORMAL, description="Patient physical state to simulate.")
    sampling_rate: float = Field(default=1.0, ge=0.1, le=10.0, description="Sampling rate in Hz (0.1 to 10.0).")


# We can initialize a global stream manager instance which will be configured on startup
_stream_manager: Optional[WearableStreamManager] = None


def get_stream_manager(app_decision_engine: Optional[Any] = None) -> WearableStreamManager:
    """Singleton getter for WearableStreamManager."""
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = WearableStreamManager(app_decision_engine)
    elif app_decision_engine is not None and _stream_manager.risk_predictor.decision_engine is None:
        _stream_manager.risk_predictor.decision_engine = app_decision_engine
    return _stream_manager


@router.post("/start")
async def start_simulation(request: StartSimulationRequest, fastapi_req: Request) -> dict[str, Any]:
    """
    Starts the real-time physiological simulator with a specified patient state and rate.
    """
    # Try to grab clinical decision engine from app state
    decision_engine = getattr(fastapi_req.app.state, "decision_engine", None)
    manager = get_stream_manager(decision_engine)
    
    success = manager.start_simulation(state=request.state, sampling_rate=request.sampling_rate)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to start simulation.")
        
    return {
        "status": "started",
        "current_state": request.state.value,
        "sampling_rate": request.sampling_rate
    }


@router.post("/stop")
async def stop_simulation() -> dict[str, str]:
    """
    Stops the background simulation loop task.
    """
    manager = get_stream_manager()
    success = manager.stop_simulation()
    if not success:
        return {"status": "inactive", "message": "Simulation loop was not running."}
    return {"status": "stopped"}


@router.get("/status")
async def get_simulation_status() -> dict[str, Any]:
    """
    Returns the running state and parameters of the simulator.
    """
    manager = get_stream_manager()
    return {
        "is_running": manager.is_running,
        "current_state": manager.current_state.value,
        "sampling_rate": manager.sampling_rate,
        "active_connections": len(manager.active_connections)
    }


@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    """
    WebSocket endpoint to subscribe to real-time simulated physiological telemetry.
    """
    manager = get_stream_manager()
    await manager.connect(websocket)
    try:
        # Keep connection open and listen for incoming messages (optional control commands)
        while True:
            data = await websocket.receive_text()
            logger.info("Received message from client: %s", data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as exc:
        logger.warning("Error in websocket connection: %s", exc)
        manager.disconnect(websocket)

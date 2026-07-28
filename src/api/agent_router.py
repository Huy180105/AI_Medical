from typing import Any
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/agent", tags=["Medical AI Agent"])


class AgentQueryRequest(BaseModel):
    query: str = Field(
        ...,
        description="Clinical query or patient medical text.",
        json_schema_extra={"example": "Bệnh nhân bị ho kéo dài 3 tuần và đau ngực, tiền sử hút thuốc lá."}
    )
    user_id: str = Field(default="user_default")


@router.post("/query")
async def process_agent_query(fastapi_req: Request, payload: AgentQueryRequest) -> dict[str, Any]:
    """
    Processes patient clinical query through Medical Agent multi-agent orchestration.
    """
    try:
        agent = getattr(fastapi_req.app.state, "medical_agent", None)
        if not agent:
            raise HTTPException(status_code=503, detail="Medical Agent uninitialized.")

        response = agent.process_query(payload.query)
        return response
    except Exception as exc:
        logger.error("Error during agent processing: %s", exc)
        raise HTTPException(status_code=500, detail=f"Agent query failed: {exc}")

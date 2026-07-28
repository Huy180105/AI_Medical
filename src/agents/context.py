from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AgentContext:
    """
    Shared context representing the state of the patient session and NLP analysis
    as it flows through the multi-agent system.
    """
    text: str
    session_id: Optional[str] = None
    patient_id: Optional[str] = None
    
    # NLP and Knowledge State
    entities: list[dict[str, Any]] = field(default_factory=list)
    normalized_entities: list[dict[str, Any]] = field(default_factory=list)
    knowledge: list[dict[str, Any]] = field(default_factory=list)
    guidelines: list[dict[str, Any]] = field(default_factory=list)
    graph_results: list[dict[str, Any]] = field(default_factory=list)
    
    # Clinical Support and Recommendation State
    clinical_decision: dict[str, Any] = field(default_factory=dict)
    recommendations: dict[str, Any] = field(default_factory=dict)
    response: dict[str, Any] = field(default_factory=dict)
    
    # Orchestration and Execution Metrics
    history: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)  # execution records
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

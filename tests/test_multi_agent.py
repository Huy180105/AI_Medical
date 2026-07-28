import pytest
from unittest.mock import MagicMock
from src.agents.context import AgentContext
from src.agents.executor import AgentExecutor
from src.agents.router_agent import RouterAgent
from src.agents.ner_agent import NERAgent
from src.agents.knowledge_agent import KnowledgeAgent
from src.agents.guideline_agent import GuidelineAgent
from src.agents.graph_agent import GraphAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.clinical_agent import ClinicalAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.response_agent import ResponseAgent
from src.agents.conversation_agent import ConversationAgent
from src.agents.memory_agent import MemoryAgent
from src.agents.base import BaseAgent


# Simple mock services
@pytest.fixture
def mock_predictor():
    p = MagicMock()
    p.predict.return_value = [{"text": "sot", "type": "SYMPTOM", "score": 0.98}]
    return p


@pytest.fixture
def mock_icd10():
    s = MagicMock()
    s.normalize.return_value = {"original": "sot", "type": "SYMPTOM", "code": "R50.9", "name": "Fever"}
    return s


@pytest.fixture
def mock_rxnorm():
    s = MagicMock()
    s.normalize.return_value = {}
    return s


@pytest.fixture
def mock_rag():
    r = MagicMock()
    r.run.return_value = [{"text": "Guideline body", "metadata": {"title": "Fever Guideline"}}]
    return r


@pytest.fixture
def mock_graph_reasoner():
    g = MagicMock()
    g.reason.return_value = [{
        "disease": "Fever unspecified",
        "node_id": "Disease:fever_unspecified",
        "confidence": 0.85,
        "path": ["[Symptom:fever] <- Disease:fever_unspecified"],
        "evidence": {"matched_symptoms": ["fever"], "contraindicated_drugs": []}
    }]
    return g


@pytest.fixture
def mock_decision_engine():
    d = MagicMock()
    d.make_decision.return_value = {
        "diagnosis_candidates": [{"disease": "Fever unspecified", "confidence": 0.85}],
        "risk_level": "Medium",
        "recommendations": {
            "recommended_labs": ["CBC"],
            "recommended_medication_categories": [],
            "referral_suggestion": "Consult GP",
            "lifestyle_advice": []
        },
        "follow_up": {"action": "Revisit", "timeframe": "48 hours"},
        "evidence": {
            "supporting_paths": ["[Symptom:fever] <- Disease:fever_unspecified"],
            "red_flags": [],
            "contraindications": [],
            "guideline_sources": []
        },
        "confidence": 0.85
    }
    return d


@pytest.fixture
def mock_memory_store():
    store = MagicMock()
    store.get_turns.return_value = [{"input": "hi", "output": {"text": "hello"}}]
    return store


class DummyFailureAgent(BaseAgent):
    """An agent designed to fail to test retry and fallback policies."""
    def __init__(self, failure_count=1):
        self.failure_count = failure_count
        self.attempts = 0

    @property
    def name(self) -> str:
        return "dummy_failure"

    def process(self, context: AgentContext) -> AgentContext:
        self.attempts += 1
        if self.attempts <= self.failure_count:
            raise ValueError("Intentional failure")
        context.metadata["dummy"] = "Success"
        return context


class DummyFallbackAgent(BaseAgent):
    """An agent executing as a backup route."""
    @property
    def name(self) -> str:
        return "dummy_fallback"

    def process(self, context: AgentContext) -> AgentContext:
        context.metadata["dummy"] = "Fallback Success"
        return context


def test_router_agent():
    router = RouterAgent()

    # 1. Test conversational greeting routing
    ctx1 = AgentContext(text="Xin chào bác sĩ")
    ctx1 = router.process(ctx1)
    assert ctx1.metadata.get("next_agent") == "conversation_agent"

    # 2. Test clinical query routing
    ctx2 = AgentContext(text="Bị sốt và ho")
    ctx2 = router.process(ctx2)
    assert "next_agent" not in ctx2.metadata


def test_clinical_pipeline_execution(
    mock_predictor, mock_icd10, mock_rxnorm, mock_rag, mock_graph_reasoner, mock_decision_engine
):
    # Instantiate agents
    agents = [
        RouterAgent(),
        NERAgent(mock_predictor),
        KnowledgeAgent(mock_icd10, mock_rxnorm),
        GuidelineAgent(mock_rag),
        GraphAgent(mock_graph_reasoner),
        ReasoningAgent(),
        ClinicalAgent(mock_decision_engine),
        RecommendationAgent(),
        ResponseAgent(),
        ConversationAgent()
    ]

    executor = AgentExecutor(agents)
    context = AgentContext(text="Bị sốt ho nhiều ngày")
    
    pipeline = [
        "router_agent", "ner_agent", "knowledge_agent", "guideline_agent",
        "graph_agent", "reasoning_agent", "clinical_agent", "recommendation_agent", "response_agent"
    ]
    
    context = executor.execute_pipeline(pipeline, context)
    
    # Assert state aggregations
    assert len(context.entities) > 0
    assert len(context.normalized_entities) > 0
    assert len(context.guidelines) > 0
    assert len(context.graph_results) > 0
    assert "reasoning_alignment" in context.metadata
    assert context.clinical_decision["risk_level"] == "Medium"
    assert "CBC" in context.recommendations["recommended_labs"]
    
    # Check trace metrics
    assert len(context.trace) == len(pipeline)
    for t in context.trace:
        assert t["status"].startswith("Success")
        assert t["duration_ms"] >= 0.0


def test_conversational_greeting_execution():
    agents = [
        RouterAgent(),
        ConversationAgent(),
        ResponseAgent()
    ]
    executor = AgentExecutor(agents)
    
    context = AgentContext(text="Xin chào")
    # Pipeline contains clinical routing, but router_agent should redirect to conversation_agent
    pipeline = ["router_agent", "ner_agent", "response_agent"]
    
    context = executor.execute_pipeline(pipeline, context)
    
    assert context.clinical_decision.get("conversational") is True
    assert "MedAgent" in context.response["clinical_decision"]["message"]
    # Check that execution jumped to conversation_agent and response_agent
    trace_agents = [t["agent"] for t in context.trace]
    assert "router_agent" in trace_agents
    assert "conversation_agent" in trace_agents
    assert "response_agent" in trace_agents
    assert "ner_agent" not in trace_agents


def test_executor_retry_mechanism():
    # Setup failure agent failing once (will succeed on 2nd attempt, retry = 1)
    fail_agent = DummyFailureAgent(failure_count=1)
    executor = AgentExecutor([fail_agent])
    
    context = AgentContext(text="test")
    context = executor.execute_agent("dummy_failure", context, retries=1)
    
    assert context.metadata["dummy"] == "Success"
    assert fail_agent.attempts == 2
    assert "Success_after_retry" in context.trace[0]["status"]


def test_executor_fallback_mechanism():
    # Setup failure agent failing twice, retries = 1, fallback will trigger
    fail_agent = DummyFailureAgent(failure_count=2)
    fallback_agent = DummyFallbackAgent()
    
    executor = AgentExecutor([fail_agent, fallback_agent])
    
    context = AgentContext(text="test")
    # Tell the executor to retry once and then use dummy_fallback
    context.metadata["dummy_failure_fallback"] = "dummy_fallback"
    context.metadata["dummy_failure_retries"] = 1
    
    context = executor.execute_pipeline(["dummy_failure"], context)
    
    # Check fallback executed
    assert context.metadata["dummy"] == "Fallback Success"
    assert fail_agent.attempts == 2
    
    # Verify trace shows failure and fallback
    trace_agents = [t["agent"] for t in context.trace]
    assert "dummy_failure" in trace_agents
    assert "dummy_fallback" in trace_agents
    
    # Check statuses
    statuses = [t["status"] for t in context.trace]
    assert "Failed" in statuses
    assert "Success" in statuses

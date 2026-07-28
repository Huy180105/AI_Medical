from src.agents.base import BaseAgent
from src.agents.context import AgentContext
from src.graph.ontology import MedicalOntology


class RouterAgent(BaseAgent):
    """
    Decides the next course of action in the Multi-Agent swarm.
    Classifies queries as clinical or general/conversational and routes accordingly.
    """

    def __init__(self) -> None:
        self.confidence = 1.0

    @property
    def name(self) -> str:
        return "router_agent"

    def process(self, context: AgentContext) -> AgentContext:
        text = context.text.strip().lower()
        
        # Keywords suggesting a conversational greeting/non-clinical intent
        conversational_keywords = {"hello", "hi", "xin chào", "chào bác sĩ", "chào", "tạm biệt", "cảm ơn"}
        
        # Keywords suggesting clinical content (symptoms, drugs, etc.)
        clinical_indicators = {
            "sốt", "sot", "ho", "đau", "dau", "khó thở", "kho tho", "paracetamol",
            "ibuprofen", "amoxicillin", "ors", "tiêu chảy", "nôn", "bệnh", "benh"
        }

        # Check if text matches indicators
        is_clinical = any(indicator in text for indicator in clinical_indicators)
        is_greeting = any(text == kw or text.startswith(kw + " ") for kw in conversational_keywords)

        if is_greeting and not is_clinical:
            # Route to conversation agent directly, skipping clinical extraction
            context.metadata["next_agent"] = "conversation_agent"
        else:
            # Let the default clinical pipeline proceed
            pass
            
        return context

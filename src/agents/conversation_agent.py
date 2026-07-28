from src.agents.base import BaseAgent
from src.agents.context import AgentContext


class ConversationAgent(BaseAgent):
    """
    Handles general conversational interactions, greetings, and simple queries
    that do not require clinical extraction or knowledge graph reasoning.
    """

    def __init__(self) -> None:
        self.confidence = 1.0

    @property
    def name(self) -> str:
        return "conversation_agent"

    def process(self, context: AgentContext) -> AgentContext:
        reply = (
            "Xin chào! Tôi là MedAgent - Trợ lý y khoa thông minh. "
            "Tôi có thể hỗ trợ trích xuất thực thể lâm sàng, tra cứu đồ thị tri thức y học, "
            "và cung cấp hỗ trợ ra quyết định lâm sàng. Hãy mô tả triệu chứng hoặc loại thuốc bạn muốn quan tâm."
        )
        
        # Populate conversational clinical decision
        context.clinical_decision = {
            "possible_diseases": [],
            "recommendations": ["Vui lòng cung cấp chi tiết triệu chứng lâm sàng."],
            "red_flags": [],
            "message": reply,
            "conversational": True
        }
        
        context.recommendations = {
            "recommended_labs": [],
            "recommended_medication_categories": [],
            "referral_suggestion": "Vui lòng thảo luận thêm với chuyên gia y tế.",
            "lifestyle_advice": ["Ăn uống lành mạnh và tập thể dục đều đặn."]
        }
        
        return context

from typing import Any, Optional
from src.agents.base import BaseAgent
from src.agents.context import AgentContext
from src.agent.memory import ConversationMemory


class MemoryAgent(BaseAgent):
    """
    Manages session memory, loading past interaction history at the start of the execution pipeline,
    and recording the final outputs to persistent storage at the end.
    """

    def __init__(self, conversation_memory: Optional[ConversationMemory] = None) -> None:
        self.memory = conversation_memory or ConversationMemory()
        self.confidence = 1.0

    @property
    def name(self) -> str:
        return "memory_agent"

    def process(self, context: AgentContext) -> AgentContext:
        if not context.session_id:
            return context

        # If response is present, this means the pipeline is finishing: Save memory
        if context.response:
            turn = {"input": context.text, "output": context.response}
            self.memory.add_turn(context.session_id, turn)
        else:
            # If no response is present, we are at the beginning: Load history
            history = self.memory.get_turns(context.session_id)
            context.history = history
            
        return context

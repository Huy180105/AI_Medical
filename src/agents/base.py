from abc import ABC, abstractmethod
from src.agents.context import AgentContext


class BaseAgent(ABC):
    """
    Abstract Base Class for all specialized agents in the Medical Multi-Agent System.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The identifier name of the agent."""
        pass

    @abstractmethod
    def process(self, context: AgentContext) -> AgentContext:
        """
        Executes the agent logic, modifying and returning the shared AgentContext.
        """
        pass

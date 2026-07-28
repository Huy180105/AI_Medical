from collections.abc import Callable
from typing import Any


class AgentStepExecutor:
    def execute(self, step_name: str, step: Callable[[], Any]) -> Any:
        return step()

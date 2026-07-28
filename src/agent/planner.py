from dataclasses import dataclass


@dataclass(frozen=True)
class AgentPlan:
    steps: list[str]


class MedicalAgentPlanner:
    def create_plan(self) -> AgentPlan:
        return AgentPlan(
            steps=[
                "ner",
                "normalize",
                "retrieve",
                "reason",
                "respond",
            ]
        )

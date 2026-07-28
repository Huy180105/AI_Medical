from typing import Any

from src.agent.executor import AgentStepExecutor
from src.agent.memory import ConversationMemory, MedicalContextMemory
from src.agent.planner import MedicalAgentPlanner
from src.agent.workflow import MedicalWorkflow, NERPredictor


class MedicalAgent:
    def __init__(
        self,
        ner_predictor: NERPredictor,
        workflow: MedicalWorkflow | None = None,
        planner: MedicalAgentPlanner | None = None,
        executor: AgentStepExecutor | None = None,
        conversation_memory: ConversationMemory | None = None,
        context_memory: MedicalContextMemory | None = None,
    ) -> None:
        self.ner_predictor = ner_predictor
        self.workflow = workflow or MedicalWorkflow(ner_predictor=ner_predictor)
        self.planner = planner or MedicalAgentPlanner()
        self.executor = executor or AgentStepExecutor()
        self.conversation_memory = conversation_memory or ConversationMemory()
        self.context_memory = context_memory or MedicalContextMemory()

    def process(
        self,
        text: str,
        session_id: str | None = None,
        patient_id: str | None = None,
    ) -> dict[str, Any]:
        plan = self.planner.create_plan()
        result = self.executor.execute("workflow", lambda: self.workflow.run(text))
        result["plan"] = plan.steps

        if session_id:
            turn = {"input": text, "output": result}
            self.conversation_memory.add_turn(session_id, turn)
            self.context_memory.save_context(
                session_id,
                {
                    "last_entities": result["entities"],
                    "last_reasoning": result["clinical_reasoning"],
                    "patient_id": patient_id,
                },
            )

        return result

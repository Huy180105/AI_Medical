import time
from typing import Any, Optional
from src.agents.base import BaseAgent
from src.agents.context import AgentContext
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AgentExecutor:
    """
    Orchestrates the execution of a multi-agent system.
    Supports retries, fallbacks, dynamic routing, execution tracing,
    and confidence score propagation.
    """

    def __init__(self, agents: list[BaseAgent]) -> None:
        self.agents = {agent.name: agent for agent in agents}

    def execute_agent(
        self,
        agent_name: str,
        context: AgentContext,
        retries: int = 2,
        fallback_name: Optional[str] = None,
    ) -> AgentContext:
        """
        Executes a single agent, managing retries, fallback routes,
        tracing metrics, and confidence level propagation.
        """
        if agent_name not in self.agents:
            err_msg = f"Agent '{agent_name}' not registered in executor."
            logger.error(err_msg)
            context.trace.append({
                "agent": agent_name,
                "status": "Failed",
                "error": err_msg,
                "duration_ms": 0.0
            })
            return context

        agent = self.agents[agent_name]
        attempt = 0
        success = False
        start_time = time.perf_counter()
        
        while attempt <= retries and not success:
            try:
                logger.info("Executing agent '%s' (Attempt %d/%d)...", agent_name, attempt + 1, retries + 1)
                context = agent.process(context)
                success = True
                duration = (time.perf_counter() - start_time) * 1000.0
                
                # Propagate confidence (multiply to accumulate overall pipeline confidence)
                # Ensure confidence stays bounded between 0.0 and 1.0
                context.confidence = round(max(0.1, min(1.0, context.confidence * getattr(agent, "confidence", 1.0))), 2)

                context.trace.append({
                    "agent": agent_name,
                    "status": "Success" if attempt == 0 else f"Success_after_retry_{attempt}",
                    "duration_ms": round(duration, 2),
                    "error": None
                })
                logger.info("Agent '%s' completed successfully in %.2fms.", agent_name, duration)
                
            except Exception as exc:
                attempt += 1
                logger.warning("Agent '%s' failed on attempt %d: %s", agent_name, attempt, exc)
                if attempt > retries:
                    duration = (time.perf_counter() - start_time) * 1000.0
                    context.trace.append({
                        "agent": agent_name,
                        "status": "Failed",
                        "duration_ms": round(duration, 2),
                        "error": str(exc)
                    })
                    
                    # Execute fallback if available
                    if fallback_name:
                        logger.info("Triggering fallback agent '%s' for failed agent '%s'...", fallback_name, agent_name)
                        return self.execute_agent(fallback_name, context, retries=0, fallback_name=None)
                    
                    raise exc

        return context

    def execute_pipeline(self, pipeline: list[str], context: AgentContext) -> AgentContext:
        """
        Executes a sequence of agents sequentially, respecting dynamic routing modifications
        made by agents in the context metadata.
        """
        execution_list = list(pipeline)
        idx = 0
        
        while idx < len(execution_list):
            agent_name = execution_list[idx]
            
            # Extract optional routing config if registered in metadata
            fallback_agent = context.metadata.get(f"{agent_name}_fallback")
            retries = context.metadata.get(f"{agent_name}_retries", 2)
            
            context = self.execute_agent(
                agent_name,
                context,
                retries=retries,
                fallback_name=fallback_agent
            )
            
            # Check for dynamic routing redirection
            next_routing = context.metadata.pop("next_agent", None)
            if next_routing:
                logger.info("Dynamic router redirected execution flow to: '%s'", next_routing)
                # Replace the immediate next agent in queue with the redirect, preserving subsequent agents
                if idx + 1 < len(execution_list):
                    execution_list[idx + 1] = next_routing
                else:
                    execution_list.append(next_routing)
                
            idx += 1
            
        return context

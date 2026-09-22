"""
SprinterAssistant — Unified Facade for the Multi-Agent System (Phase 10).
Provides a single, intuitive API for running questions, doc generation,
and requirements gathering with persistence and HITL resumption.
"""

import time
from typing import Optional, Dict, Any
from langgraph.types import Command

from src.core.service_registry import ServiceRegistry, services as default_services
from src.supervisor.state import SupervisorState, SupervisorResult
from src.supervisor.graph import compile_supervisor_graph
from src.subagents.srs.graph import invoke_srs_subagent


class SprinterAssistant:
    """
    Unified entry point for the DevDocs AI / Sprinter assistant.
    Dispatches autonomously across QA, DocGen (Phase 1 baseline), and SRS.
    """

    def __init__(
        self,
        services: Optional[ServiceRegistry] = None,
        docgen_version: str = "v1"
    ):
        self.services = services or default_services
        self.docgen_version = docgen_version
        self.graph, self.nodes = compile_supervisor_graph(
            services=self.services,
            docgen_version=self.docgen_version,
            checkpointer=self.services.checkpointer
        )

    def set_docgen_version(self, version: str):
        """Switches the DocGen subgraph implementation (e.g. 'v1' vs 'v2' / 'v3')."""
        self.docgen_version = version
        self.nodes.set_docgen_version(version)
        print(f"✓ [Sprinter] DocGen engine switched to version: '{version}'")

    def run(
        self,
        query: str,
        thread_id: Optional[str] = None,
        repo_url: Optional[str] = None,
        local_path: Optional[str] = None
    ) -> SupervisorResult:
        """
        Executes a user request through the top-level supervisor graph.
        
        Args:
            query: The user prompt or instruction.
            thread_id: Session identifier for conversational memory & checkpointer.
            repo_url: Optional remote GitHub repo URL.
            local_path: Optional local repository directory path.
        """
        active_thread = thread_id or f"thread_{int(time.time())}"
        config = {"configurable": {"thread_id": active_thread}}

        initial_state: SupervisorState = {
            "query": query,
            "conversation_history": [{"role": "user", "content": query}],
            "route": None,
            "repo_url": repo_url,
            "local_path": local_path,
            "final_output": "",
            "thread_id": active_thread,
            "status": "processing",
            "pending_question": None,
            "metadata": {},
            "error": None
        }

        try:
            result_state = self.graph.invoke(initial_state, config=config)
            return SupervisorResult(
                status=result_state.get("status", "completed"),
                route=str(result_state.get("route", "unknown")),
                thread_id=active_thread,
                output=result_state.get("final_output"),
                pending_question=result_state.get("pending_question"),
                metadata=result_state.get("metadata", {}),
                error=result_state.get("error")
            )
        except Exception as exc:
            return SupervisorResult(
                status="failed",
                route="unknown",
                thread_id=active_thread,
                error=str(exc)
            )

    def resume(
        self,
        thread_id: str,
        user_response: str
    ) -> SupervisorResult:
        """
        Resumes an interrupted session (e.g., answering an SRS clarification question).
        """
        # Resume through the SRS child graph checkpointer
        srs_response = invoke_srs_subagent(
            graph=self.nodes.srs_subgraph,
            thread_id=thread_id,
            resume_answer=user_response
        )

        if srs_response.status == "waiting_human_input":
            return SupervisorResult(
                status="waiting_human_input",
                route="srs",
                thread_id=thread_id,
                pending_question=srs_response.pending_question,
                output=f"Clarification Needed:\n{srs_response.pending_question}",
                metadata={"unresolved_gaps": srs_response.unresolved_gaps}
            )
        elif srs_response.status == "completed":
            return SupervisorResult(
                status="completed",
                route="srs",
                thread_id=thread_id,
                output=srs_response.final_document,
                metadata={
                    "diagram_count": len(srs_response.diagram_manifest or []),
                    "unresolved_gaps": srs_response.unresolved_gaps
                }
            )
        elif srs_response.status == "processing":
            return SupervisorResult(
                status="processing",
                route="srs",
                thread_id=thread_id,
                output="Processing requirements...",
                metadata={"unresolved_gaps": srs_response.unresolved_gaps}
            )
        else:
            return SupervisorResult(
                status="failed",
                route="srs",
                thread_id=thread_id,
                error=srs_response.error or "Unknown failure"
            )


# Global default instance
sprinter = SprinterAssistant()

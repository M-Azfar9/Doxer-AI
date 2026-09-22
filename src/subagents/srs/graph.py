"""
Assembly and compilation of the composite SRS Subagent (Feature 3).
Connects the clarification subgraph, generation subgraph, and parent supervisor bridge.
"""

import time
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

from src.core.service_registry import ServiceRegistry, services as default_services
from src.subagents.srs.state import (
    ClarificationState, GenerationState, ParentSRSState,
    SRSSubagentResponse, RequirementsModel
)
from src.subagents.srs.clarification_nodes import ClarificationNodes, route_after_completeness
from src.subagents.srs.generation_nodes import GenerationNodes


def compile_clarification_subgraph(services: Optional[ServiceRegistry] = None):
    """Builds the clarification loop subgraph."""
    nodes = ClarificationNodes(services=services or default_services)
    builder = StateGraph(ClarificationState)

    builder.add_node("analyze_initial_requirements", nodes.analyze_initial_requirements)
    builder.add_node("check_completeness", nodes.check_completeness)
    builder.add_node("ask_question", nodes.ask_question)
    builder.add_node("update_requirements", nodes.update_requirements)

    builder.add_edge(START, "analyze_initial_requirements")
    builder.add_edge("analyze_initial_requirements", "check_completeness")
    builder.add_conditional_edges(
        "check_completeness",
        route_after_completeness,
        {
            "ask_question": "ask_question",
            END: END
        }
    )
    builder.add_edge("ask_question", "update_requirements")
    builder.add_edge("update_requirements", "check_completeness")

    # Compiled without internal checkpointer to allow parent checkpointer inheritance
    return builder.compile()


def compile_generation_subgraph(services: Optional[ServiceRegistry] = None):
    """Builds the document & diagram generation subgraph."""
    nodes = GenerationNodes(services=services or default_services)
    builder = StateGraph(GenerationState)

    builder.add_node("generate_outline", nodes.generate_outline)
    builder.add_node("generate_section", nodes.generate_section)
    builder.add_node("determine_diagrams", nodes.determine_diagrams)
    builder.add_node("generate_diagram", nodes.generate_diagram)
    builder.add_node("assemble_document", nodes.assemble_document)

    builder.add_edge(START, "generate_outline")
    builder.add_edge("generate_outline", "generate_section")
    builder.add_edge("generate_section", "determine_diagrams")
    builder.add_edge("determine_diagrams", "generate_diagram")
    builder.add_edge("generate_diagram", "assemble_document")
    builder.add_edge("assemble_document", END)

    return builder.compile()


def compile_srs_subgraph(
    services: Optional[ServiceRegistry] = None,
    checkpointer: Optional[Any] = None
):
    """
    Compiles the Parent SRS StateGraph.
    Integrates Clarification -> Generation with checkpointer support for HITL pause/resume.
    """
    services = services or default_services
    checkpointer = checkpointer or services.checkpointer or MemorySaver()

    clarification_subgraph = compile_clarification_subgraph(services=services)
    generation_subgraph = compile_generation_subgraph(services=services)

    def run_clarification(state: ParentSRSState) -> Dict[str, Any]:
        """Parent Bridge Node 1: Runs clarification."""
        clarification_input: ClarificationState = {
            "user_prompt": state["user_prompt"],
            "project_context": state.get("project_context"),
            "conversation_log": [],
            "turn_count": 0,
            "max_turns": state.get("max_turns", 5),
            "requirements": state.get("requirements") or RequirementsModel(
                project_title="Software System",
                project_scope=""
            ),
            "completeness": None,
            "unresolved_gaps": []
        }
        try:
            child_result = clarification_subgraph.invoke(clarification_input)
            extracted_requirements = child_result.get("requirements")
            unresolved_gaps = child_result.get("unresolved_gaps", [])
            return {
                "requirements": extracted_requirements,
                "unresolved_gaps": unresolved_gaps,
                "status": "processing",
                "pending_question": None,
                "error": None
            }
        except GraphInterrupt:
            # Re-raise so parent checkpointer transparently halts execution!
            raise
        except Exception as exc:
            return {
                "status": "failed",
                "error": f"Clarification subgraph failure: {str(exc)}"
            }

    def route_parent_transition(state: ParentSRSState) -> str:
        """Conditional Edge: Routes to run_generation or END."""
        if state.get("status") == "failed" or state.get("requirements") is None:
            return END
        return "run_generation"

    def run_generation(state: ParentSRSState) -> Dict[str, Any]:
        """Parent Bridge Node 2: Runs document generation."""
        gen_input: GenerationState = {
            "requirements": state["requirements"],
            "unresolved_gaps": state.get("unresolved_gaps", []),
            "outline": None,
            "section_drafts": [],
            "diagram_specs": [],
            "validated_diagrams": [],
            "final_document": None,
            "diagram_manifest": []
        }
        try:
            gen_output = generation_subgraph.invoke(gen_input)
            return {
                "final_document": gen_output.get("final_document", ""),
                "diagram_manifest": gen_output.get("diagram_manifest", []),
                "status": "completed",
                "error": None
            }
        except Exception as exc:
            return {
                "status": "failed",
                "error": f"Generation subgraph failure: {str(exc)}"
            }

    parent_builder = StateGraph(ParentSRSState)
    parent_builder.add_node("run_clarification", run_clarification)
    parent_builder.add_node("run_generation", run_generation)

    parent_builder.add_edge(START, "run_clarification")
    parent_builder.add_conditional_edges(
        "run_clarification",
        route_parent_transition,
        {
            "run_generation": "run_generation",
            END: END
        }
    )
    parent_builder.add_edge("run_generation", END)

    return parent_builder.compile(checkpointer=checkpointer)


def invoke_srs_subagent(
    graph,
    thread_id: str,
    user_prompt: Optional[str] = None,
    project_context: Optional[str] = None,
    resume_answer: Optional[str] = None,
    max_turns: int = 5
) -> SRSSubagentResponse:
    """
    Standard interface runner for the SRS Subagent.
    Handles initial execution and multi-turn HITL resumption.
    Returns typed SRSSubagentResponse for the Top-Level Supervisor.
    """
    config = {"configurable": {"thread_id": thread_id}}

    try:
        if resume_answer is not None:
            graph.invoke(Command(resume=resume_answer), config=config)
        else:
            if not user_prompt:
                raise ValueError("user_prompt must be provided for initial invocation.")
            initial_state: ParentSRSState = {
                "request_id": f"req_{int(time.time())}",
                "thread_id": thread_id,
                "user_prompt": user_prompt,
                "project_context": project_context,
                "max_turns": max_turns,
                "requirements": None,
                "unresolved_gaps": [],
                "final_document": None,
                "diagram_manifest": [],
                "status": "processing",
                "pending_question": None,
                "error": None
            }
            graph.invoke(initial_state, config=config)
    except GraphInterrupt:
        pass
    except Exception as exc:
        return SRSSubagentResponse(
            status="failed",
            thread_id=thread_id,
            error=str(exc)
        )

    current_state = graph.get_state(config)

    # Case A: Paused at interrupt or waiting for human input
    if current_state.values.get("status") == "waiting_human_input" or current_state.next:
        question_text = current_state.values.get("pending_question")
        if not question_text and current_state.tasks and current_state.tasks[0].interrupts:
            question_text = current_state.tasks[0].interrupts[0].value

        return SRSSubagentResponse(
            status="waiting_human_input",
            thread_id=thread_id,
            pending_question=str(question_text) if question_text else "Clarification needed.",
            unresolved_gaps=current_state.values.get("unresolved_gaps", [])
        )

    # Case B: Completed
    if current_state.values.get("final_document"):
        return SRSSubagentResponse(
            status="completed",
            thread_id=thread_id,
            final_document=current_state.values["final_document"],
            diagram_manifest=current_state.values.get("diagram_manifest", []),
            unresolved_gaps=current_state.values.get("unresolved_gaps", [])
        )

    if current_state.values.get("status") == "failed":
        return SRSSubagentResponse(
            status="failed",
            thread_id=thread_id,
            error=current_state.values.get("error", "SRS execution failure")
        )

    return SRSSubagentResponse(
        status="processing",
        thread_id=thread_id
    )

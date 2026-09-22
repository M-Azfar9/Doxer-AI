"""
Top-Level Supervisor StateGraph Assembly & Compilation (Phase 7 & Phase 10).
"""

from typing import Optional, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.core.service_registry import ServiceRegistry, services as default_services
from src.supervisor.state import SupervisorState
from src.supervisor.nodes import SupervisorNodes


def route_supervisor_intent(state: SupervisorState) -> str:
    """Conditional Edge: Routes to the appropriate specialized subagent."""
    route = state.get("route")
    if route == "doc_gen":
        return "dispatch_docgen"
    elif route == "srs":
        return "dispatch_srs"
    return "dispatch_qa"


def compile_supervisor_graph(
    services: Optional[ServiceRegistry] = None,
    docgen_version: str = "v1",
    checkpointer: Optional[Any] = None
):
    """
    Compiles the unified Top-Level Supervisor Graph.
    Connects QA (Feature 2), DocGen (Feature 1), and SRS (Feature 3).
    """
    services = services or default_services
    checkpointer = checkpointer or services.checkpointer or MemorySaver()

    nodes = SupervisorNodes(services=services, docgen_version=docgen_version)
    builder = StateGraph(SupervisorState)

    # 1. Register Supervisor Nodes
    builder.add_node("supervisor_classify", nodes.supervisor_classify)
    builder.add_node("dispatch_qa", nodes.dispatch_qa)
    builder.add_node("dispatch_docgen", nodes.dispatch_docgen)
    builder.add_node("dispatch_srs", nodes.dispatch_srs)

    # 2. Register Edges
    builder.add_edge(START, "supervisor_classify")
    builder.add_conditional_edges(
        "supervisor_classify",
        route_supervisor_intent,
        {
            "dispatch_qa": "dispatch_qa",
            "dispatch_docgen": "dispatch_docgen",
            "dispatch_srs": "dispatch_srs"
        }
    )
    builder.add_edge("dispatch_qa", END)
    builder.add_edge("dispatch_docgen", END)
    builder.add_edge("dispatch_srs", END)

    # 3. Compile with Checkpointer for Multi-Turn State Persistence
    graph = builder.compile(checkpointer=checkpointer)
    return graph, nodes

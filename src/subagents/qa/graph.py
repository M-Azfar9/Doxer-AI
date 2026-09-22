"""
LangGraph assembly and compilation for QA Subagent.
"""

from typing import Optional
from langgraph.graph import StateGraph, START, END

from src.core.service_registry import ServiceRegistry, services as default_services
from src.subagents.qa.state import QAState, RouteDecision
from src.subagents.qa.nodes import QANodes


def route_qa_intent(state: QAState) -> str:
    """Conditional edge routing based on classified route."""
    route = state.get("route")
    if route == RouteDecision.WEB_SEARCH:
        return "web_search"
    elif route == RouteDecision.RAG:
        return "retrieve_code"
    return "answer_direct"


def compile_qa_subgraph(services: Optional[ServiceRegistry] = None):
    """
    Compiles the 3-Way QA StateGraph.
    Can be run standalone or invoked as a node in the Top-Level Supervisor.
    """
    nodes = QANodes(services=services or default_services)
    builder = StateGraph(QAState)

    # Add nodes
    builder.add_node("classify_intent", nodes.classify_intent)
    builder.add_node("answer_direct", nodes.answer_direct)
    builder.add_node("web_search", nodes.web_search)
    builder.add_node("synthesize_web", nodes.synthesize_web)
    builder.add_node("retrieve_code", nodes.retrieve_code)
    builder.add_node("synthesize_code", nodes.synthesize_code)

    # Wire edges
    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        route_qa_intent,
        {
            "answer_direct": "answer_direct",
            "web_search": "web_search",
            "retrieve_code": "retrieve_code"
        }
    )
    builder.add_edge("answer_direct", END)
    builder.add_edge("web_search", "synthesize_web")
    builder.add_edge("synthesize_web", END)
    builder.add_edge("retrieve_code", "synthesize_code")
    builder.add_edge("synthesize_code", END)

    return builder.compile()

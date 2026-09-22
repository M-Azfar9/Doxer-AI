"""
Assembly and compilation for the DocGen Subagent graph.
Supports version switching: 'v1' (Phase 1 baseline) vs enhanced implementations.
"""

from typing import Optional
from langgraph.graph import StateGraph, START, END

from src.core.service_registry import ServiceRegistry, services as default_services
from src.subagents.docgen.state import DocGenState
from src.subagents.docgen.v1_baseline import DocGenV1BaselinePipeline


def compile_docgen_subgraph(
    services: Optional[ServiceRegistry] = None,
    version: str = "v1"
):
    """
    Compiles the DocGen StateGraph.
    Default version 'v1' represents Phase 1 Baseline:
    START -> repo_map_generator -> doc_intent_router -> direct_tool_fetch -> baseline_docgen -> END
    """
    services = services or default_services

    if version == "v1":
        pipeline = DocGenV1BaselinePipeline(services=services)
        builder = StateGraph(DocGenState)

        # Register nodes
        builder.add_node("repo_map_generator", pipeline.repo_map_generator_node)
        builder.add_node("doc_intent_router", pipeline.doc_intent_router_node)
        builder.add_node("direct_tool_fetch", pipeline.direct_tool_fetch_node)
        builder.add_node("baseline_docgen", pipeline.baseline_docgen_node)

        # Linear edges
        builder.add_edge(START, "repo_map_generator")
        builder.add_edge("repo_map_generator", "doc_intent_router")
        builder.add_edge("doc_intent_router", "direct_tool_fetch")
        builder.add_edge("direct_tool_fetch", "baseline_docgen")
        builder.add_edge("baseline_docgen", END)

        return builder.compile()

    raise ValueError(f"Unknown DocGen version: {version}. Currently supported: 'v1'")

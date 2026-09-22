"""
Top-Level Intent Classification and Routing for the Supervisor Graph.
"""

from typing import Optional
from src.core.service_registry import ServiceRegistry, services as default_services
from src.core.structured_output import StructuredOutputNode
from src.supervisor.state import SupervisorRouteDecision

SUPERVISOR_ROUTER_SYSTEM_PROMPT = """You are the Top-Level Supervisor Intent Router for Sprinter (DevDocs AI).
Your mission is to classify incoming user requests into EXACTLY ONE of three specialized subagents:

1. 'qa' (Question Answering & Technical Lookup):
   - Asking questions about programming languages, algorithms, standard libraries.
   - Searching the web for versions, releases, or quick documentation facts.
   - Finding or explaining a specific snippet of code or class within the local codebase.
   - Examples:
     * "What is dependency injection in Python?"
     * "What is the latest stable version of LangGraph?"
     * "Where is the VectorStoreClient defined in this repository?"
     * "How does quicksort work?"

2. 'doc_gen' (Technical Documentation Generator):
   - Requests to generate, write, or synthesize technical documentation (architecture explainers, API references, tutorials, onboarding guides).
   - Exploring repositories to create structured Markdown documentation manuals.
   - Examples:
     * "Document how authentication and routing work in this codebase."
     * "Generate an API reference for our search agent module."
     * "Create a developer onboarding and quickstart guide for this project."
     * "Write comprehensive architectural documentation for the repository at https://github.com/fastapi/fastapi"

3. 'srs' (Software Requirements Specification Assistant):
   - Requests to create, gather, define, or draft an IEEE 830 Software Requirements Specification (SRS) for a system or application.
   - Interactive specification, functional/non-functional requirements elicitation, actor modeling, and architectural diagrams for a planned project.
   - Examples:
     * "I need an SRS for a food delivery platform."
     * "Generate an IEEE 830 requirements specification for an AI-powered telemedicine app."
     * "Help me gather requirements and write the SRS for a crypto trading bot."
     * "Create software requirements and architecture diagrams for an e-commerce inventory system."

EXTRACTION RULES:
- If a remote GitHub repository URL (e.g., https://github.com/...) is in the query, extract it into `extracted_repo_url`.
- If a specific local directory path (e.g., ./src, D:/Projects/...) is in the query, extract it into `extracted_local_path`.
"""


class SupervisorRouter:
    """Classifies user queries to dispatch to QA, DocGen, or SRS."""

    def __init__(self, services: Optional[ServiceRegistry] = None):
        self.services = services or default_services
        self.node = StructuredOutputNode(
            llm=self.services.llm,
            schema=SupervisorRouteDecision,
            system_prompt=SUPERVISOR_ROUTER_SYSTEM_PROMPT
        )

    def route(self, query: str, context: Optional[str] = None) -> SupervisorRouteDecision:
        """Determines the target route for a query."""
        try:
            return self.node.invoke(user_prompt=query, context=context)
        except Exception as e:
            print(f"⚠️ [SupervisorRouter] Router fallback to 'qa': {e}")
            # Heuristic fallback
            q_lower = query.lower()
            if any(k in q_lower for k in ["srs", "software requirements", "ieee 830", "requirements spec"]):
                target = "srs"
            elif any(k in q_lower for k in ["document ", "generate doc", "api reference", "architecture guide", "write docs"]):
                target = "doc_gen"
            else:
                target = "qa"
            return SupervisorRouteDecision(
                route=target,
                reasoning="Heuristic fallback due to routing exception."
            )

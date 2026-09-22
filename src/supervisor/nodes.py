"""
Dispatcher bridge nodes connecting the Supervisor to child subagents.
"""

from typing import Dict, Any, Optional
from src.core.service_registry import ServiceRegistry, services as default_services
from src.supervisor.state import SupervisorState, SupervisorRouteDecision
from src.supervisor.router import SupervisorRouter

from src.subagents.qa import compile_qa_subgraph, QAState
from src.subagents.docgen import compile_docgen_subgraph, DocGenState
from src.subagents.srs import compile_srs_subgraph, invoke_srs_subagent, ParentSRSState


class SupervisorNodes:
    """Encapsulates nodes for the Supervisor Graph."""

    def __init__(
        self,
        services: Optional[ServiceRegistry] = None,
        docgen_version: str = "v1"
    ):
        self.services = services or default_services
        self.router = SupervisorRouter(services=self.services)
        self.docgen_version = docgen_version

        # Pre-compile subgraphs
        self.qa_subgraph = compile_qa_subgraph(services=self.services)
        self.docgen_subgraph = compile_docgen_subgraph(services=self.services, version=self.docgen_version)
        self.srs_subgraph = compile_srs_subgraph(services=self.services, checkpointer=self.services.checkpointer)

    def set_docgen_version(self, version: str):
        """Allows toggling DocGen version (v1 baseline vs v2/v3)."""
        self.docgen_version = version
        self.docgen_subgraph = compile_docgen_subgraph(services=self.services, version=version)

    def supervisor_classify(self, state: SupervisorState) -> Dict[str, Any]:
        """Node 1: Classifies incoming query into qa, doc_gen, or srs."""
        decision: SupervisorRouteDecision = self.router.route(state["query"])
        
        repo_url = state.get("repo_url") or decision.extracted_repo_url
        local_path = state.get("local_path") or decision.extracted_local_path

        metadata = state.get("metadata", {})
        metadata["route_reasoning"] = decision.reasoning

        return {
            "route": decision.route,
            "repo_url": repo_url,
            "local_path": local_path,
            "metadata": metadata
        }

    def dispatch_qa(self, state: SupervisorState) -> Dict[str, Any]:
        """Node 2A: Dispatches to Feature 2 (QA Subagent)."""
        qa_input: QAState = {
            "query": state["query"],
            "route": None,
            "needs_web_search": False,
            "max_results": 5,
            "search_results": [],
            "retrieved_chunks": [],
            "answer": "",
            "error": None
        }
        try:
            qa_output = self.qa_subgraph.invoke(qa_input)
            answer = qa_output.get("answer", "No answer was generated.")
            metadata = state.get("metadata", {})
            metadata["qa_route"] = str(qa_output.get("route", ""))
            return {
                "final_output": answer,
                "status": "completed",
                "metadata": metadata
            }
        except Exception as e:
            return {
                "final_output": f"QA Subagent error: {str(e)}",
                "status": "failed",
                "error": str(e)
            }

    def dispatch_docgen(self, state: SupervisorState) -> Dict[str, Any]:
        """Node 2B: Dispatches to Feature 1 (DocGen Subagent - Phase 1 Baseline)."""
        docgen_input: DocGenState = {
            "user_query": state["query"],
            "repo_url": state.get("repo_url"),
            "local_path": state.get("local_path"),
            "repo_map": "",
            "detected_tech_stack": [],
            "intent_plan": None,
            "retrieved_evidence": {},
            "draft_markdown": "",
            "citations": [],
            "error": None
        }
        try:
            docgen_output = self.docgen_subgraph.invoke(docgen_input)
            draft = docgen_output.get("draft_markdown", "")
            citations = docgen_output.get("citations", [])

            # Format final doc with citations
            if citations and "[^" not in draft:
                draft += "\n\n### References & Citations\n" + "\n".join([f"- {c}" for c in citations])

            metadata = state.get("metadata", {})
            metadata["docgen_version"] = self.docgen_version
            metadata["citations_count"] = len(citations)

            return {
                "final_output": draft,
                "status": "completed",
                "metadata": metadata
            }
        except Exception as e:
            return {
                "final_output": f"DocGen Subagent error: {str(e)}",
                "status": "failed",
                "error": str(e)
            }

    def dispatch_srs(self, state: SupervisorState) -> Dict[str, Any]:
        """Node 2C: Dispatches to Feature 3 (SRS Subagent with HITL & Workers)."""
        thread_id = state.get("thread_id") or "default_srs_thread"
        user_prompt = state["query"]
        project_context = state.get("local_path") or state.get("repo_url")

        res = invoke_srs_subagent(
            graph=self.srs_subgraph,
            thread_id=thread_id,
            user_prompt=user_prompt,
            project_context=project_context
        )

        metadata = state.get("metadata", {})
        metadata["srs_status"] = res.status
        metadata["diagram_count"] = len(res.diagram_manifest or [])

        if res.status == "waiting_human_input":
            return {
                "status": "waiting_human_input",
                "pending_question": res.pending_question,
                "final_output": f"Clarification Needed:\n{res.pending_question}",
                "metadata": metadata
            }
        elif res.status == "completed":
            return {
                "status": "completed",
                "final_output": res.final_document or "SRS Document generated successfully.",
                "metadata": metadata
            }
        else:
            return {
                "status": "failed",
                "final_output": f"SRS generation failed: {res.error}",
                "error": res.error,
                "metadata": metadata
            }

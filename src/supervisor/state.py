"""
Supervisor State schema and data models for Phase 7 & Phase 10.
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class SupervisorRouteDecision(BaseModel):
    """Structured output from the Top-Level Supervisor Intent Router."""
    route: Literal["qa", "doc_gen", "srs"] = Field(
        description="Target subagent: 'qa' for answering questions/search/code-lookup, 'doc_gen' for synthesizing documentation guides/API specs/architecture overviews, 'srs' for IEEE 830 software requirements specification generation."
    )
    reasoning: str = Field(description="Architectural rationale for why this route was selected.")
    extracted_repo_url: Optional[str] = Field(
        default=None,
        description="Remote GitHub repo URL if explicitly mentioned in query."
    )
    extracted_local_path: Optional[str] = Field(
        default=None,
        description="Local directory path if explicitly mentioned in query."
    )


class SupervisorState(TypedDict):
    """State schema for the Top-Level Supervisor Graph."""
    query: str
    conversation_history: List[Dict[str, str]]
    route: Optional[Literal["qa", "doc_gen", "srs"]]
    repo_url: Optional[str]
    local_path: Optional[str]
    final_output: str
    thread_id: str
    status: Literal["processing", "waiting_human_input", "completed", "failed"]
    pending_question: Optional[str]
    metadata: Dict[str, Any]
    error: Optional[str]


class SupervisorResult(BaseModel):
    """Clean user-facing response container from Sprinter."""
    status: Literal["processing", "waiting_human_input", "completed", "failed"]
    route: str
    thread_id: str
    output: Optional[str] = None
    pending_question: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

"""
State schema and structured output models for DocGen Subagent (Feature 1).
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class SpecialistScope(BaseModel):
    """Scope configuration for an individual data source."""
    enabled: bool = Field(description="Whether this source should be consulted")
    focus_scope: str = Field(description="Strict scope guidance for what to search or read")
    target_paths_or_topics: List[str] = Field(
        default_factory=list,
        description="Specific file paths, directories, or search queries"
    )


class DocIntentPlan(BaseModel):
    """Structured plan produced by the intent router node."""
    doc_type: Literal["architecture_explainer", "api_reference", "tutorial_quickstart"] = Field(
        description="The archetype of technical documentation to synthesize"
    )
    needs_github: SpecialistScope = Field(description="Guidance for remote GitHub exploration")
    needs_filesystem: SpecialistScope = Field(description="Guidance for local filesystem exploration")
    needs_web: SpecialistScope = Field(description="Guidance for web search")
    planning_rationale: str = Field(description="Detailed explanation of why these sources and doc type were selected")


class DocGenState(TypedDict):
    """Internal State schema for the DocGen Subgraph."""
    user_query: str
    repo_url: Optional[str]
    local_path: Optional[str]
    repo_map: str
    detected_tech_stack: List[str]
    intent_plan: Optional[DocIntentPlan]
    retrieved_evidence: Dict[str, Any]
    draft_markdown: str
    citations: List[str]
    error: Optional[str]

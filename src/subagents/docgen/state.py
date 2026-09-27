"""
State schema and structured output models for DocGen Subagent (Feature 1).
Supports Phase 1 Baseline and Phase 3 Architecture D closed-loop pipeline.
"""

import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Router Schemas
# ---------------------------------------------------------------------------
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
    needs_web: SpecialistScope = Field(description="Guidance for web search (official docs, versions, tutorials)")
    planning_rationale: str = Field(description="Detailed explanation of why these sources and doc type were selected")


# ---------------------------------------------------------------------------
# 2. Sufficiency Gate Schema
# ---------------------------------------------------------------------------
class SufficiencyCheck(BaseModel):
    """Structured decision returned by the specialist's sufficiency gate."""
    is_sufficient: bool = Field(
        description="True if the accumulated evidence is sufficient to thoroughly document the scoped topic; False if critical gaps remain."
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Confidence level in sufficiency (0.0 = completely inadequate, 1.0 = fully verified)."
    )
    missing_information: Optional[str] = Field(
        default=None,
        description="Concise summary of missing files, classes, endpoints, or concepts if is_sufficient is False."
    )
    next_action_guidance: Optional[str] = Field(
        default=None,
        description="Concrete suggestion for the next tool hop."
    )
    suggested_targets: List[str] = Field(
        default_factory=list,
        description="Specific next file paths or next search queries to execute in the next hop."
    )


# ---------------------------------------------------------------------------
# 3. AST Symbols & Evidence Packages
# ---------------------------------------------------------------------------
class APIEndpointSymbol(BaseModel):
    """Structured symbol extracted deterministically via Python AST analysis."""
    name: str = Field(description="Class, function, or method name")
    kind: Literal["function", "async_function", "class", "method", "route"] = Field(
        description="Type of symbol"
    )
    path: str = Field(description="File path where symbol is defined")
    signature: str = Field(description="Function/method signature with arguments and return types")
    decorators: List[str] = Field(default_factory=list, description="Decorators applied, e.g., @router.get or @property")
    docstring: Optional[str] = Field(default=None, description="Docstring summary if present")


class GitHubEvidencePackage(BaseModel):
    """Typed evidence package emitted by the GitHub specialist."""
    source: Literal["github"] = "github"
    files_examined: List[str] = Field(default_factory=list)
    key_abstractions: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of key classes/functions with their role and file path"
    )
    file_contents: Dict[str, str] = Field(
        default_factory=dict,
        description="Key extracted code blocks or full file text"
    )
    architectural_notes: List[str] = Field(default_factory=list)


class FSEvidencePackage(BaseModel):
    """Typed evidence package emitted by the FileSystem specialist with AST awareness."""
    source: Literal["filesystem"] = "filesystem"
    scanned_paths: List[str] = Field(default_factory=list)
    key_symbols_or_routes: List[str] = Field(default_factory=list)
    endpoints: List[APIEndpointSymbol] = Field(
        default_factory=list,
        description="Structured AST symbols and signatures discovered"
    )
    models: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Discovered Pydantic/dataclass models with fields and types"
    )
    file_contents: Dict[str, str] = Field(
        default_factory=dict,
        description="Key extracted file code contents"
    )
    architectural_notes: List[str] = Field(default_factory=list)


class WebEvidencePackage(BaseModel):
    """Typed evidence package emitted by the Web Research specialist."""
    source: Literal["web"] = "web"
    claims_supported: List[str] = Field(default_factory=list)
    sources: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of sources consulted: [{'title': ..., 'url': ..., 'snippet': ...}]"
    )
    verified_code_examples: List[str] = Field(default_factory=list)
    compatibility_notes: List[str] = Field(default_factory=list)


class MergedEvidenceContext(BaseModel):
    """Unified, normalized evidence context ready for document synthesis."""
    local_files: List[Dict[str, Any]] = Field(default_factory=list)
    github_files: List[Dict[str, Any]] = Field(default_factory=list)
    web_snippets: List[Dict[str, Any]] = Field(default_factory=list)
    endpoints: List[APIEndpointSymbol] = Field(default_factory=list)
    evidence_summary: str = Field(default="", description="High-level synopsis of total evidence collected")


# ---------------------------------------------------------------------------
# 4. Grounding Critique Schemas
# ---------------------------------------------------------------------------
class GroundingViolation(BaseModel):
    """Specific factual discrepancy or unsupported assertion flagged by the critic."""
    claim: str = Field(description="The exact claim or signature in the draft that lacks evidence")
    location: str = Field(description="Section heading or paragraph where the violation occurs")
    issue: Literal["hallucination", "incorrect_signature", "unsupported_assertion", "missing_reference"] = Field(
        description="Category of grounding violation"
    )
    correction_instruction: str = Field(description="Precise instruction on how to correct or verify the claim")


class GroundingReport(BaseModel):
    """Audit report produced by the grounding critic node."""
    is_grounded: bool = Field(description="True if all claims are substantiated by evidence; False if hallucinations exist")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Verification confidence score (0.0 to 1.0)")
    violations: List[GroundingViolation] = Field(default_factory=list, description="List of detected discrepancies")
    targeted_refinement_queries: List[str] = Field(
        default_factory=list,
        description="Concrete queries for targeted gap filling"
    )


# ---------------------------------------------------------------------------
# 5. Specialist Subgraph Internal State Schemas
# ---------------------------------------------------------------------------
class FSSpecialistState(TypedDict):
    """Internal state for the FileSystem specialist subgraph."""
    local_path: str
    scope: str
    target_paths: List[str]
    repo_map: str
    hop_count: int
    max_hops: int
    files_accumulated: Dict[str, str]
    discovered_symbols: List[str]
    discovered_endpoints: List[APIEndpointSymbol]
    discovered_models: List[Dict[str, Any]]
    sufficiency: Optional[SufficiencyCheck]
    package: Optional[FSEvidencePackage]
    specialist_outputs: Optional[List[Dict[str, Any]]]


class GitHubSpecialistState(TypedDict):
    """Internal state for the remote GitHub specialist subgraph."""
    repo_url: str
    scope: str
    target_paths: List[str]
    repo_map: str
    hop_count: int
    max_hops: int
    files_accumulated: Dict[str, str]
    key_abstractions: List[Dict[str, str]]
    sufficiency: Optional[SufficiencyCheck]
    package: Optional[GitHubEvidencePackage]
    specialist_outputs: Optional[List[Dict[str, Any]]]


class WebSpecialistState(TypedDict):
    """Internal state for the Web Research specialist subgraph."""
    user_query: str
    scope: str
    search_topics: List[str]
    hop_count: int
    max_hops: int
    snippets_accumulated: List[Dict[str, str]]
    sufficiency: Optional[SufficiencyCheck]
    package: Optional[WebEvidencePackage]
    specialist_outputs: Optional[List[Dict[str, Any]]]


# ---------------------------------------------------------------------------
# 6. Top-Level Graph State Schema
# ---------------------------------------------------------------------------
class DocGenState(TypedDict, total=False):
    """
    Comprehensive State schema for DocGen Subgraph.
    Compatible across Phase 1 Baseline and Phase 3 Architecture D closed-loop pipeline.
    """
    user_query: str
    repo_url: Optional[str]
    local_path: Optional[str]
    repo_map: str
    detected_tech_stack: List[str]
    intent_plan: Optional[DocIntentPlan]

    # Legacy Phase 1 Baseline fields
    retrieved_evidence: Dict[str, Any]

    # Phase 3 Architecture D LangGraph Fan-In Reducer
    specialist_outputs: Annotated[List[Dict[str, Any]], operator.add]
    merged_evidence: Optional[MergedEvidenceContext]

    # Synthesis & Critique Loop State
    draft_markdown: str
    citations: List[str]
    grounding_report: Optional[GroundingReport]
    refinement_count: int

    # Final validated output & Metadata
    final_doc_markdown: str
    citations_manifest: List[Dict[str, Any]]
    grounding_score: float
    error: Optional[str]


# Backward-compatibility aliases
DocGenPhase3State = DocGenState
DocGenPhase2State = DocGenState
DocGenPhase1State = DocGenState

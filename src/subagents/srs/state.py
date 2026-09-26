"""
State schemas and data models for the SRS Subagent (Feature 3).
"""

import json
import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, model_validator


def _parse_dict_stringified_lists(data: Any) -> Any:
    """Safely converts stringified JSON arrays into lists if an LLM returns them as strings."""
    if isinstance(data, dict):
        for k, v in list(data.items()):
            if isinstance(v, str):
                v_clean = v.strip()
                if v_clean.startswith("[") and v_clean.endswith("]"):
                    try:
                        data[k] = json.loads(v_clean)
                    except Exception:
                        pass
    return data


# ---------------------------------------------------------------------------
# Core Requirements Models (IEEE 830 compliant)
# ---------------------------------------------------------------------------
class Actor(BaseModel):
    name: str = Field(description="Name of the user role or external system")
    description: str = Field(description="Role responsibilities and access permissions")


class FunctionalRequirement(BaseModel):
    req_id: str = Field(description="Unique identifier e.g. FR-001")
    title: str = Field(description="Short, descriptive title")
    description: str = Field(description="Detailed requirement description")
    priority: Literal["Must Have", "Should Have", "Could Have"] = Field(
        default="Must Have", 
        description="Requirement priority"
    )
    acceptance_criteria: List[str] = Field(
        default_factory=list,
        description="Testable conditions that satisfy the requirement"
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_lists(cls, data: Any) -> Any:
        return _parse_dict_stringified_lists(data)


class NonFunctionalRequirement(BaseModel):
    category: Literal["Performance", "Security", "Scalability", "Reliability", "Usability"] = Field(
        description="NFR category"
    )
    description: str = Field(description="Specific quantifiable constraint")
    metric: Optional[str] = Field(
        default=None, 
        description="Measurable target e.g. 'p99 latency < 200ms at 5000 RPS'"
    )


class RequirementsModel(BaseModel):
    project_title: str = Field(default="Software System", description="Title of the project")
    project_scope: str = Field(default="", description="High-level product vision and operational boundaries")
    target_users_and_actors: List[Actor] = Field(default_factory=list)
    functional_requirements: List[FunctionalRequirement] = Field(default_factory=list)
    non_functional_requirements: List[NonFunctionalRequirement] = Field(default_factory=list)
    system_constraints: List[str] = Field(
        default_factory=list, 
        description="Technical stack, regulatory, or environment rules"
    )
    assumptions_and_dependencies: List[str] = Field(
        default_factory=list,
        description="Underlying assumptions and third-party dependencies"
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_lists(cls, data: Any) -> Any:
        return _parse_dict_stringified_lists(data)


# ---------------------------------------------------------------------------
# Completeness Auditor Schema
# ---------------------------------------------------------------------------
class CompletenessCheck(BaseModel):
    reasoning: str = Field(
        description="Step-by-step audit of current requirements against standard IEEE 830 dimensions"
    )
    missing_areas: List[str] = Field(
        description="Explicit list of ambiguous or missing dimensions. MUST be populated before deciding is_complete."
    )
    is_complete: bool = Field(
        description="True ONLY if no critical functional or architectural ambiguities remain"
    )
    next_question: Optional[str] = Field(
        default=None, 
        description="A targeted, clear question focusing on the highest-priority missing area. None if is_complete is True."
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_lists(cls, data: Any) -> Any:
        return _parse_dict_stringified_lists(data)


# ---------------------------------------------------------------------------
# Clarification Subgraph State Schema
# ---------------------------------------------------------------------------
class ClarificationState(TypedDict):
    user_prompt: str
    project_context: Optional[str]
    conversation_log: Annotated[List[Dict[str, str]], operator.add]
    turn_count: int
    max_turns: int
    requirements: RequirementsModel
    completeness: Optional[CompletenessCheck]
    unresolved_gaps: List[str]


# ---------------------------------------------------------------------------
# Generation Stage Models (Phase 2)
# ---------------------------------------------------------------------------
class OutlineSection(BaseModel):
    section_id: str = Field(description="Section number e.g. 1.0, 2.1")
    title: str = Field(description="Section heading")
    purpose: str = Field(description="Goal and scope of this section")
    target_word_count: int = Field(default=350, description="Target length")
    key_points: List[str] = Field(default_factory=list, description="Mandatory elements to cover")
    needs_diagram: bool = Field(default=False, description="Whether this section includes a visual diagram")
    diagram_type: Optional[Literal["sequence", "flowchart", "class", "component", "state"]] = Field(
        default=None, 
        description="Architecture diagram archetype if needs_diagram is True"
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_lists(cls, data: Any) -> Any:
        return _parse_dict_stringified_lists(data)


class DocOutline(BaseModel):
    document_title: str
    target_standard: str = "IEEE 830-1998"
    version: str = "1.0.0"
    sections: List[OutlineSection]

    @model_validator(mode="before")
    @classmethod
    def _coerce_lists(cls, data: Any) -> Any:
        return _parse_dict_stringified_lists(data)


class SectionDraft(BaseModel):
    section_id: str
    title: str
    content_markdown: str
    generated_by: str = "SectionWorker"


class DiagramSpec(BaseModel):
    diagram_id: str
    diagram_type: Literal["sequence", "flowchart", "class", "component", "state"]
    format: Literal["mermaid"] = "mermaid"
    title: str
    code: str
    caption: str


class ValidatedDiagram(BaseModel):
    diagram_id: str
    diagram_type: str
    code: str
    caption: str
    is_valid: bool
    syntax_error: Optional[str] = None
    rendered_path: Optional[str] = None


class PlannedDiagrams(BaseModel):
    diagrams: List[DiagramSpec]

    @model_validator(mode="before")
    @classmethod
    def _coerce_lists(cls, data: Any) -> Any:
        return _parse_dict_stringified_lists(data)


class GenerationState(TypedDict):
    requirements: RequirementsModel
    unresolved_gaps: List[str]
    outline: Optional[DocOutline]
    section_drafts: Annotated[List[SectionDraft], operator.add]
    diagram_specs: List[DiagramSpec]
    validated_diagrams: List[ValidatedDiagram]
    final_document: Optional[str]
    diagram_manifest: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Parent Graph State & Supervisor Interface (Phase 3)
# ---------------------------------------------------------------------------
class ParentSRSState(TypedDict):
    request_id: str
    thread_id: str
    user_prompt: str
    project_context: Optional[str]
    max_turns: int
    requirements: Optional[RequirementsModel]
    unresolved_gaps: List[str]
    final_document: Optional[str]
    diagram_manifest: List[Dict[str, Any]]
    status: Literal["processing", "waiting_human_input", "completed", "failed"]
    pending_question: Optional[str]
    error: Optional[str]


class SRSSubagentResponse(BaseModel):
    status: Literal["waiting_human_input", "processing", "completed", "failed"]
    thread_id: str
    pending_question: Optional[str] = None
    final_document: Optional[str] = None
    diagram_manifest: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    unresolved_gaps: Optional[List[str]] = Field(default_factory=list)
    error: Optional[str] = None

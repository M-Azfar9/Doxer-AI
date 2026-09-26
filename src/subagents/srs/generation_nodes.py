"""
Generation stage nodes for the SRS Subagent (Phase 2).
Outlines document, drafts sections, plans/validates Mermaid diagrams, and deterministically assembles the final IEEE 830 document.
"""

import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.messages import SystemMessage, HumanMessage

from src.core.service_registry import ServiceRegistry, services as default_services
from src.core.structured_output import StructuredOutputNode
from src.subagents.srs.state import (
    GenerationState, RequirementsModel, DocOutline, OutlineSection,
    SectionDraft, DiagramSpec, ValidatedDiagram, PlannedDiagrams
)

OUTLINE_GENERATION_SYSTEM_PROMPT = """You are a Principal Technical Writer and Software Architect designing an IEEE 830-compliant Software Requirements Specification (SRS).
Your task is to review the structured RequirementsModel and any unresolved gaps, and generate a comprehensive, professional document outline (DocOutline).

Standard IEEE 830 Sections to Include:
1. "1.0 Introduction" (Purpose, Scope, Intended Audience, Document Conventions)
2. "2.0 Overall Description" (Product Perspective, User Classes & Actors, Operating Environment, Design Constraints, Assumptions & Dependencies)
3. "3.0 System Features & Functional Requirements" (Grouped by major feature or user workflow with clear functional breakdown)
4. "4.0 External Interface Requirements" (User, Hardware, Software/API, and Communication Interfaces)
5. "5.0 Non-Functional & Quality Attributes" (Performance, Security/Auth, Reliability, Scalability, and Compliance)

Thin-Input & Gap Resiliency Rules:
1. If `unresolved_gaps` contains items, append a dedicated section: "6.0 Unresolved Requirements & Ambiguity Register" to track open items.
2. Provide specific instructions in each section detailing which actors, functional requirements, or constraints should be drafted.
3. Keep the outline to 5–7 major sections to allow clean parallel worker drafting.
"""

SECTION_DRAFTING_SYSTEM_PROMPT = """You are an Expert Systems Architect drafting a specific section of an IEEE 830 Software Requirements Specification.
Draft thorough, exhaustive, professional Markdown for the requested section. Satisfy the target word count and cover all prescribed key points.
Ground your drafting directly in the provided RequirementsModel without hallucinating unsubstantiated features.
"""

DIAGRAM_PLANNER_SYSTEM_PROMPT = """You are a Principal Software Solutions Architect determining visual modeling specifications for an IEEE 830 SRS document.
Review the system requirements and plan between 2 and 4 essential architectural diagrams using Mermaid syntax.

Diagram Typology Mapping Rules:
1. High-Level Architecture & Component Decomposition: Use diagram_type="flowchart" or "component". The Mermaid code MUST start with `flowchart TD` or `flowchart LR` (using subgraphs for components). Never write "component diagram" or "component TD" because Mermaid requires `flowchart TD`.
2. User Flows, Authentication Workflows, & Step-by-Step Message Sequences: Use diagram_type="sequence". The Mermaid code MUST start with `sequenceDiagram`.
3. Domain Entities, Data Models, & Schemas: Use diagram_type="class". The Mermaid code MUST start with `classDiagram`.
4. State Transitions or Lifecycle Workflows: Use diagram_type="state". The Mermaid code MUST start with `stateDiagram-v2`.

Mermaid Syntax & Quality Requirements:
1. The first line of Mermaid code MUST be one of these exact declarations:
   - `flowchart TD` or `flowchart LR` (for architecture, flowchart, and component models)
   - `sequenceDiagram` (for interaction and message sequences)
   - `classDiagram` (for entity and data structure models)
   - `stateDiagram-v2` (for lifecycle state machines)
2. Ensure all node brackets e.g. `[ ]`, `( )`, `{ }`, and quotes `" "` are strictly matched and closed.
3. Node IDs must be alphanumeric identifiers without spaces (e.g., `ClientApp["Client Application"]`).
4. In sequence diagrams, define participants and use valid arrows (`->>`, `-->>`).
5. Output clean, raw Mermaid code without markdown code fences (do NOT include ```mermaid).
6. Provide an informative title and a formal IEEE 830 figure caption for each diagram (e.g. "Figure 1: High-Level System Architecture").
7. Plan at least 2 and at most 4 diagrams representing the most critical architectural perspectives of the system.
"""

SUPPORTED_DIAGRAM_TYPES = {"flowchart", "sequence", "class", "component", "state", "erDiagram"}
VALID_MERMAID_HEADERS = (
    "graph ", "graph\n", "flowchart ", "flowchart\n", "sequencediagram", "classdiagram", 
    "statediagram", "statediagram-v2", "erdiagram", "c4context", "c4container"
)
DANGEROUS_PATTERNS = [
    r"<script\b", r"javascript:", r"<iframe\b", r"onerror=", r"onload=", r"<embed\b"
]


def validate_mermaid_code(diagram_type: str, raw_code: str) -> Tuple[bool, Optional[str]]:
    """
    Validates Mermaid syntax, checks delimiter balance, verifies headers,
    and prevents render crashes (invalid syntax tags, infinite loops, unsupported types).
    """
    if diagram_type not in SUPPORTED_DIAGRAM_TYPES:
        return False, f"Unsupported diagram type '{diagram_type}'. Supported types: {sorted(SUPPORTED_DIAGRAM_TYPES)}"

    code = raw_code.strip()
    # Strip markdown code fences if wrapped
    if code.startswith("```"):
        lines = code.splitlines()
        if len(lines) > 2:
            code = "\n".join(lines[1:-1]).strip()
        else:
            return False, "Empty or malformed code block"

    # Normalize component header to valid Mermaid flowchart TD
    code = re.sub(r'^(component diagram|component TD|component LR|component\s*(\n|$))\b', 'flowchart TD\n', code, flags=re.IGNORECASE).strip()

    if not code:
        return False, "Diagram code is empty"

    lines = [ln.strip() for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("%%")]
    if not lines:
        return False, "Diagram code contains no statements"

    # Check header
    header_line = lines[0].lower()
    has_valid_header = any(header_line.startswith(h) or header_line == h.strip() for h in VALID_MERMAID_HEADERS)
    if not has_valid_header:
        return False, f"Missing or malformed Mermaid diagram header: '{lines[0]}'"

    if len(lines) < 2:
        return False, "Empty diagram body: no nodes or transitions defined"

    # Check dangerous / crash tags (XSS & injection defense)
    for pat in DANGEROUS_PATTERNS:
        if re.search(pat, code, re.IGNORECASE):
            return False, "Security violation: Forbidden script tag or injection pattern detected"

    # Check delimiter balance: [ ], ( ), { }
    pairs = {'[': ']', '(': ')', '{': '}'}
    stack = []
    in_quote = False
    escaped = False

    for ch in code:
        if ch == '\n':
            in_quote = False
            escaped = False
            continue
        if escaped:
            escaped = False
            continue
        if ch == '\\':
            escaped = True
            continue
        if ch == '"':
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        
        if ch in pairs:
            stack.append(ch)
        elif ch in pairs.values():
            if not stack:
                return False, f"Unmatched closing delimiter '{ch}'"
            top = stack.pop()
            if pairs[top] != ch:
                return False, f"Mismatched delimiters: expected '{pairs[top]}' but found '{ch}'"

    if stack:
        unmatched = [pairs[ch] for ch in stack]
        return False, f"Unclosed opening delimiter(s): expected '{', '.join(unmatched)}'"

    # Check dangling arrows or broken relationship operators
    for line in lines[1:]:
        if line.rstrip().endswith(("-->", "->>", "-->>", "==>", "-.->")):
            return False, f"Malformed syntax: dangling arrow without destination in line: '{line}'"

    return True, None


class GenerationNodes:
    """Encapsulates execution nodes for the generation subgraph."""

    def __init__(self, services: Optional[ServiceRegistry] = None):
        self.services = services or default_services
        self.outline_generator = StructuredOutputNode(
            llm=self.services.llm,
            schema=DocOutline,
            system_prompt=OUTLINE_GENERATION_SYSTEM_PROMPT
        )
        self.diagram_planner = StructuredOutputNode(
            llm=self.services.llm,
            schema=PlannedDiagrams,
            system_prompt=DIAGRAM_PLANNER_SYSTEM_PROMPT
        )

    def generate_outline(self, state: GenerationState) -> Dict[str, Any]:
        """Node 1: Generates DocOutline from captured requirements."""
        reqs = state["requirements"]
        unresolved = state.get("unresolved_gaps", [])

        prompt = (
            f"Requirements Model:\n{json.dumps(reqs.model_dump(), indent=2)}\n\n"
            f"Unresolved Gaps Register:\n{json.dumps(unresolved, indent=2)}"
        )
        try:
            outline = self.outline_generator.invoke(user_prompt=prompt)
        except Exception as e:
            print(f"⚠️ [generate_outline] Fallback outline: {e}")
            outline = DocOutline(
                document_title=f"{reqs.project_title} Software Requirements Specification",
                sections=[
                    OutlineSection(section_id="1.0", title="Introduction", purpose="System overview", key_points=["Purpose", "Scope"]),
                    OutlineSection(section_id="2.0", title="Overall Description", purpose="Actors and constraints", key_points=["User Classes", "Constraints"]),
                    OutlineSection(section_id="3.0", title="System Features & Functional Requirements", purpose="Functional specs", key_points=["Features", "Criteria"]),
                    OutlineSection(section_id="4.0", title="Non-Functional Requirements", purpose="Quality attributes", key_points=["Security", "Performance"])
                ]
            )
        return {"outline": outline}

    def generate_section(self, state: GenerationState) -> Dict[str, Any]:
        """Node 2: Generates drafts for all sections in outline."""
        outline = state.get("outline")
        reqs = state["requirements"]
        drafts: List[SectionDraft] = []

        if not outline or not outline.sections:
            return {"section_drafts": drafts}

        reqs_dump = json.dumps(reqs.model_dump(), indent=2)

        for sec in outline.sections:
            sec_prompt = (
                f"Document: {outline.document_title}\n"
                f"Section: {sec.section_id} - {sec.title}\n"
                f"Purpose: {sec.purpose}\n"
                f"Key Points to Cover: {', '.join(sec.key_points)}\n"
                f"Target Word Count: {sec.target_word_count}\n\n"
                f"System Requirements:\n{reqs_dump}"
            )
            messages = [
                SystemMessage(content=SECTION_DRAFTING_SYSTEM_PROMPT),
                HumanMessage(content=sec_prompt)
            ]
            response = self.services.llm.invoke(messages)
            content = getattr(response, "content", str(response))
            drafts.append(SectionDraft(
                section_id=sec.section_id,
                title=sec.title,
                content_markdown=content
            ))

        return {"section_drafts": drafts}

    def determine_diagrams(self, state: GenerationState) -> Dict[str, Any]:
        """Node 3: Plans architectural diagrams."""
        reqs = state["requirements"]
        prompt = f"Requirements Model:\n{json.dumps(reqs.model_dump(), indent=2)}"
        try:
            planned = self.diagram_planner.invoke(user_prompt=prompt)
            specs = planned.diagrams
        except Exception as e:
            print(f"⚠️ [determine_diagrams] Fallback diagrams: {e}")
            specs = [
                DiagramSpec(
                    diagram_id="diag-01",
                    diagram_type="flowchart",
                    title="High-Level Architecture",
                    code="graph TD\n    User[Client Application] --> API[API Gateway]\n    API --> Service[Core Service]\n    Service --> DB[(Database)]",
                    caption="Figure 1: High-Level Architecture Overview"
                )
            ]
        return {"diagram_specs": specs}

    def generate_diagram(self, state: GenerationState) -> Dict[str, Any]:
        """Node 4: Validates planned diagrams and formats them."""
        specs = state.get("diagram_specs", [])
        validated: List[ValidatedDiagram] = []
        manifest: List[Dict[str, Any]] = []

        for spec in specs:
            code = spec.code.strip()
            # Normalize component diagram header to standard flowchart TD
            code = re.sub(r'^(component diagram|component TD|component LR|component\s*(\n|$))\b', 'flowchart TD\n', code, flags=re.IGNORECASE).strip()
            is_valid, syntax_error = validate_mermaid_code(spec.diagram_type, code)
            val = ValidatedDiagram(
                diagram_id=spec.diagram_id,
                diagram_type=spec.diagram_type,
                code=code,
                caption=spec.caption,
                is_valid=is_valid,
                syntax_error=syntax_error
            )
            validated.append(val)
            manifest.append({
                "diagram_id": spec.diagram_id,
                "title": spec.title,
                "type": spec.diagram_type,
                "caption": spec.caption,
                "is_valid": is_valid,
                "syntax_error": syntax_error
            })

        return {"validated_diagrams": validated, "diagram_manifest": manifest}

    def assemble_document(self, state: GenerationState) -> Dict[str, Any]:
        """Node 5: Deterministically merges drafts, requirements, and diagrams into final IEEE 830 Markdown."""
        reqs = state["requirements"]
        outline = state.get("outline")
        drafts = state.get("section_drafts", [])
        diagrams = state.get("validated_diagrams", [])
        unresolved = state.get("unresolved_gaps", [])

        title = outline.document_title if outline else f"{reqs.project_title} Software Requirements Specification"
        now_str = datetime.now().strftime("%B %d, %Y")

        doc_lines = [
            f"# {title}",
            f"**Standard:** IEEE 830-1998 Compliant  ",
            f"**Generated:** {now_str}  ",
            f"**Version:** 1.0.0  \n",
            "---",
            "\n## Table of Contents"
        ]

        for d in drafts:
            doc_lines.append(f"- [{d.section_id} {d.title}](#{d.section_id.replace('.', '').lower()}-{d.title.lower().replace(' ', '-')})")

        if diagrams:
            doc_lines.append("- [Visual Architecture Diagrams](#visual-architecture-diagrams)")

        doc_lines.append("\n---\n")

        for d in drafts:
            doc_lines.append(f"## {d.section_id} {d.title}\n")
            doc_lines.append(d.content_markdown.strip())
            doc_lines.append("\n---\n")

        if diagrams:
            doc_lines.append("## Visual Architecture Diagrams\n")
            for diag in diagrams:
                doc_lines.append(f"### {diag.caption}")
                doc_lines.append("```mermaid")
                doc_lines.append(diag.code)
                doc_lines.append("```\n")
            doc_lines.append("---\n")

        if unresolved:
            doc_lines.append("## Appendix: Unresolved Requirements & Ambiguity Register\n")
            doc_lines.append("The following items were identified as ambiguities during specification and are slated for Phase 2 validation:\n")
            for idx, gap in enumerate(unresolved, 1):
                doc_lines.append(f"{idx}. {gap}")
            doc_lines.append("\n")

        final_doc = "\n".join(doc_lines)
        return {"final_document": final_doc}

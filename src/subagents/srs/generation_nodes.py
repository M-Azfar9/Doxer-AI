"""
Generation stage nodes for the SRS Subagent (Phase 2).
Outlines document, drafts sections, plans/validates Mermaid diagrams, and deterministically assembles the final IEEE 830 document.
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
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
Review the system requirements and plan 2 to 4 essential architectural diagrams (flowchart, sequence, class, component, or state) using Mermaid syntax.
Ensure the Mermaid code is syntactically valid and well-formatted.
"""


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
            # Basic mermaid validation
            is_valid = ("graph " in code or "sequenceDiagram" in code or "classDiagram" in code or "flowchart " in code or "stateDiagram" in code)
            val = ValidatedDiagram(
                diagram_id=spec.diagram_id,
                diagram_type=spec.diagram_type,
                code=code,
                caption=spec.caption,
                is_valid=is_valid,
                syntax_error=None if is_valid else "Mermaid diagram header missing"
            )
            validated.append(val)
            manifest.append({
                "diagram_id": spec.diagram_id,
                "title": spec.title,
                "type": spec.diagram_type,
                "caption": spec.caption,
                "is_valid": is_valid
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

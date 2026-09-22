"""
Clarification stage nodes for the SRS Subagent (Phase 1).
Includes human-in-the-loop interrupt and requirements saturation checking.
"""

import json
from typing import Dict, Any, Optional
from langgraph.types import interrupt
from langgraph.graph import END
from langchain_core.messages import SystemMessage, HumanMessage

from src.core.service_registry import ServiceRegistry, services as default_services
from src.core.structured_output import StructuredOutputNode
from src.subagents.srs.state import (
    ClarificationState, RequirementsModel, CompletenessCheck
)

INITIAL_ANALYSIS_SYSTEM_PROMPT = """You are a Principal Software Systems Architect initializing an IEEE 830 Software Requirements Specification (SRS).
Your job is to analyze the user's initial project prompt and any provided technical context, and construct an initial structured RequirementsModel.

Guidelines:
1. Extract all explicit functional and non-functional requirements.
2. Identify user roles and external system actors.
3. Identify known system constraints (tech stack, regulatory, database preferences) and assumptions.
4. DO NOT hallucinate or invent specific architectural decisions if the user has not mentioned them. Leave ambiguous areas empty or general so the completeness auditor can clarify them with the user.
5. Provide clear, professional requirement descriptions with testable acceptance criteria where applicable.
"""

COMPLETENESS_AUDIT_SYSTEM_PROMPT = """You are a Lead Systems Auditor evaluating an IEEE 830 Software Requirements Specification (SRS).
Your job is to audit the current structured RequirementsModel for completeness, clarity, and architectural soundness.

IEEE 830 Core Dimensions to Check:
1. Target Users & Actors: Are roles and external integrations clearly identified?
2. Functional Scope: Are core workflows, data flows, and edge cases defined with clear acceptance criteria?
3. Non-Functional Requirements: Are performance, security (auth/RBAC), and reliability quantified with measurable metrics?
4. Technical Constraints: Are tech stacks, data storage mechanisms, or regulatory requirements specified?

Operational Rules:
- Enumerate every specific ambiguity or gap in `missing_areas` FIRST.
- If critical dimensions remain undefined and the user has not yet reached turn limits, set `is_complete = False`.
- In `next_question`, ask exactly ONE crisp, prioritized, highly actionable question targeting the most critical missing area.
- If the core system boundaries and critical requirements are well-defined, set `is_complete = True` and `next_question = None`.
"""

DELTA_UPDATE_SYSTEM_PROMPT = """You are a Requirements Engineering Lead updating an IEEE 830 RequirementsModel based on a developer's latest clarification answer.

Rules:
1. You are given the CURRENT RequirementsModel and the LATEST question and answer.
2. Incorporate all new details: update or add functional requirements, add actors, specify NFR metrics, or record system constraints.
3. Preserve all existing valid requirements—do not drop or overwrite previously established requirements unless the user explicitly contradicted them.
4. Keep requirement IDs consistent (e.g. FR-001, FR-002, NFR-001).
5. Ensure acceptance criteria and metrics reflect the new information provided.
"""


class ClarificationNodes:
    """Encapsulates execution nodes for the clarification subgraph."""

    def __init__(self, services: Optional[ServiceRegistry] = None):
        self.services = services or default_services
        self.initial_analyzer = StructuredOutputNode(
            llm=self.services.llm,
            schema=RequirementsModel,
            system_prompt=INITIAL_ANALYSIS_SYSTEM_PROMPT
        )
        self.completeness_auditor = StructuredOutputNode(
            llm=self.services.llm,
            schema=CompletenessCheck,
            system_prompt=COMPLETENESS_AUDIT_SYSTEM_PROMPT
        )
        self.delta_updater = StructuredOutputNode(
            llm=self.services.llm,
            schema=RequirementsModel,
            system_prompt=DELTA_UPDATE_SYSTEM_PROMPT
        )

    def analyze_initial_requirements(self, state: ClarificationState) -> Dict[str, Any]:
        """Node 1: Parses raw user prompt into initial structured RequirementsModel."""
        user_prompt = state["user_prompt"]
        context = state.get("project_context") or "No additional context provided."
        prompt = f"Project Prompt:\n{user_prompt}\n\nTechnical & Architecture Context:\n{context}"
        try:
            reqs = self.initial_analyzer.invoke(user_prompt=prompt)
        except Exception as e:
            print(f"⚠️ [analyze_initial_requirements] Fallback: {e}")
            reqs = RequirementsModel(
                project_title="Software System",
                project_scope=user_prompt[:200]
            )
        return {"requirements": reqs}

    def check_completeness(self, state: ClarificationState) -> Dict[str, Any]:
        """Node 2: Evaluates requirements completeness."""
        reqs = state["requirements"]
        turn_count = state.get("turn_count", 0)
        max_turns = state.get("max_turns", 5)

        # Build summary
        model_summary = f"""Project Title: {reqs.project_title}
Scope: {reqs.project_scope}
Actors: {json.dumps([a.model_dump() for a in reqs.target_users_and_actors], indent=2)}
Functional Requirements: {json.dumps([fr.model_dump() for fr in reqs.functional_requirements], indent=2)}
Non-Functional Requirements: {json.dumps([nfr.model_dump() for nfr in reqs.non_functional_requirements], indent=2)}
System Constraints: {json.dumps(reqs.system_constraints, indent=2)}
Current Turn: {turn_count} / {max_turns}
"""
        try:
            completeness = self.completeness_auditor.invoke(user_prompt=f"Review requirements:\n{model_summary}")
        except Exception as e:
            print(f"⚠️ [check_completeness] Fallback: {e}")
            completeness = CompletenessCheck(
                reasoning="Automated assessment",
                missing_areas=[],
                is_complete=True,
                next_question=None
            )

        if turn_count >= max_turns:
            completeness.is_complete = True
            completeness.next_question = None

        return {"completeness": completeness}

    def ask_question(self, state: ClarificationState) -> Dict[str, Any]:
        """Node 3: Zero-LLM Human-in-the-Loop Interrupt."""
        completeness = state["completeness"]
        question = completeness.next_question if completeness else "Could you provide additional system specifications?"
        if not question:
            question = "Could you specify core system functional requirements?"

        # Execution halts here; checkpointer stores state
        human_answer = interrupt(question)

        return {
            "conversation_log": [
                {"role": "assistant", "content": question},
                {"role": "human", "content": str(human_answer)}
            ],
            "turn_count": state.get("turn_count", 0) + 1
        }

    def update_requirements(self, state: ClarificationState) -> Dict[str, Any]:
        """Node 4: Merges user response into updated RequirementsModel."""
        current_reqs = state["requirements"]
        log = state.get("conversation_log", [])
        latest_user_answer = "No answer"
        latest_question = "Clarify requirements."

        if len(log) >= 2 and log[-1].get("role") == "human":
            latest_user_answer = log[-1]["content"]
            latest_question = log[-2]["content"]
        elif len(log) >= 1 and log[-1].get("role") == "human":
            latest_user_answer = log[-1]["content"]

        update_prompt = f"""Current Requirements Model:
{json.dumps(current_reqs.model_dump(), indent=2)}

Latest Clarification Question Asked:
"{latest_question}"

Developer's Answer Provided:
"{latest_user_answer}"

Produce the updated, merged RequirementsModel incorporating this latest clarification."""

        try:
            updated_reqs = self.delta_updater.invoke(user_prompt=update_prompt)
        except Exception as e:
            print(f"⚠️ [update_requirements] Fallback to current requirements: {e}")
            updated_reqs = current_reqs

        return {"requirements": updated_reqs}


def route_after_completeness(state: ClarificationState) -> str:
    """Conditional Edge: Routes to ask_question or END."""
    completeness = state.get("completeness")
    if completeness and not completeness.is_complete:
        return "ask_question"
    return END

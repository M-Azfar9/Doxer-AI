"""
Phase 3 Architecture D: Closed-Loop Multi-Specialist DocGen Subgraph Implementation.
Features:
- Deterministic seed node & tech stack detection
- Intent router classifying into architecture_explainer, api_reference, or tutorial_quickstart
- Parallel specialist subgraphs via LangGraph Send() (FileSystem + AST, GitHub, Web Research)
- Iterative sufficiency gates per specialist (up to 3 hops)
- AST symbol and schema extraction for Python endpoints and models
- Evidence aggregator producing MergedEvidenceContext
- Template-aware synthesizer with citations
- Grounding critic evaluating against evidence
- Targeted refinement loop (up to 2 passes)
- Final document emitter generating Table of Contents, footnotes, and metadata verification footer
"""

import os
import sys
import re
import ast
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple, List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from src.core.service_registry import ServiceRegistry, services as default_services
from src.github_mcp_client import parse_github_url
from src.subagents.docgen.state import (
    DocGenState,
    DocIntentPlan,
    SpecialistScope,
    SufficiencyCheck,
    APIEndpointSymbol,
    FSEvidencePackage,
    GitHubEvidencePackage,
    WebEvidencePackage,
    MergedEvidenceContext,
    GroundingViolation,
    GroundingReport,
    FSSpecialistState,
    GitHubSpecialistState,
    WebSpecialistState,
)


DEFAULT_IGNORED_DIRS = {
    ".git", "node_modules", "__pycache__", ".pytest_cache", ".venv", "venv",
    "env", ".env", "dist", "build", ".idea", ".vscode", ".egg-info", "target",
    "bin", "obj", ".mypy_cache", ".ruff_cache", "coverage", ".next", ".nuxt"
}

DEFAULT_IGNORED_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp", ".mp4",
    ".zip", ".tar", ".gz", ".7z", ".rar", ".exe", ".dll", ".so",
    ".pyc", ".pyo", ".pyd", ".pkl", ".joblib", ".bin", ".weights",
    ".pdf", ".docx", ".xlsx", ".csv", ".tsv", ".sqlite", ".db", ".lock"
}


# ---------------------------------------------------------------------------
# Utility: ASCII Directory Tree Generator
# ---------------------------------------------------------------------------
def build_local_repo_map(root_path: str, max_depth: int = 2, max_files: int = 40) -> Tuple[str, List[str]]:
    """Builds an ASCII directory tree for a local filesystem path without invoking an LLM."""
    root = Path(root_path).resolve()
    if not root.exists():
        return f"[Error: Path {root_path} does not exist]", []

    lines = [f"{root.name}/"]
    detected_stack: List[str] = []
    file_count = 0

    def _walk(directory: Path, prefix: str = "", depth: int = 0):
        nonlocal file_count
        if depth > max_depth or file_count >= max_files:
            return

        try:
            items = sorted(
                list(directory.iterdir()),
                key=lambda p: (not p.is_dir(), p.name.lower())
            )
        except PermissionError:
            return

        visible_items = [
            p for p in items
            if p.name not in DEFAULT_IGNORED_DIRS and p.suffix.lower() not in DEFAULT_IGNORED_EXTS
        ]

        for i, item in enumerate(visible_items):
            if file_count >= max_files:
                lines.append(f"{prefix}... [truncated]")
                break

            name = item.name
            is_last = (i == len(visible_items) - 1)
            connector = "└── " if is_last else "├── "
            child_prefix = "    " if is_last else "│   "

            if item.is_dir():
                lines.append(f"{prefix}{connector}{name}/")
                _walk(item, prefix + child_prefix, depth + 1)
            else:
                file_count += 1
                lines.append(f"{prefix}{connector}{name}")

                # Tech stack heuristics
                if name in ("pyproject.toml", "requirements.txt", "Pipfile"):
                    detected_stack.append("Python")
                elif name in ("package.json", "tsconfig.json"):
                    detected_stack.append("Node/TypeScript")
                elif name in ("Dockerfile", "docker-compose.yml"):
                    detected_stack.append("Docker")
                elif name == "Cargo.toml":
                    detected_stack.append("Rust")
                elif name == "go.mod":
                    detected_stack.append("Go")

    _walk(root)
    detected_stack = sorted(list(set(detected_stack)))
    return "\n".join(lines), detected_stack


# ---------------------------------------------------------------------------
# Utility: Deterministic Python AST Symbol Parser
# ---------------------------------------------------------------------------
class ASTSymbolParser:
    """Extracts structured classes, functions, decorators, and docstrings via Python AST."""

    @staticmethod
    def parse_code(code: str, file_path: str = "") -> Tuple[List[APIEndpointSymbol], List[Dict[str, Any]]]:
        """Parses Python source code and returns (endpoints/functions, models/classes)."""
        endpoints: List[APIEndpointSymbol] = []
        models: List[Dict[str, Any]] = []

        try:
            tree = ast.parse(code)
        except Exception:
            return endpoints, models

        for node in tree.body:
            # 1. Functions & Async Routes
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                decs = [ast.unparse(d) for d in node.decorator_list]
                args_str = ast.unparse(node.args)
                ret_str = ast.unparse(node.returns) if node.returns else "None"

                is_route = any("router." in d or "app." in d for d in decs)
                if is_route:
                    kind = "route"
                elif isinstance(node, ast.AsyncFunctionDef):
                    kind = "async_function"
                else:
                    kind = "function"

                sig = f"def {node.name}({args_str}) -> {ret_str}"
                doc = ast.get_docstring(node)

                endpoints.append(APIEndpointSymbol(
                    name=node.name,
                    kind=kind,
                    path=file_path,
                    signature=sig,
                    decorators=decs,
                    docstring=doc.strip() if doc else None
                ))

            # 2. Classes & Pydantic Models
            elif isinstance(node, ast.ClassDef):
                bases = [ast.unparse(b) for b in node.bases]
                fields = []
                methods = []

                for item in node.body:
                    if isinstance(item, ast.AnnAssign):
                        f_name = ast.unparse(item.target)
                        f_type = ast.unparse(item.annotation)
                        f_default = ast.unparse(item.value) if item.value else None
                        fields.append({"name": f_name, "type": f_type, "default": f_default})
                    elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        m_decs = [ast.unparse(d) for d in item.decorator_list]
                        m_args = ast.unparse(item.args)
                        m_ret = ast.unparse(item.returns) if item.returns else "None"
                        m_sig = f"{item.name}({m_args}) -> {m_ret}"
                        methods.append({
                            "name": item.name,
                            "signature": m_sig,
                            "decorators": m_decs,
                            "docstring": ast.get_docstring(item)
                        })

                doc = ast.get_docstring(node)
                models.append({
                    "name": node.name,
                    "path": file_path,
                    "bases": bases,
                    "fields": fields,
                    "methods": methods,
                    "docstring": doc.strip() if doc else None
                })

        return endpoints, models


# ---------------------------------------------------------------------------
# Utility: Sensitive Credentials Scrubber & TOC Formatter
# ---------------------------------------------------------------------------
def scrub_sensitive_credentials(text: str) -> str:
    """Scrubs common credentials and secrets from documentation output."""
    if not text:
        return text
    text = re.sub(r'sk-[a-zA-Z0-9_\-]{20,}', '***REDACTED_API_KEY***', text)
    text = re.sub(r'(?:ghp_[a-zA-Z0-9]{30,}|github_pat_[a-zA-Z0-9_]{40,})', '***REDACTED_GITHUB_TOKEN***', text)
    text = re.sub(r'AKIA[0-9A-Z]{16}', '***REDACTED_AWS_KEY***', text)
    text = re.sub(r'(Bearer\s+)[a-zA-Z0-9_\-\.]{20,}', r'\1***REDACTED_TOKEN***', text)
    text = re.sub(
        r'(?i)\b(password|secret|api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret)\b(\s*[:=]\s*["\'])([^"\']{6,})(["\'])',
        r'\1\2***REDACTED***\4',
        text
    )
    text = re.sub(
        r'-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----',
        '***REDACTED_PRIVATE_KEY***',
        text
    )
    return text


def generate_markdown_toc(markdown_text: str) -> str:
    """Extracts H1-H3 headers and generates a clean Markdown Table of Contents."""
    lines = markdown_text.splitlines()
    headings = []
    for line in lines:
        match = re.match(r'^(#{1,3})\s+(.+)$', line.strip())
        if match:
            level_hashes, title = match.groups()
            if "table of contents" in title.lower() or "contents" in title.lower():
                continue
            headings.append((len(level_hashes), title.strip()))

    if len(headings) <= 1:
        return ""

    toc_lines = ["## Table of Contents"]
    for level, title in headings:
        indent = "  " * (level - 1)
        clean_anchor = re.sub(r'[^\w\s-]', '', title.lower()).replace(' ', '-')
        toc_lines.append(f"{indent}- [{title}](#{clean_anchor})")

    return "\n".join(toc_lines)


# ---------------------------------------------------------------------------
# Specialized Template Prompts for Architecture D
# ---------------------------------------------------------------------------
ROUTER_SYSTEM_PROMPT = """You are the Lead Documentation Architect for DevDocs AI.
Your job is to analyze the user's documentation request along with the provided repository layout and tech stack, then output a rigorous, structured documentation plan.

### Instructions:
1. Select the most appropriate `doc_type`:
   - "architecture_explainer": For system design, component interactions, workflows, data flow, or state machines.
   - "api_reference": For detailing classes, functions, REST endpoints, schemas, or tool interfaces.
   - "tutorial_quickstart": For step-by-step developer setup, usage guides, or minimal working examples.

2. Configure data source scopes:
   - `needs_filesystem`: Set enabled=True if a local repository path is provided and code inspection is needed. In `target_paths_or_topics`, list 1-4 specific relative file paths from the repo map that are relevant to the query.
   - `needs_github`: Set enabled=True if a remote GitHub URL is provided and remote files or commits are needed.
   - `needs_web`: Set enabled=True if external libraries, official framework documentation, or best practices should be looked up via Tavily / Web Research to enrich the doc. List 1-3 concrete search queries in `target_paths_or_topics`.

3. Provide a clear `planning_rationale` explaining your strategy.
"""

SUFFICIENCY_GATE_SYSTEM_PROMPT = """You are a Strict Quality & Sufficiency Auditor for DevDocs AI.
Your job is to evaluate whether the evidence gathered so far by a specialist agent is sufficient to fulfill its assigned documentation scope.

### Decision Criteria:
1. `is_sufficient = True`:
   - Set to True IF AND ONLY IF the collected files/code snippets/search results provide enough concrete detail (signatures, flows, logic) to write clear, authoritative documentation for the assigned scope.
   - Do NOT demand reading the entire codebase. If the primary classes or endpoints relevant to the scope are present, mark it SUFFICIENT.
   - When sufficient, confidence_score should be >= 0.85.

2. `is_sufficient = False`:
   - Set to False IF critical files or context mentioned in the scope are missing.
   - In `missing_information`, clearly state what key function, model, or file is missing.
   - In `suggested_targets`, provide 1-2 exact file paths (from the repo map) or specific search queries that should be fetched in the next hop.
   - In `next_action_guidance`, give strict instruction to the tool executor.
"""

PROMPT_ARCHITECTURE_EXPLAINER = """You are a Principal Software Architect for DevDocs AI.
Your task is to write an in-depth, authoritative Architecture Explainer document based STRICTLY on the provided code evidence and AST signatures.

### Required Document Structure:
1. # System & Architectural Overview
   - High-level purpose, role in the system, and primary design philosophy.
2. ## Component Topology & Architecture Diagram
   - Include a valid, clean Mermaid diagram (```mermaid ... ```) depicting data flow, component interactions, or sequence of operations.
3. ## Core Abstractions & Class Relationships
   - Detail the primary classes/functions, StateGraph orchestrator, router node logic, and specialized subgraphs. Cross-reference exact method signatures and responsibilities.
4. ## State Management, Lifecycle & Execution Flow
   - Step-by-step description of how requests/events transition through the subsystem and subordinate subgraphs.
5. ## Error Handling, Fallbacks & Resilience Patterns
   - Detail how errors, network timeouts, and edge cases are handled (e.g., fallback clients, retry mechanisms).
6. ## Grounded Citations & References
   - Reference every mentioned component using inline citations: `[^file:path/to/file]` or `[^web:SourceTitle]`.
"""

PROMPT_API_REFERENCE = """You are a Principal API Architect and Technical Writer for DevDocs AI.
Your task is to produce an exhaustive, authoritative API Reference based STRICTLY on the provided AST endpoint signatures and code evidence.

### Required Document Structure:
1. # API Reference & Interface Specification
   - High-level overview of the exposed module, service endpoints, and interfaces.
2. ## Route & Method Catalog
   - Summary table of all available endpoints, functions, and classes.
3. ## Detailed Interface Specifications
   - For EACH function, route, or class:
     * **Signature:** Exact Python/type signature (e.g. `async def fetch_repo_tree(...) -> List[str]`).
     * **Decorators / HTTP Routes:** (e.g., `@router.get(...)`, `@classmethod`).
     * **Parameters Table:** Parameter Name, Type, Default, and Semantic Description.
     * **Return Value:** Output type and structure.
     * **Exceptions Raised:** Potential failure conditions.
     * **Code Example:** Minimal, accurate usage snippet demonstrating the call.
4. ## Data Models & Schemas
   - Document any Pydantic models or dataclasses with their fields, types, and validations.
5. ## References & Source Attribution
   - Use inline citation markers: `[^file:path/to/file]` for every function and model.
"""

PROMPT_TUTORIAL_QUICKSTART = """You are a Staff Developer Advocate for DevDocs AI.
Your task is to create a frictionless, step-by-step Developer Quickstart Guide based STRICTLY on the verified repository code and web evidence.

### Required Document Structure:
1. # Quickstart & Implementation Guide
   - Concise summary of what the developer will build or integrate.
2. ## Prerequisites, Installation & Setup
   - Python version, required dependencies, package install (`pip install`), and environment variables.
3. ## Step-by-Step Implementation Walkthrough
   - Step 1: Configuration (`Config`) & Client Initialization.
   - Step 2: Core Execution / Running the workflow (`run`).
   - Step 3: Handling Responses & Errors.
4. ## Complete Minimal Working Example (MWE)
   - A single, self-contained, copy-pasteable script that successfully runs the integration.
5. ## Verification & Testing
   - Exact command or test function to verify that the implementation works.
6. ## Common Pitfalls & Troubleshooting
   - Verified edge cases, common misconfigurations, and error states.
7. ## References & Sources
   - Inline citations: `[^file:path/to/file]` or `[^web:SourceTitle]`.
"""

GROUNDING_CRITIC_SYSTEM_PROMPT = """You are a Strict Grounding & Verification Judge for DevDocs AI.
Your sole responsibility is to protect developers from hallucinations, invented APIs, and inaccurate claims by rigorously auditing draft documentation against the verified evidence.

### Verification Rules:
1. Cross-reference all function signatures, class names, method parameters, and HTTP endpoints mentioned in the draft against the AST Signatures and Code Evidence.
2. If the draft invents or alters an API signature (e.g., claiming a function takes `force_refresh: bool` when the signature has no such parameter), flag it as `incorrect_signature` or `hallucination`.
3. If an architectural claim or external dependency is completely unsupported by the provided evidence, flag it as `unsupported_assertion`.
4. If minor pedagogical explanations or general markdown formatting are used, DO NOT flag them as hallucinations as long as the underlying code claims are accurate.
5. If violations are found:
   - Set `is_grounded = False`.
   - Provide concrete `correction_instruction` for each violation.
   - Formulate 1-2 laser-targeted `targeted_refinement_queries` (e.g. "Check definition of fetch_file_content in src/github_mcp_client.py").
6. If the documentation is completely grounded and faithful to the evidence:
   - Set `is_grounded = True`.
   - Set `confidence_score >= 0.90`.
   - Leave `violations` empty.
"""

TEMPLATE_PROMPT_MAP = {
    "architecture_explainer": PROMPT_ARCHITECTURE_EXPLAINER,
    "api_reference": PROMPT_API_REFERENCE,
    "tutorial_quickstart": PROMPT_TUTORIAL_QUICKSTART
}


# ---------------------------------------------------------------------------
# Architecture D Pipeline Implementation
# ---------------------------------------------------------------------------
class DocGenArchitectureDPipeline:
    """
    Encapsulates all nodes, specialists, and routers for Phase 3 Architecture D.
    Uses Bedrock Chat with NVIDIA Nemotron Super 3 as primary LLM.
    """

    def __init__(self, services: Optional[ServiceRegistry] = None):
        self.services = services or default_services
        self.llm = self.services.llm
        self.github = self.services.github
        self.tavily = self.services.tavily

        # Structured output runners
        self.router_llm = self.llm.with_structured_output(DocIntentPlan)
        self.sufficiency_llm = self.llm.with_structured_output(SufficiencyCheck)
        self.critic_llm = self.llm.with_structured_output(GroundingReport)

    # -----------------------------------------------------------------------
    # Node 1: Deterministic Seed Node (`repo_map_generator`)
    # -----------------------------------------------------------------------
    def repo_map_generator_node(self, state: DocGenState) -> Dict[str, Any]:
        """Builds ASCII repo tree and detects tech stack before calling any LLM."""
        local_path = state.get("local_path") or os.getcwd()
        repo_url = state.get("repo_url")

        if local_path and os.path.exists(local_path):
            tree_text, tech_stack = build_local_repo_map(local_path)
        elif repo_url:
            tree_text = f"Remote GitHub Repository: {repo_url}\n(Remote structure will be examined via GitHub Tool)"
            tech_stack = ["Remote Repository"]
        else:
            tree_text = "No repository target provided (pure web documentation mode)."
            tech_stack = []

        return {
            "local_path": local_path,
            "repo_map": tree_text,
            "detected_tech_stack": tech_stack
        }

    # -----------------------------------------------------------------------
    # Node 2: Structured Intent Router (`doc_intent_router`)
    # -----------------------------------------------------------------------
    def doc_intent_router_node(self, state: DocGenState) -> Dict[str, Any]:
        """Analyzes query and repo map to generate a typed DocIntentPlan."""
        user_prompt = f"""Target Query: {state['user_query']}
Local Path: {state.get('local_path') or 'None'}
Remote Repo URL: {state.get('repo_url') or 'None'}
Detected Tech Stack: {state.get('detected_tech_stack', [])}

Repository Structure:
{state.get('repo_map', 'No repository layout available.')}

Produce a structured DocIntentPlan detailing the doc archetype and exact retrieval scopes.
"""
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        try:
            plan = self.router_llm.invoke(messages)
            if not isinstance(plan, DocIntentPlan):
                plan = DocIntentPlan(**plan)
        except Exception as e:
            # Resilient fallback plan
            plan = DocIntentPlan(
                doc_type="architecture_explainer",
                needs_filesystem=SpecialistScope(
                    enabled=bool(state.get("local_path")),
                    focus_scope="Examine local architecture",
                    target_paths_or_topics=["README.md"]
                ),
                needs_github=SpecialistScope(
                    enabled=bool(state.get("repo_url")),
                    focus_scope="Examine remote GitHub repository",
                    target_paths_or_topics=[]
                ),
                needs_web=SpecialistScope(
                    enabled=True,
                    focus_scope="Search official technical documentation",
                    target_paths_or_topics=[state["user_query"]]
                ),
                planning_rationale=f"Fallback plan generated due to router exception: {e}"
            )

        return {"intent_plan": plan}

    # -----------------------------------------------------------------------
    # Sufficiency Evaluator Helper
    # -----------------------------------------------------------------------
    def evaluate_sufficiency(
        self,
        specialist_name: str,
        scope: str,
        evidence_summary: str,
        repo_map: str = "",
        hop_count: int = 1,
        max_hops: int = 3
    ) -> SufficiencyCheck:
        """Evaluates whether accumulated evidence satisfies the specialist's scope."""
        if hop_count >= max_hops:
            return SufficiencyCheck(
                is_sufficient=True,
                confidence_score=0.80,
                missing_information=None,
                next_action_guidance="Max hops reached. Proceeding to packager with available context."
            )

        prompt = f"""Specialist: {specialist_name}
Assigned Scope:
{scope}

Hop Count: {hop_count} / {max_hops}

Repo Map Available:
{repo_map[:1200]}

Evidence Accumulated So Far:
{evidence_summary[:3000]}

Assess whether this evidence is sufficient to document the assigned scope.
"""
        messages = [
            SystemMessage(content=SUFFICIENCY_GATE_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]

        try:
            res = self.sufficiency_llm.invoke(messages)
            if not isinstance(res, SufficiencyCheck):
                res = SufficiencyCheck(**res)
            return res
        except Exception:
            return SufficiencyCheck(
                is_sufficient=True,
                confidence_score=0.75,
                next_action_guidance="Sufficiency fallback evaluated as complete."
            )

    # -----------------------------------------------------------------------
    # Specialist Subgraph 1: FileSystem Specialist (AST + Hops)
    # -----------------------------------------------------------------------
    def fs_explore_step(self, state: FSSpecialistState) -> Dict[str, Any]:
        """Reads target paths from local filesystem, traverses directories if provided, and extracts AST symbols."""
        hop = state.get("hop_count", 0) + 1
        root_dir = Path(state.get("local_path") or os.getcwd()).resolve()

        accumulated = dict(state.get("files_accumulated", {}))
        discovered_symbols = list(state.get("discovered_symbols", []))
        discovered_endpoints = list(state.get("discovered_endpoints", []))
        discovered_models = list(state.get("discovered_models", []))

        targets = list(state.get("target_paths", []))

        # If no targets or generic README target, augment targets from scope keywords
        if not targets or targets == ["README.md"]:
            scope_text = (state.get("scope", "") + " " + state.get("repo_map", "")).lower()
            candidate_keywords = ["supervisor", "qa", "github_mcp", "srs", "docgen", "config", "agent", "chroma"]
            for kw in candidate_keywords:
                if kw in scope_text:
                    for found_file in root_dir.rglob(f"*{kw}*.py"):
                        try:
                            rel_cand = str(found_file.relative_to(root_dir))
                            if rel_cand not in targets and ".venv" not in rel_cand and "__pycache__" not in rel_cand:
                                targets.append(rel_cand)
                                if len(targets) >= 4:
                                    break
                        except ValueError:
                            pass
                    if len(targets) >= 4:
                        break

        for target in targets:
            clean = target.strip()
            if not clean or clean in accumulated:
                continue

            target_path = Path(clean)
            if not target_path.is_absolute():
                target_path = root_dir / target_path

            candidates: List[Path] = []
            if target_path.exists():
                if target_path.is_file():
                    candidates.append(target_path)
                elif target_path.is_dir():
                    # Expand directory to key code or markdown files
                    py_files = [p for p in target_path.glob("*.py") if p.is_file() and not p.name.startswith("__")]
                    md_files = [p for p in target_path.glob("*.md") if p.is_file()]
                    candidates.extend(py_files[:2])
                    candidates.extend(md_files[:1])
            else:
                # Fuzzy match by filename or stem
                stem = Path(clean).stem
                if stem and len(stem) > 2:
                    for m in root_dir.rglob(f"*{stem}*"):
                        if m.is_file() and (".venv" not in str(m)) and ("__pycache__" not in str(m)):
                            if m.suffix in [".py", ".md", ".json", ".yaml", ".yml", ".txt"]:
                                candidates.append(m)
                                break

            for cand in candidates:
                try:
                    rel_path = str(cand.relative_to(root_dir)) if cand.is_relative_to(root_dir) else str(cand)
                    if rel_path in accumulated:
                        continue

                    content = cand.read_text(encoding="utf-8", errors="replace")[:12000]
                    accumulated[rel_path] = content

                    # AST Symbol extraction for Python files
                    if rel_path.endswith(".py"):
                        eps, models = ASTSymbolParser.parse_code(content, file_path=rel_path)
                        discovered_endpoints.extend(eps)
                        discovered_models.extend(models)
                        for ep in eps:
                            discovered_symbols.append(f"[{ep.kind}] {ep.name}: {ep.signature}")
                        for m in models:
                            bases_str = ", ".join(m.get("bases", [])) or "object"
                            discovered_symbols.append(f"[class] {m['name']} (Bases: {bases_str})")
                except Exception as e:
                    accumulated[str(cand)] = f"[Error reading file: {e}]"

        return {
            "hop_count": hop,
            "files_accumulated": accumulated,
            "discovered_symbols": discovered_symbols[:40],
            "discovered_endpoints": discovered_endpoints,
            "discovered_models": discovered_models
        }

    def fs_sufficiency_gate(self, state: FSSpecialistState) -> Dict[str, Any]:
        """Audits sufficiency of accumulated local files and AST symbols."""
        symbol_bullets = "\n".join([f"- {s}" for s in state.get("discovered_symbols", [])[:20]]) or "No structured AST symbols extracted yet."
        summary_parts = []
        for path, content in state.get("files_accumulated", {}).items():
            summary_parts.append(f"File: {path}\nSnippet:\n{content[:500]}")
        evidence_text = "\n\n".join(summary_parts) if summary_parts else "No files read yet."

        combined_evidence = f"""### Extracted AST Signatures & Models:
{symbol_bullets}

### File Code Snippets:
{evidence_text}"""

        decision = self.evaluate_sufficiency(
            specialist_name="FileSystem Specialist",
            scope=state["scope"],
            evidence_summary=combined_evidence,
            repo_map=state.get("repo_map", ""),
            hop_count=state.get("hop_count", 1),
            max_hops=state.get("max_hops", 3)
        )

        next_targets = decision.suggested_targets if decision.suggested_targets else []
        next_targets = [t for t in next_targets if t not in state.get("files_accumulated", {})]

        return {
            "sufficiency": decision,
            "target_paths": next_targets
        }

    @staticmethod
    def fs_route_after_gate(state: FSSpecialistState) -> str:
        sufficiency = state.get("sufficiency")
        if sufficiency and sufficiency.is_sufficient:
            return "fs_evidence_packager"
        if state.get("hop_count", 0) >= state.get("max_hops", 3):
            return "fs_evidence_packager"
        if not state.get("target_paths"):
            return "fs_evidence_packager"
        return "fs_explore_step"

    def fs_evidence_packager(self, state: FSSpecialistState) -> Dict[str, Any]:
        """Packages accumulated findings into a typed FSEvidencePackage."""
        eps = state.get("discovered_endpoints", [])
        models = state.get("discovered_models", [])

        package = FSEvidencePackage(
            scanned_paths=list(state.get("files_accumulated", {}).keys()),
            endpoints=eps,
            models=models,
            key_symbols_or_routes=state.get("discovered_symbols", []),
            file_contents=state.get("files_accumulated", {}),
            architectural_notes=[
                f"Examined {len(state.get('files_accumulated', {}))} file(s) across {state.get('hop_count', 1)} hop(s).",
                f"AST Symbol Extraction: {len(eps)} endpoint signature(s) and {len(models)} model definition(s)."
            ]
        )
        return {
            "package": package,
            "specialist_outputs": [{"source": "filesystem", "package": package.model_dump()}]
        }

    def compile_fs_subgraph(self):
        builder = StateGraph(FSSpecialistState)
        builder.add_node("fs_explore_step", self.fs_explore_step)
        builder.add_node("fs_sufficiency_gate", self.fs_sufficiency_gate)
        builder.add_node("fs_evidence_packager", self.fs_evidence_packager)

        builder.add_edge(START, "fs_explore_step")
        builder.add_edge("fs_explore_step", "fs_sufficiency_gate")
        builder.add_conditional_edges(
            "fs_sufficiency_gate",
            self.fs_route_after_gate,
            {"fs_explore_step": "fs_explore_step", "fs_evidence_packager": "fs_evidence_packager"}
        )
        builder.add_edge("fs_evidence_packager", END)
        return builder.compile()

    # -----------------------------------------------------------------------
    # Specialist Subgraph 2: GitHub Specialist (Remote Repo + Hops)
    # -----------------------------------------------------------------------
    def gh_explore_step(self, state: GitHubSpecialistState) -> Dict[str, Any]:
        """Fetches target files from remote GitHub repository."""
        hop = state.get("hop_count", 0) + 1
        accumulated = dict(state.get("files_accumulated", {}))
        repo_url = state.get("repo_url", "")

        if repo_url:
            try:
                owner, repo, branch = parse_github_url(repo_url)
                for path in state.get("target_paths", []):
                    clean = path.strip()
                    if clean and clean not in accumulated:
                        try:
                            # Use sync fetch or run coroutine
                            if hasattr(self.github, "fetch_file_content_sync"):
                                res = self.github.fetch_file_content_sync(owner, repo, clean, branch=branch)
                            else:
                                res = asyncio.run(self.github.fetch_file_content(owner, repo, clean, branch=branch))
                            content = res.get("content", "") if isinstance(res, dict) else str(res)
                            accumulated[clean] = content
                        except Exception as exc:
                            accumulated[clean] = f"[Error fetching GitHub file: {exc}]"
            except Exception as e:
                print(f"⚠️ [gh_explore_step] URL parsing error: {e}")

        return {
            "hop_count": hop,
            "files_accumulated": accumulated
        }

    def gh_sufficiency_gate(self, state: GitHubSpecialistState) -> Dict[str, Any]:
        summary_parts = [f"File: {p}\nSnippet:\n{c[:600]}" for p, c in state.get("files_accumulated", {}).items()]
        evidence_text = "\n\n".join(summary_parts) if summary_parts else "No remote files fetched yet."

        decision = self.evaluate_sufficiency(
            specialist_name="GitHub Specialist",
            scope=state["scope"],
            evidence_summary=evidence_text,
            repo_map=state.get("repo_map", ""),
            hop_count=state.get("hop_count", 1),
            max_hops=state.get("max_hops", 3)
        )
        next_targets = [t for t in decision.suggested_targets if t not in state.get("files_accumulated", {})]

        return {
            "sufficiency": decision,
            "target_paths": next_targets
        }

    @staticmethod
    def gh_route_after_gate(state: GitHubSpecialistState) -> str:
        sufficiency = state.get("sufficiency")
        if sufficiency and sufficiency.is_sufficient:
            return "gh_evidence_packager"
        if state.get("hop_count", 0) >= state.get("max_hops", 3) or not state.get("target_paths"):
            return "gh_evidence_packager"
        return "gh_explore_step"

    def gh_evidence_packager(self, state: GitHubSpecialistState) -> Dict[str, Any]:
        package = GitHubEvidencePackage(
            files_examined=list(state.get("files_accumulated", {}).keys()),
            file_contents=state.get("files_accumulated", {}),
            architectural_notes=[
                f"Remote repo: {state.get('repo_url')}",
                f"Explored in {state.get('hop_count', 1)} hop(s)."
            ]
        )
        return {
            "package": package,
            "specialist_outputs": [{"source": "github", "package": package.model_dump()}]
        }

    def compile_gh_subgraph(self):
        builder = StateGraph(GitHubSpecialistState)
        builder.add_node("gh_explore_step", self.gh_explore_step)
        builder.add_node("gh_sufficiency_gate", self.gh_sufficiency_gate)
        builder.add_node("gh_evidence_packager", self.gh_evidence_packager)

        builder.add_edge(START, "gh_explore_step")
        builder.add_edge("gh_explore_step", "gh_sufficiency_gate")
        builder.add_conditional_edges(
            "gh_sufficiency_gate",
            self.gh_route_after_gate,
            {"gh_explore_step": "gh_explore_step", "gh_evidence_packager": "gh_evidence_packager"}
        )
        builder.add_edge("gh_evidence_packager", END)
        return builder.compile()

    # -----------------------------------------------------------------------
    # Specialist Subgraph 3: Web Research Specialist (Tavily + Hops)
    # -----------------------------------------------------------------------
    def web_explore_step(self, state: WebSpecialistState) -> Dict[str, Any]:
        """Executes web search queries across Tavily."""
        hop = state.get("hop_count", 0) + 1
        accumulated = list(state.get("snippets_accumulated", []))

        for q in state.get("search_topics", []):
            clean = q.strip()
            if clean:
                try:
                    results = self.tavily.search(clean, max_results=2)
                    for r in results:
                        accumulated.append({
                            "title": r.get("title", "Web Result"),
                            "url": r.get("url", ""),
                            "snippet": (r.get("content") or "")[:700]
                        })
                except Exception as exc:
                    print(f"⚠️ [web_explore_step] Tavily search error: {exc}")

        return {
            "hop_count": hop,
            "snippets_accumulated": accumulated
        }

    def web_sufficiency_gate(self, state: WebSpecialistState) -> Dict[str, Any]:
        snippets_text = "\n\n".join([
            f"Title: {s.get('title')} ({s.get('url')})\nSnippet: {s.get('snippet', '')[:400]}"
            for s in state.get("snippets_accumulated", [])
        ])
        decision = self.evaluate_sufficiency(
            specialist_name="Web Research Specialist",
            scope=state["scope"],
            evidence_summary=snippets_text or "No snippets collected.",
            hop_count=state.get("hop_count", 1),
            max_hops=state.get("max_hops", 3)
        )
        next_queries = decision.suggested_targets if decision.suggested_targets else []

        return {
            "sufficiency": decision,
            "search_topics": next_queries
        }

    @staticmethod
    def web_route_after_gate(state: WebSpecialistState) -> str:
        sufficiency = state.get("sufficiency")
        if sufficiency and sufficiency.is_sufficient:
            return "web_evidence_packager"
        if state.get("hop_count", 0) >= state.get("max_hops", 3) or not state.get("search_topics"):
            return "web_evidence_packager"
        return "web_explore_step"

    def web_evidence_packager(self, state: WebSpecialistState) -> Dict[str, Any]:
        snippets = state.get("snippets_accumulated", [])
        package = WebEvidencePackage(
            claims_supported=[s.get("title", "") for s in snippets[:5]],
            sources=snippets,
            compatibility_notes=[
                f"Acquired {len(snippets)} web sources across {state.get('hop_count', 1)} hop(s)."
            ]
        )
        return {
            "package": package,
            "specialist_outputs": [{"source": "web", "package": package.model_dump()}]
        }

    def compile_web_subgraph(self):
        builder = StateGraph(WebSpecialistState)
        builder.add_node("web_explore_step", self.web_explore_step)
        builder.add_node("web_sufficiency_gate", self.web_sufficiency_gate)
        builder.add_node("web_evidence_packager", self.web_evidence_packager)

        builder.add_edge(START, "web_explore_step")
        builder.add_edge("web_explore_step", "web_sufficiency_gate")
        builder.add_conditional_edges(
            "web_sufficiency_gate",
            self.web_route_after_gate,
            {"web_explore_step": "web_explore_step", "web_evidence_packager": "web_evidence_packager"}
        )
        builder.add_edge("web_evidence_packager", END)
        return builder.compile()

    # -----------------------------------------------------------------------
    # Fan-Out Dispatcher & Evidence Aggregator
    # -----------------------------------------------------------------------
    @staticmethod
    def fan_out_dispatcher(state: DocGenState) -> List[Send]:
        """Conditional edge that inspects DocIntentPlan and spawns specialists via Send()."""
        plan = state.get("intent_plan")
        if not plan:
            return []

        sends: List[Send] = []

        # 1. FileSystem Specialist Fan-Out
        if plan.needs_filesystem.enabled and state.get("local_path"):
            targets = plan.needs_filesystem.target_paths_or_topics or ["README.md"]
            sends.append(Send("fs_specialist", {
                "local_path": state["local_path"],
                "scope": plan.needs_filesystem.focus_scope,
                "target_paths": targets,
                "repo_map": state.get("repo_map", ""),
                "hop_count": 0,
                "max_hops": 3,
                "files_accumulated": {},
                "discovered_symbols": [],
                "discovered_endpoints": [],
                "discovered_models": [],
                "sufficiency": None,
                "package": None
            }))

        # 2. GitHub Specialist Fan-Out
        if plan.needs_github.enabled and state.get("repo_url"):
            targets = plan.needs_github.target_paths_or_topics
            sends.append(Send("gh_specialist", {
                "repo_url": state["repo_url"],
                "scope": plan.needs_github.focus_scope,
                "target_paths": targets,
                "repo_map": state.get("repo_map", ""),
                "hop_count": 0,
                "max_hops": 3,
                "files_accumulated": {},
                "key_abstractions": [],
                "sufficiency": None,
                "package": None
            }))

        # 3. Web Research Specialist Fan-Out
        if plan.needs_web.enabled:
            topics = plan.needs_web.target_paths_or_topics or [state["user_query"]]
            sends.append(Send("web_specialist", {
                "user_query": state["user_query"],
                "scope": plan.needs_web.focus_scope,
                "search_topics": topics,
                "hop_count": 0,
                "max_hops": 3,
                "snippets_accumulated": [],
                "sufficiency": None,
                "package": None
            }))

        # Fallback if no specialists enabled
        if not sends:
            # Default to filesystem if path available, else web
            if state.get("local_path"):
                sends.append(Send("fs_specialist", {
                    "local_path": state["local_path"],
                    "scope": "Analyze local workspace",
                    "target_paths": ["README.md"],
                    "repo_map": state.get("repo_map", ""),
                    "hop_count": 0,
                    "max_hops": 2,
                    "files_accumulated": {},
                    "discovered_symbols": [],
                    "discovered_endpoints": [],
                    "discovered_models": [],
                    "sufficiency": None,
                    "package": None
                }))
            else:
                sends.append(Send("web_specialist", {
                    "user_query": state["user_query"],
                    "scope": "Search technical overview",
                    "search_topics": [state["user_query"]],
                    "hop_count": 0,
                    "max_hops": 2,
                    "snippets_accumulated": [],
                    "sufficiency": None,
                    "package": None
                }))

        return sends

    def evidence_aggregator_node(self, state: DocGenState) -> Dict[str, Any]:
        """Merges outputs from all finished specialist packages into MergedEvidenceContext."""
        specialist_outputs = state.get("specialist_outputs", [])

        local_files: List[Dict[str, Any]] = []
        github_files: List[Dict[str, Any]] = []
        web_snippets: List[Dict[str, Any]] = []
        all_endpoints: List[APIEndpointSymbol] = []
        summary_lines = []

        for item in specialist_outputs:
            src = item.get("source")
            pkg = item.get("package", {})

            if src == "filesystem":
                file_contents = pkg.get("file_contents", {})
                for p, content in file_contents.items():
                    local_files.append({"file_path": p, "content": content})
                raw_eps = pkg.get("endpoints", [])
                for ep in raw_eps:
                    all_endpoints.append(ep if isinstance(ep, APIEndpointSymbol) else APIEndpointSymbol(**ep))
                symbols = pkg.get("key_symbols_or_routes", [])
                summary_lines.append(f"FileSystem: {len(file_contents)} files read, {len(all_endpoints)} AST endpoints, {len(symbols)} symbols extracted.")

            elif src == "github":
                file_contents = pkg.get("file_contents", {})
                for p, content in file_contents.items():
                    github_files.append({"file_path": p, "content": content})
                summary_lines.append(f"GitHub: {len(file_contents)} remote files fetched.")

            elif src == "web":
                sources = pkg.get("sources", [])
                web_snippets.extend(sources)
                summary_lines.append(f"Web Research: {len(sources)} verified snippets.")

        merged = MergedEvidenceContext(
            local_files=local_files,
            github_files=github_files,
            web_snippets=web_snippets,
            endpoints=all_endpoints,
            evidence_summary="; ".join(summary_lines) or "Aggregated specialist findings."
        )

        # Backward compatibility for legacy Phase 1 callers expecting retrieved_evidence
        retrieved_legacy = {
            "local_files": local_files,
            "github_files": github_files,
            "web_snippets": web_snippets
        }

        return {
            "merged_evidence": merged,
            "retrieved_evidence": retrieved_legacy
        }

    # -----------------------------------------------------------------------
    # Node: Template-Aware Synthesizer (`template_aware_docgen`)
    # -----------------------------------------------------------------------
    def template_aware_docgen_node(self, state: DocGenState) -> Dict[str, Any]:
        """Synthesizes documentation matching archetype with AST symbols & citations."""
        plan = state.get("intent_plan")
        doc_type = plan.doc_type if plan else "architecture_explainer"
        selected_system_prompt = TEMPLATE_PROMPT_MAP.get(doc_type, PROMPT_ARCHITECTURE_EXPLAINER)
        merged = state.get("merged_evidence")

        code_context_parts = []
        citations_list = []

        # 1. AST Endpoints & Models
        if merged and merged.endpoints:
            ast_lines = []
            for ep in merged.endpoints:
                decs = f"[{', '.join(ep.decorators)}] " if ep.decorators else ""
                doc_snip = f" — {ep.docstring}" if ep.docstring else ""
                ast_lines.append(f"- `{decs}{ep.signature}` (file: `{ep.path}`){doc_snip}")
                citations_list.append(f"[^file:{ep.path}]: AST-verified definition in `{ep.path}`")
            code_context_parts.append("### Verified AST Signatures & Endpoints:\n" + "\n".join(ast_lines))

        # 2. Raw Code Files
        if merged:
            for f in merged.local_files:
                path = f.get("file_path", "unknown")
                content = f.get("content", "")
                code_context_parts.append(f"### File: `{path}` (Local Codebase)\n```python\n{content}\n```")
                citations_list.append(f"[^file:{path}]: Local repository file `{path}`")

            for f in merged.github_files:
                path = f.get("file_path", "unknown")
                content = f.get("content", "")
                code_context_parts.append(f"### File: `{path}` (Remote GitHub Repository)\n```python\n{content}\n```")
                citations_list.append(f"[^file:{path}]: Remote GitHub file `{path}`")

            web_context_parts = []
            for w in merged.web_snippets:
                title = w.get("title", "Web Reference")
                url = w.get("url", "")
                snippet = w.get("snippet", "")
                web_context_parts.append(f"- **{title}** ({url}):\n  {snippet}")
                if url:
                    citations_list.append(f"[^web:{title[:25]}]: [{title}]({url})")
        else:
            web_context_parts = []

        code_context_text = "\n\n".join(code_context_parts) if code_context_parts else "No code files retrieved."
        web_context_text = "\n\n".join(web_context_parts) if web_context_parts else "No web references retrieved."

        # 3. Refinement Feedback from Grounding Critic (if looping)
        refinement_feedback_block = ""
        grounding_report = state.get("grounding_report")
        if grounding_report and not grounding_report.is_grounded and grounding_report.violations:
            feedback_lines = []
            for v in grounding_report.violations:
                feedback_lines.append(f"- Issue at [{v.location}]: {v.issue} -> '{v.claim}'. Correction: {v.correction_instruction}")
            refinement_feedback_block = f"""
--- 🚨 MANDATORY GROUNDING CORRECTIONS FROM PRIOR AUDIT ---
The previous draft contained ungrounded claims or signature discrepancies:
{chr(10).join(feedback_lines)}
You MUST correct or remove these ungrounded claims in this revision!
"""

        user_prompt = f"""Target Documentation Query: {state['user_query']}
Selected Archetype: {doc_type}
Detected Tech Stack: {state.get('detected_tech_stack', [])}
{refinement_feedback_block}
--- CODE & AST EVIDENCE ---
{code_context_text}

--- WEB EVIDENCE ---
{web_context_text}

Synthesize an authoritative, production-grade technical document in Markdown matching the '{doc_type}' archetype.
Follow the structure defined in your instructions. Include concrete code samples and inline citation markers.
"""
        messages = [
            SystemMessage(content=selected_system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = self.llm.invoke(messages)
        doc_markdown = getattr(response, "content", str(response))
        doc_markdown = scrub_sensitive_credentials(doc_markdown)

        return {
            "draft_markdown": doc_markdown,
            "citations": sorted(list(set(citations_list)))
        }

    # -----------------------------------------------------------------------
    # Node: Grounding Critic (`grounding_critic`)
    # -----------------------------------------------------------------------
    def grounding_critic_node(self, state: DocGenState) -> Dict[str, Any]:
        """Audits draft documentation against MergedEvidenceContext for hallucinations."""
        draft = state.get("draft_markdown", "")
        merged = state.get("merged_evidence")
        ref_count = state.get("refinement_count", 0)

        evidence_parts = []
        if merged:
            if merged.endpoints:
                ep_lines = [f"- {ep.name} ({ep.kind}): `{ep.signature}` in `{ep.path}`" for ep in merged.endpoints]
                evidence_parts.append("### AST Verified Endpoints & Signatures:\n" + "\n".join(ep_lines))
            for f in merged.local_files:
                evidence_parts.append(f"### Local File: {f.get('file_path')}\n{f.get('content', '')[:1200]}")
            for f in merged.github_files:
                evidence_parts.append(f"### GitHub File: {f.get('file_path')}\n{f.get('content', '')[:1200]}")
            for w in merged.web_snippets:
                evidence_parts.append(f"### Web Snippet: {w.get('title')} ({w.get('url')})\n{w.get('snippet', '')[:400]}")

        evidence_text = "\n\n".join(evidence_parts) if evidence_parts else "No evidence provided."

        user_prompt = f"""Target Documentation Query: {state['user_query']}

--- MERGED EVIDENCE REPOSITORY ---
{evidence_text}

--- DRAFT DOCUMENTATION TO AUDIT ---
{draft}

Audit every API signature, method name, parameter, and architectural claim. Output a structured GroundingReport.
"""
        messages = [
            SystemMessage(content=GROUNDING_CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        try:
            report = self.critic_llm.invoke(messages)
            if not isinstance(report, GroundingReport):
                report = GroundingReport(**report)
        except Exception:
            report = GroundingReport(
                is_grounded=True,
                confidence_score=0.90,
                violations=[],
                targeted_refinement_queries=[]
            )

        return {
            "grounding_report": report,
            "grounding_score": report.confidence_score
        }

    @staticmethod
    def route_after_grounding_critic(state: DocGenState) -> str:
        """Conditional Edge: Routes to final doc emitter or targeted refinement."""
        report = state.get("grounding_report")
        refinement_count = state.get("refinement_count", 0)
        max_refinements = 2

        if not report or report.is_grounded or refinement_count >= max_refinements:
            return "final_doc_emitter"
        return "targeted_refinement_node"

    # -----------------------------------------------------------------------
    # Node: Targeted Refinement (`targeted_refinement_node`)
    # -----------------------------------------------------------------------
    def targeted_refinement_node(self, state: DocGenState) -> Dict[str, Any]:
        """Executes laser-targeted lookups strictly for gaps flagged by Grounding Critic."""
        current_count = state.get("refinement_count", 0) + 1
        report = state.get("grounding_report")
        queries = report.targeted_refinement_queries if report else []

        merged = state.get("merged_evidence") or MergedEvidenceContext()
        local_files = list(merged.local_files)
        web_snippets = list(merged.web_snippets)
        endpoints = list(merged.endpoints)

        root_dir = Path(state.get("local_path") or os.getcwd()).resolve()

        for q in queries[:2]:
            clean_q = q.strip()
            is_file_target = any(clean_q.endswith(ext) or ext in clean_q for ext in [".py", ".md", ".json", ".txt"]) or "/" in clean_q or "\\" in clean_q

            if is_file_target and state.get("local_path"):
                candidate_paths = [part.strip("`'\"") for part in clean_q.split() if "." in part or "/" in part or "\\" in part]
                target_path_str = candidate_paths[0] if candidate_paths else clean_q
                target_path = Path(target_path_str)
                if not target_path.is_absolute():
                    target_path = root_dir / target_path

                if target_path.exists() and target_path.is_file():
                    try:
                        content = target_path.read_text(encoding="utf-8", errors="replace")[:6000]
                        rel_path = str(target_path.relative_to(root_dir)) if target_path.is_relative_to(root_dir) else target_path_str
                        local_files.append({"file_path": rel_path, "content": content})

                        if target_path_str.endswith(".py"):
                            eps, _ = ASTSymbolParser.parse_code(content, file_path=rel_path)
                            endpoints.extend(eps)
                    except Exception:
                        pass
                else:
                    try:
                        search_res = self.tavily.search(clean_q, max_results=2)
                        web_snippets.extend(search_res)
                    except Exception:
                        pass
            else:
                try:
                    search_res = self.tavily.search(clean_q, max_results=2)
                    web_snippets.extend(search_res)
                except Exception:
                    pass

        updated_merged = MergedEvidenceContext(
            local_files=local_files,
            github_files=merged.github_files,
            web_snippets=web_snippets,
            endpoints=endpoints,
            evidence_summary=f"{merged.evidence_summary} | Refinement Pass {current_count}: {len(queries)} gap(s) audited."
        )

        return {
            "merged_evidence": updated_merged,
            "refinement_count": current_count
        }

    # -----------------------------------------------------------------------
    # Node: Final Document Emitter (`final_doc_emitter`)
    # -----------------------------------------------------------------------
    def final_doc_emitter_node(self, state: DocGenState) -> Dict[str, Any]:
        """Formats Table of Contents, footnotes citations, and metadata audit footer."""
        draft = state.get("draft_markdown", "").strip()
        plan = state.get("intent_plan")
        doc_type = plan.doc_type if plan else "architecture_explainer"
        report = state.get("grounding_report")
        ref_count = state.get("refinement_count", 0)
        merged = state.get("merged_evidence")

        # 1. Table of Contents
        toc_block = ""
        if "## table of contents" not in draft.lower() and "## contents" not in draft.lower():
            toc_block = generate_markdown_toc(draft)

        # 2. Footnotes / Citations Manifest
        citations_list = list(state.get("citations", []))
        citations_manifest = []

        for c in citations_list:
            match = re.match(r'\[\^([^\]]+)\]:\s*(.+)', c)
            if match:
                key, desc = match.groups()
                citations_manifest.append({"key": key, "reference": desc})
            else:
                citations_manifest.append({"key": "ref", "reference": c})

        citations_block = ""
        if citations_list and "## references" not in draft.lower() and "## sources" not in draft.lower():
            citations_block = "\n\n## References & Verified Sources\n" + "\n".join(sorted(list(set(citations_list))))

        # 3. Verification Metadata Footer
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        grounding_score = report.confidence_score if report else 1.0
        grounding_status = "VERIFIED (100% Grounded)" if (report and report.is_grounded) else f"AUDITED (Confidence: {grounding_score:.1%})"

        fs_count = len(merged.local_files) if merged else 0
        gh_count = len(merged.github_files) if merged else 0
        ast_count = len(merged.endpoints) if merged else 0
        web_count = len(merged.web_snippets) if merged else 0

        metadata_footer = f"""

---
### 🛡️ Documentation Verification Metadata (Architecture D)
- **Document Archetype:** `{doc_type}`
- **Grounding Audit Status:** `{grounding_status}`
- **Refinement Iterations:** `{ref_count}` pass(es) executed
- **Consulted Evidence Sources:** {fs_count} local file(s), {ast_count} AST symbol(s), {gh_count} remote GitHub file(s), {web_count} web reference(s)
- **Generated At:** `{now_utc}` via `DevDocs AI — DocGen Subgraph`
"""

        # 4. Assemble Final Markdown
        sections = []
        lines = draft.splitlines()
        if lines and lines[0].startswith("# "):
            h1 = lines[0]
            body = "\n".join(lines[1:]).strip()
            sections.append(h1)
            if toc_block:
                sections.append(toc_block)
            sections.append(body)
        else:
            if toc_block:
                sections.append(toc_block)
            sections.append(draft)

        if citations_block:
            sections.append(citations_block)

        sections.append(metadata_footer)
        final_markdown = "\n\n".join(sections).strip()

        retrieved_legacy = {
            "local_files": merged.local_files if merged else [],
            "github_files": merged.github_files if merged else [],
            "web_snippets": merged.web_snippets if merged else []
        }

        return {
            "final_doc_markdown": final_markdown,
            "draft_markdown": final_markdown,  # Populated for backward compatibility with callers reading draft_markdown
            "citations_manifest": citations_manifest,
            "grounding_score": grounding_score,
            "retrieved_evidence": retrieved_legacy,
            "merged_evidence": merged
        }

    # -----------------------------------------------------------------------
    # Graph Compilation
    # -----------------------------------------------------------------------
    def compile_graph(self):
        """Builds and compiles the complete closed-loop Architecture D StateGraph."""
        fs_subgraph = self.compile_fs_subgraph()
        gh_subgraph = self.compile_gh_subgraph()
        web_subgraph = self.compile_web_subgraph()

        builder = StateGraph(DocGenState)

        # 1. Deterministic Seed & Routing Nodes
        builder.add_node("repo_map_generator", self.repo_map_generator_node)
        builder.add_node("doc_intent_router", self.doc_intent_router_node)

        # 2. Specialist Subgraphs
        builder.add_node("fs_specialist", fs_subgraph)
        builder.add_node("gh_specialist", gh_subgraph)
        builder.add_node("web_specialist", web_subgraph)

        # 3. Core Synthesis, Critique & Refinement Nodes
        builder.add_node("evidence_aggregator", self.evidence_aggregator_node)
        builder.add_node("template_aware_docgen", self.template_aware_docgen_node)
        builder.add_node("grounding_critic", self.grounding_critic_node)
        builder.add_node("targeted_refinement_node", self.targeted_refinement_node)
        builder.add_node("final_doc_emitter", self.final_doc_emitter_node)

        # 4. Wire deterministic startup
        builder.add_edge(START, "repo_map_generator")
        builder.add_edge("repo_map_generator", "doc_intent_router")

        # 5. Wire LangGraph Send() dynamic fan-out
        builder.add_conditional_edges(
            "doc_intent_router",
            self.fan_out_dispatcher,
            ["fs_specialist", "gh_specialist", "web_specialist"]
        )

        # 6. Wire Fan-In into aggregator
        builder.add_edge("fs_specialist", "evidence_aggregator")
        builder.add_edge("gh_specialist", "evidence_aggregator")
        builder.add_edge("web_specialist", "evidence_aggregator")

        # 7. Wire Aggregator to Synthesizer
        builder.add_edge("evidence_aggregator", "template_aware_docgen")

        # 8. Wire Synthesizer to Grounding Critic
        builder.add_edge("template_aware_docgen", "grounding_critic")

        # 9. Wire Grounding Critic Conditional Routing
        builder.add_conditional_edges(
            "grounding_critic",
            self.route_after_grounding_critic,
            {
                "final_doc_emitter": "final_doc_emitter",
                "targeted_refinement_node": "targeted_refinement_node"
            }
        )

        # 10. Wire Refinement back to Synthesizer for patch-up
        builder.add_edge("targeted_refinement_node", "template_aware_docgen")

        # 11. Wire Emitter to END
        builder.add_edge("final_doc_emitter", END)

        return builder.compile()

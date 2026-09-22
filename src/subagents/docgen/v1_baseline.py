"""
Phase 1 Baseline DocGen Subgraph Implementation.
Topology: repo_map_generator -> doc_intent_router -> direct_tool_fetch -> baseline_docgen -> END.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from src.core.config import config
from src.core.service_registry import ServiceRegistry, services as default_services
from src.core.structured_output import StructuredOutputNode
from src.subagents.docgen.state import DocGenState, DocIntentPlan, SpecialistScope

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

        total = len(visible_items)
        for idx, item in enumerate(visible_items):
            if file_count >= max_files:
                lines.append(f"{prefix}└── ... [truncated at max_files={max_files}]")
                break

            is_last = (idx == total - 1)
            connector = "└── " if is_last else "├── "
            child_prefix = "    " if is_last else "│   "

            if item.is_dir():
                lines.append(f"{prefix}{connector}{item.name}/")
                _walk(item, prefix=prefix + child_prefix, depth=depth + 1)
            else:
                lines.append(f"{prefix}{connector}{item.name}")
                file_count += 1
                # Detect tech stack hints
                if item.name == "requirements.txt" or item.suffix == ".py":
                    if "Python" not in detected_stack:
                        detected_stack.append("Python")
                elif item.name == "package.json":
                    if "Node.js" not in detected_stack:
                        detected_stack.append("Node.js")
                elif item.name == "Dockerfile":
                    if "Docker" not in detected_stack:
                        detected_stack.append("Docker")

    _walk(root)
    return "\n".join(lines), detected_stack


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


SYNTHESIZER_SYSTEM_PROMPT = """You are a Principal Technical Writer and Software Architect.
Your task is to write high-density, authoritative, production-grade technical documentation based EXCLUSIVELY on the provided code and web evidence.

### Documentation Archetypes:
- If archetype is "architecture_explainer":
  * System Overview & Purpose
  * Core Components & Class Topology
  * Execution Flow / Sequence of Events
  * Fallback, Error Handling & Resilience Patterns
- If archetype is "api_reference":
  * Interface/Class Declarations & Signatures
  * Method Parameters, Types, Return Values
  * Exceptions & Error States
  * Minimal Concrete Code Usage Snippets
- If archetype is "tutorial_quickstart":
  * Prerequisites & Environment Setup
  * Configuration Variables
  * Step-by-Step Implementation Flow
  * Verification & Execution Testing

### Rigorous Grounding Rules:
1. Ground all claims in the provided evidence. DO NOT invent classes, methods, endpoints, or arguments that do not appear in the evidence.
2. Embed inline citation keys:
   - For local or github files: `[^file:path/to/file]`
   - For web search references: `[^web:DomainOrTitle]`
3. Output clean, complete Markdown with a title, table of contents, structured headings, and code blocks.
"""


class DocGenV1BaselinePipeline:
    """Encapsulates nodes for DocGen Phase 1 Baseline."""

    def __init__(self, services: Optional[ServiceRegistry] = None):
        self.services = services or default_services
        self.router_node = StructuredOutputNode(
            llm=self.services.llm,
            schema=DocIntentPlan,
            system_prompt=ROUTER_SYSTEM_PROMPT
        )

    def repo_map_generator_node(self, state: DocGenState) -> Dict[str, Any]:
        """Node 1: Deterministic Seed Node building local repo map."""
        local_path = state.get("local_path") or os.getcwd()
        repo_map_str, tech_stack = build_local_repo_map(local_path)
        return {
            "local_path": local_path,
            "repo_map": repo_map_str,
            "detected_tech_stack": tech_stack
        }

    def doc_intent_router_node(self, state: DocGenState) -> Dict[str, Any]:
        """Node 2: Structured Intent Router."""
        prompt = (
            f"Target Query: {state['user_query']}\n"
            f"Local Path: {state.get('local_path') or 'None'}\n"
            f"Remote GitHub URL: {state.get('repo_url') or 'None'}\n"
            f"Detected Tech Stack: {', '.join(state.get('detected_tech_stack', [])) or 'General'}\n\n"
            f"Repository Directory Map:\n```\n{state.get('repo_map', '')}\n```"
        )
        try:
            plan = self.router_node.invoke(user_prompt=prompt)
        except Exception as e:
            print(f"⚠️ [doc_intent_router] Fallback to default plan: {e}")
            plan = DocIntentPlan(
                doc_type="architecture_explainer",
                needs_filesystem=SpecialistScope(enabled=True, focus_scope="Analyze local files", target_paths_or_topics=[]),
                needs_github=SpecialistScope(enabled=bool(state.get("repo_url")), focus_scope="Analyze remote repo", target_paths_or_topics=[]),
                needs_web=SpecialistScope(enabled=True, focus_scope="Search web context", target_paths_or_topics=[state["user_query"]]),
                planning_rationale="Fallback plan due to routing error."
            )
        return {"intent_plan": plan}

    def direct_tool_fetch_node(self, state: DocGenState) -> Dict[str, Any]:
        """Node 3: Single-pass direct evidence retrieval."""
        plan = state.get("intent_plan")
        evidence: Dict[str, Any] = {
            "local_files": [],
            "github_files": [],
            "web_snippets": []
        }
        citations: List[str] = []

        if not plan:
            return {"retrieved_evidence": evidence, "citations": citations}

        # 1. Fetch Local Files if requested
        if plan.needs_filesystem.enabled:
            root_dir = Path(state.get("local_path") or os.getcwd())
            targets = plan.needs_filesystem.target_paths_or_topics or ["README.md"]
            for target in targets[:4]:
                target_path = root_dir / target
                if target_path.exists() and target_path.is_file():
                    try:
                        with open(target_path, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read(12000)
                        rel_path = str(target_path.relative_to(root_dir))
                        evidence["local_files"].append({
                            "file_path": rel_path,
                            "content": content,
                            "source": "local-filesystem"
                        })
                        citations.append(f"[^file:{rel_path}]")
                    except Exception as e:
                        print(f"⚠️ Error reading local file {target}: {e}")

        # 2. Fetch GitHub Files if requested and repo_url given
        if plan.needs_github.enabled and state.get("repo_url"):
            # Use GitHubMCPClient
            repo_url = state["repo_url"]
            targets = plan.needs_github.target_paths_or_topics
            try:
                from src.github_mcp_client import parse_github_url
                owner, repo, branch = parse_github_url(repo_url)
                for target in targets[:3]:
                    res = self.services.github.fetch_file_content_sync(owner, repo, target, branch=branch) if hasattr(self.services.github, "fetch_file_content_sync") else None
                    if res and "content" in res:
                        evidence["github_files"].append(res)
                        citations.append(f"[^github:{target}]")
            except Exception as e:
                print(f"⚠️ Error fetching GitHub files: {e}")

        # 3. Fetch Web Search snippets if requested
        if plan.needs_web.enabled:
            queries = plan.needs_web.target_paths_or_topics or [state["user_query"]]
            for q in queries[:2]:
                results = self.services.tavily.search(q, max_results=3)
                for r in results:
                    evidence["web_snippets"].append(r)
                    if r.get("url"):
                        citations.append(f"[^web:{r.get('title') or r.get('url')}]")

        return {"retrieved_evidence": evidence, "citations": citations}

    def baseline_docgen_node(self, state: DocGenState) -> Dict[str, Any]:
        """Node 4: Baseline Synthesis without critique loops."""
        plan = state.get("intent_plan")
        doc_type = plan.doc_type if plan else "architecture_explainer"
        evidence = state.get("retrieved_evidence", {})

        # Assemble Evidence Block
        evidence_text = []
        if evidence.get("local_files"):
            evidence_text.append("### Local Repository Files:")
            for item in evidence["local_files"]:
                evidence_text.append(f"**Path: {item['file_path']}**\n```\n{item['content']}\n```")

        if evidence.get("github_files"):
            evidence_text.append("### Remote GitHub Files:")
            for item in evidence["github_files"]:
                evidence_text.append(f"**Path: {item.get('file_path')}**\n```\n{item.get('content')}\n```")

        if evidence.get("web_snippets"):
            evidence_text.append("### Web Search & Documentation Snippets:")
            for item in evidence["web_snippets"]:
                evidence_text.append(f"- **{item.get('title', 'Web Result')}** ({item.get('url', '')}):\n  {item.get('content', '')}")

        evidence_str = "\n\n".join(evidence_text) or "No external file content was retrieved."

        user_content = (
            f"Documentation Request: {state['user_query']}\n"
            f"Archetype: {doc_type}\n"
            f"Local Path: {state.get('local_path')}\n\n"
            f"EVIDENCE CONTEXT:\n{evidence_str}"
        )

        messages = [
            SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
            HumanMessage(content=user_content)
        ]
        response = self.services.llm.invoke(messages)
        draft = getattr(response, "content", str(response))

        return {"draft_markdown": draft}

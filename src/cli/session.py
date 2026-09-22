"""
Session Management and Streaming Execution Engine for DevDocs AI CLI.
Coordinates active workspace boundaries, target repository metadata,
streaming agent action events, and multi-turn human-in-the-loop resumptions.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from src.cli.sandbox import SandboxManager, SandboxSecurityError
from src.cli.ui import (
    console,
    render_node_badge,
    render_markdown_response,
    render_clarification_dialog,
    render_error,
    render_warning,
    render_success,
    render_info
)
from src.github_mcp_client import parse_github_url
from src.supervisor.sprinter import SprinterAssistant, sprinter as default_sprinter


def detect_git_repo(path: Path) -> Optional[str]:
    """Attempts to auto-detect the GitHub repo ID from local git remotes."""
    try:
        res = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=str(path),
            capture_output=True,
            text=True,
            timeout=3
        )
        if res.returncode == 0 and res.stdout.strip():
            url = res.stdout.strip()
            owner, repo, _ = parse_github_url(url)
            return f"{owner}/{repo}"
    except Exception:
        pass
    return None


class CLISessionContext:
    """
    Encapsulates live CLI session state, sandbox policy, and agent interaction.
    """

    def __init__(
        self,
        allowed_dir: Optional[str] = None,
        repo_id: Optional[str] = None,
        docgen_version: str = "v1",
        assistant: Optional[SprinterAssistant] = None
    ):
        # 1. Resolve allowed directory
        target_path = Path(allowed_dir).resolve() if allowed_dir else Path.cwd().resolve()
        self.sandbox = SandboxManager(target_path)
        self.allowed_dir = self.sandbox.base_path

        # 2. Resolve target repo ID
        if repo_id:
            try:
                owner, repo, _ = parse_github_url(repo_id)
                self.repo_id = f"{owner}/{repo}"
            except Exception:
                self.repo_id = repo_id
        else:
            detected = detect_git_repo(self.allowed_dir)
            self.repo_id = detected or "local/workspace"

        # 3. Setup conversation session
        self.thread_id = f"cli_{int(time.time())}"
        self.docgen_version = docgen_version
        self.sprinter = assistant or default_sprinter
        if self.docgen_version != "v1":
            self.sprinter.set_docgen_version(self.docgen_version)

        # 4. Human-in-the-loop state
        self.waiting_clarification = False
        self.pending_question: Optional[str] = None
        self.last_route: Optional[str] = None

    def switch_directory(self, new_dir: str) -> Tuple[bool, str]:
        """Dynamically switches the allowed sandbox directory."""
        try:
            resolved = Path(new_dir).resolve()
            self.sandbox.update_allowed_directory(resolved)
            self.allowed_dir = resolved
            # Check if git remote exists in the new directory
            detected_repo = detect_git_repo(self.allowed_dir)
            if detected_repo and self.repo_id in ("local/workspace", ""):
                self.repo_id = detected_repo
            return True, f"Allowed workspace updated to: {self.allowed_dir}"
        except Exception as e:
            return False, f"Failed to switch directory: {e}"

    def switch_repo(self, new_repo: str) -> Tuple[bool, str]:
        """Dynamically switches the target GitHub repository."""
        try:
            owner, repo, _ = parse_github_url(new_repo)
            self.repo_id = f"{owner}/{repo}"
            return True, f"Target repository updated to: {self.repo_id}"
        except Exception as e:
            return False, f"Invalid repository identifier: {e}. Expected 'owner/repo' or GitHub URL."

    def switch_docgen_version(self, version: str) -> Tuple[bool, str]:
        """Dynamically switches the DocGen engine version."""
        v = version.strip().lower()
        if v not in ("v1", "v2", "v3"):
            return False, f"Invalid version '{version}'. Supported options: v1, v2, v3."
        self.docgen_version = v
        self.sprinter.set_docgen_version(v)
        return True, f"DocGen engine version set to: {v.upper()}"

    def reset_thread(self) -> str:
        """Resets conversational memory and starts a fresh thread."""
        self.thread_id = f"cli_{int(time.time())}"
        self.waiting_clarification = False
        self.pending_question = None
        self.last_route = None
        return self.thread_id

    def execute_turn(self, user_input: str):
        """
        Executes a user turn with live Claude Code-grade action streaming.
        Handles both normal queries and human-in-the-loop resume turns.
        """
        stripped_input = user_input.strip()
        if not stripped_input:
            return

        # --- A. Resuming from SRS Human-in-the-Loop Clarification ---
        if self.waiting_clarification:
            console.print()
            with console.status("[bold magenta]Resuming SRS Agent with your clarification...[/bold magenta]", spinner="dots"):
                res = self.sprinter.resume(self.thread_id, stripped_input)

            if res.status == "waiting_human_input":
                render_node_badge("dispatch_srs", "Further clarification required")
                self.waiting_clarification = True
                self.pending_question = res.pending_question
                render_clarification_dialog(res.pending_question or "Please provide more details.")
            elif res.status == "completed":
                render_node_badge("dispatch_srs", "SRS specification completed successfully")
                self.waiting_clarification = False
                self.pending_question = None
                render_markdown_response(res.output or "Specification generated successfully.", route="srs")
            else:
                render_node_badge("dispatch_srs", "Execution halted", str(res.error))
                render_error(res.error or "SRS resumption failed")
                self.waiting_clarification = False
            return

        # --- B. Normal Turn Execution with Node Action Streaming ---
        initial_state: Dict[str, Any] = {
            "query": stripped_input,
            "conversation_history": [{"role": "user", "content": stripped_input}],
            "route": None,
            "repo_url": self.repo_id,
            "local_path": str(self.allowed_dir),
            "final_output": "",
            "thread_id": self.thread_id,
            "status": "processing",
            "pending_question": None,
            "metadata": {
                "sandbox_root": str(self.allowed_dir),
                "docgen_version": self.docgen_version
            },
            "error": None
        }

        thread_config = {"configurable": {"thread_id": self.thread_id}}

        console.print()
        status_spinner = console.status("[bold cyan]Supervisor is analyzing request...[/bold cyan]", spinner="dots")
        status_spinner.start()

        try:
            stream_gen = self.sprinter.graph.stream(
                initial_state,
                config=thread_config,
                stream_mode="updates"
            )

            final_rendered = False

            for update in stream_gen:
                if not isinstance(update, dict):
                    continue

                for node_name, node_state in update.items():
                    # 1. Supervisor Classification Node
                    if node_name == "supervisor_classify":
                        route = node_state.get("route", "unknown")
                        self.last_route = route
                        reasoning = node_state.get("metadata", {}).get("route_reasoning", "")
                        status_spinner.stop()
                        render_node_badge(
                            "supervisor_classify",
                            f"Routing decision ➔ {route.upper()}",
                            f"Reason: {reasoning}" if reasoning else None
                        )
                        status_spinner.update(f"[bold cyan]Dispatching to {route.upper()} agent...[/bold cyan]")
                        status_spinner.start()

                    # 2. QA Dispatcher Node
                    elif node_name == "dispatch_qa":
                        status_spinner.stop()
                        render_node_badge("dispatch_qa", "Knowledge retrieval & code context completed")
                        status = node_state.get("status")
                        if status == "completed":
                            render_markdown_response(node_state.get("final_output", ""), route="qa")
                            final_rendered = True
                        elif status == "failed":
                            render_error(node_state.get("error") or "QA generation failed")
                            final_rendered = True

                    # 3. DocGen Dispatcher Node
                    elif node_name == "dispatch_docgen":
                        status_spinner.stop()
                        citations_count = node_state.get("metadata", {}).get("citations_count", 0)
                        detail = f"{citations_count} citations verified" if citations_count else "AST & tech stack analyzed"
                        render_node_badge("dispatch_docgen", f"DocGen ({self.docgen_version.upper()}) synthesized documentation", detail)
                        status = node_state.get("status")
                        if status == "completed":
                            render_markdown_response(node_state.get("final_output", ""), route="doc_gen")
                            final_rendered = True
                        elif status == "failed":
                            render_error(node_state.get("error") or "DocGen generation failed")
                            final_rendered = True

                    # 4. SRS Dispatcher Node
                    elif node_name == "dispatch_srs":
                        status_spinner.stop()
                        status = node_state.get("status")
                        if status == "waiting_human_input":
                            self.waiting_clarification = True
                            self.pending_question = node_state.get("pending_question")
                            render_node_badge("dispatch_srs", "Interrupt triggered • Clarification required")
                            render_clarification_dialog(self.pending_question or "Please clarify requirements.")
                            final_rendered = True
                        elif status == "completed":
                            render_node_badge("dispatch_srs", "SRS specification completed")
                            render_markdown_response(node_state.get("final_output", ""), route="srs")
                            final_rendered = True
                        elif status == "failed":
                            render_node_badge("dispatch_srs", "SRS generation failed")
                            render_error(node_state.get("error") or "SRS generation failed")
                            final_rendered = True

            status_spinner.stop()

        except Exception as e:
            status_spinner.stop()
            render_error(f"Execution error: {str(e)}")

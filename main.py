"""
DevDocs AI / Sprinter — Production-Grade Agentic Command-Line Interface.
Claude Code-style interactive developer assistant with real-time agent action streaming,
strict sandbox security, and multi-turn human-in-the-loop SRS requirements gathering.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output across Windows and Unix terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Optional prompt_toolkit for arrow keys, history, and modern terminal editing
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.formatted_text import HTML
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

from src.cli.sandbox import SandboxManager, SandboxSecurityError
from src.cli.session import CLISessionContext, detect_git_repo
from src.cli.ui import (
    console,
    render_banner,
    render_session_panel,
    render_help,
    render_error,
    render_warning,
    render_success,
    render_info
)


def parse_arguments() -> argparse.Namespace:
    """Parses CLI flags and runtime parameters."""
    parser = argparse.ArgumentParser(
        description="DevDocs AI — Production-Grade Claude Code-style Agentic CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-d", "--dir",
        dest="allowed_dir",
        type=str,
        default=None,
        help="Allowed local repository directory boundary (Strict Sandboxing)"
    )
    parser.add_argument(
        "-r", "--repo",
        dest="repo_id",
        type=str,
        default=None,
        help="Target GitHub repository identifier (e.g. 'owner/repo' or full URL)"
    )
    parser.add_argument(
        "-v", "--docgen-version",
        dest="docgen_version",
        type=str,
        default="v1",
        choices=["v1", "v2", "v3"],
        help="DocGen subgraph implementation version"
    )
    parser.add_argument(
        "-q", "--query",
        dest="single_query",
        type=str,
        default=None,
        help="Execute a single query non-interactively and exit"
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Suppress the pyfiglet startup banner"
    )
    return parser.parse_args()


def interactive_setup(
    cli_dir: Optional[str],
    cli_repo: Optional[str]
) -> tuple[Path, str]:
    """
    Implements Option D startup configuration:
    If flags are omitted and terminal is interactive, prompts user with detected defaults.
    """
    current_cwd = Path.cwd().resolve()
    detected_repo = detect_git_repo(current_cwd) or "M-Azfar9/Doxer-AI"

    # If both were provided via CLI flags, validate and use them directly
    if cli_dir and cli_repo:
        return Path(cli_dir).resolve(), cli_repo

    # Check if we can interactively prompt
    if not sys.stdin.isatty():
        # Non-interactive / piped execution
        chosen_dir = Path(cli_dir).resolve() if cli_dir else current_cwd
        chosen_repo = cli_repo or detected_repo
        return chosen_dir, chosen_repo

    console.print("[bold cyan]Setup Target Workspace & Repository Permissions:[/bold cyan]")
    console.print("[dim]Press Enter to accept detected defaults, or enter a custom path/ID.[/dim]\n")

    # 1. Workspace directory prompt
    if not cli_dir:
        dir_input = console.input(f"  [bold]Allowed Workspace Directory[/bold] [dim]({current_cwd})[/dim]: ").strip()
        chosen_dir = Path(dir_input).resolve() if dir_input else current_cwd
    else:
        chosen_dir = Path(cli_dir).resolve()

    if not chosen_dir.exists() or not chosen_dir.is_dir():
        render_warning(f"Specified directory '{chosen_dir}' not found. Defaulting to '{current_cwd}'.")
        chosen_dir = current_cwd

    # 2. Target GitHub repository prompt
    if not cli_repo:
        repo_input = console.input(f"  [bold]Target GitHub Repository[/bold]    [dim]({detected_repo})[/dim]: ").strip()
        chosen_repo = repo_input if repo_input else detected_repo
    else:
        chosen_repo = cli_repo

    console.print()
    return chosen_dir, chosen_repo


def handle_slash_command(cmd_line: str, session: CLISessionContext) -> bool:
    """
    Handles in-session slash commands (/help, /status, /dir, /repo, /version, /reset, /clear, /exit).
    Returns True if execution should continue, or False to exit.
    """
    parts = cmd_line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/exit", "/quit", "/q"):
        console.print("[dim]Exiting DevDocs AI. Goodbye![/dim]")
        return False

    elif cmd in ("/help", "/h", "/?"):
        render_help()

    elif cmd == "/status":
        render_session_panel(
            session.allowed_dir,
            session.repo_id,
            session.thread_id,
            session.docgen_version
        )

    elif cmd == "/dir":
        if not arg:
            render_info(f"Current allowed directory: [bold]{session.allowed_dir}[/bold]")
            render_info("To switch, enter: [cyan]/dir <path_to_directory>[/cyan]")
        else:
            success, msg = session.switch_directory(arg)
            if success:
                render_success(msg)
            else:
                render_error(msg)

    elif cmd == "/repo":
        if not arg:
            render_info(f"Current target repository: [bold]{session.repo_id}[/bold]")
            render_info("To switch, enter: [cyan]/repo <owner/repo>[/cyan]")
        else:
            success, msg = session.switch_repo(arg)
            if success:
                render_success(msg)
            else:
                render_error(msg)

    elif cmd == "/version":
        if not arg:
            render_info(f"Current DocGen version: [bold]{session.docgen_version.upper()}[/bold]")
            render_info("To switch, enter: [cyan]/version <v1|v2|v3>[/cyan]")
        else:
            success, msg = session.switch_docgen_version(arg)
            if success:
                render_success(msg)
            else:
                render_error(msg)

    elif cmd == "/reset":
        new_thread = session.reset_thread()
        render_success(f"Session memory cleared. Active thread: [dim]{new_thread}[/dim]")

    elif cmd == "/clear":
        os.system("cls" if os.name == "nt" else "clear")
        render_banner()
        render_session_panel(
            session.allowed_dir,
            session.repo_id,
            session.thread_id,
            session.docgen_version
        )

    else:
        render_warning(f"Unknown command '{cmd}'. Type [bold cyan]/help[/bold cyan] for available commands.")

    return True


def get_user_input(session: CLISessionContext, prompt_session=None) -> str:
    """Prompts the user for input with Claude Code styling."""
    if session.waiting_clarification:
        prompt_str = f"DevDocs [bold yellow]Clarification[/bold yellow] ❯ "
        html_prompt = "<ansiyellow>DevDocs [Clarification] ❯ </ansiyellow>"
    else:
        repo_short = session.repo_id.split("/")[-1] if "/" in session.repo_id else session.repo_id
        prompt_str = f"[bold cyan]DevDocs[/bold cyan] [dim][{repo_short}][/dim] ❯ "
        html_prompt = f"<ansicyan><b>DevDocs</b> [{repo_short}] ❯ </ansicyan>"

    if PROMPT_TOOLKIT_AVAILABLE and prompt_session and sys.stdin.isatty():
        try:
            return prompt_session.prompt(HTML(html_prompt)).strip()
        except (KeyboardInterrupt, EOFError):
            raise
    else:
        return console.input(prompt_str).strip()


def main():
    """Main execution entry point."""
    args = parse_arguments()

    # 1. Print pyfiglet ASCII banner unless suppressed
    if not args.no_banner:
        render_banner()

    # 2. Setup directory & repository permissions (Option D Hybrid)
    allowed_dir, repo_id = interactive_setup(args.allowed_dir, args.repo_id)

    # 3. Initialize Session Context & Sandboxing
    try:
        session = CLISessionContext(
            allowed_dir=str(allowed_dir),
            repo_id=repo_id,
            docgen_version=args.docgen_version
        )
    except Exception as e:
        render_error(f"Failed to initialize CLI session: {e}")
        sys.exit(1)

    # 4. Display Session Info Panel
    if not args.single_query:
        render_session_panel(
            session.allowed_dir,
            session.repo_id,
            session.thread_id,
            session.docgen_version
        )
        console.print("[dim]Type your prompt, or [/dim][bold cyan]/help[/bold cyan][dim] for commands, [/dim][bold cyan]/exit[/bold cyan][dim] to quit.[/dim]\n")

    # 5. Non-interactive single query execution
    if args.single_query:
        session.execute_turn(args.single_query)
        return

    # 6. Interactive REPL loop
    prompt_session = None
    if PROMPT_TOOLKIT_AVAILABLE and sys.stdin.isatty() and sys.stdout.isatty():
        try:
            prompt_session = PromptSession(history=InMemoryHistory())
        except Exception:
            prompt_session = None

    while True:
        try:
            user_input = get_user_input(session, prompt_session)
            if not user_input:
                continue

            # Check if this is a slash command
            if user_input.startswith("/"):
                keep_running = handle_slash_command(user_input, session)
                if not keep_running:
                    break
                continue

            # Execute turn through supervisor with action streaming
            session.execute_turn(user_input)

        except KeyboardInterrupt:
            console.print("\n[dim]Operation cancelled (Ctrl+C). Type /exit to quit.[/dim]\n")
            # If we were in clarification mode, reset clarification flag
            if session.waiting_clarification:
                session.waiting_clarification = False
                render_info("Clarification pause cancelled. Returning to main prompt.")
        except EOFError:
            console.print("\n[dim]Exiting DevDocs AI.[/dim]")
            break
        except Exception as e:
            render_error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()

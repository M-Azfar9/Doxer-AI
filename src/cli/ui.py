"""
Claude Code-Grade Terminal UI Components for DevDocs AI.
Powered by pyfiglet and Rich for beautiful terminal typography, live spinners,
action badges, and Markdown rendering.
"""

import sys
import pyfiglet
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich.rule import Rule
from rich.theme import Theme

# Ensure stdout and stderr are reconfigured to UTF-8
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

# Custom Claude Code-inspired theme
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "bold yellow",
    "danger": "bold red",
    "success": "bold green",
    "agent.supervisor": "bold cyan",
    "agent.qa": "bold yellow",
    "agent.docgen": "bold blue",
    "agent.srs": "bold magenta",
    "badge": "bold reverse",
})

# Use legacy_windows=False so rich uses modern ANSI / UTF-8 rendering
console = Console(theme=custom_theme, highlight=False, legacy_windows=False)


def render_banner():
    """Renders the pyfiglet ASCII art banner with Claude Code aesthetic."""
    fig = pyfiglet.figlet_format("DevDocs AI", font="slant")
    
    # Create colored gradient effect across banner lines
    banner_text = Text()
    colors = ["bright_cyan", "cyan", "deep_sky_blue1", "dodger_blue1", "royal_blue1"]
    lines = fig.splitlines()
    for idx, line in enumerate(lines):
        color = colors[idx % len(colors)]
        banner_text.append(line + "\n", style=f"bold {color}")
    
    console.print(banner_text)
    
    subtitle = Text.assemble(
        ("  ⚡ Autonomous Multi-Agent Developer Assistant  ", "bold white on dodger_blue1"),
        ("  Claude Code Edition  ", "bold black on bright_cyan"),
        ("  v1.0.0  ", "dim white")
    )
    console.print(subtitle)
    console.print(Rule(style="dim dodger_blue1"))
    console.print()


def render_session_panel(
    allowed_dir: Path,
    repo_id: str,
    thread_id: str,
    docgen_version: str = "v1",
    sandboxing_mode: str = "Strict"
):
    """Displays an active session configuration summary card."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold bright_cyan", justify="right")
    table.add_column(style="white")

    table.add_row("📁 Workspace:", f"[bold green]{allowed_dir}[/bold green]")
    table.add_row("🐙 Target Repo:", f"[bold yellow]{repo_id}[/bold yellow]")
    table.add_row("🛡️ Sandbox:", f"[bold cyan]{sandboxing_mode}[/bold cyan] (Access restricted to workspace)")
    table.add_row("⚡ Engine / DocGen:", f"[cyan]{docgen_version.upper()} Baseline[/cyan]")
    table.add_row("🧵 Thread ID:", f"[dim]{thread_id}[/dim]")

    panel = Panel(
        table,
        title="[bold white]Active Session Context[/bold white]",
        border_style="cyan",
        padding=(1, 2)
    )
    console.print(panel)
    console.print()


def render_node_badge(node_name: str, message: str, detail: Optional[str] = None):
    """Renders an action badge when an agent or node executes."""
    badges: Dict[str, Tuple[str, str, str]] = {
        "supervisor_classify": ("🤖 SUPERVISOR", "agent.supervisor", "🧭"),
        "dispatch_qa": ("💡 QA AGENT", "agent.qa", "🔍"),
        "dispatch_docgen": ("📚 DOCGEN AGENT", "agent.docgen", "📝"),
        "dispatch_srs": ("📋 SRS AGENT", "agent.srs", "⚙️"),
    }

    title, style, icon = badges.get(node_name, ("⚡ AGENT", "bold cyan", "▶"))
    
    header = Text()
    header.append(f" {icon}  ", style="bold")
    header.append(f"[{title}]", style=style)
    header.append(f" {message}", style="bold white")
    
    if detail:
        header.append(f" • {detail}", style="dim")
    
    console.print(header)


def render_markdown_response(content: str, route: Optional[str] = None):
    """Renders formatted Markdown output from the agent."""
    console.print()
    route_labels = {
        "qa": "QA Answer & Code Findings",
        "doc_gen": "Generated Documentation",
        "docgen": "Generated Documentation",
        "srs": "Software Requirements Specification",
    }
    title = route_labels.get(str(route).lower(), "Response")

    panel = Panel(
        Markdown(content),
        title=f"[bold bright_cyan]DevDocs AI ➔ {title}[/bold bright_cyan]",
        border_style="bright_blue",
        padding=(1, 2)
    )
    console.print(panel)
    console.print()


def render_clarification_dialog(question: str):
    """Renders an interactive human-in-the-loop clarification prompt."""
    console.print()
    prompt_text = Text()
    prompt_text.append("❓ The SRS Subagent requires additional domain clarification:\n\n", style="bold yellow")
    prompt_text.append(question, style="white")

    panel = Panel(
        prompt_text,
        title="[bold yellow]⚡ Clarification Requested (Human-in-the-Loop)[/bold yellow]",
        border_style="yellow",
        padding=(1, 2)
    )
    console.print(panel)
    console.print()


def render_help():
    """Displays the list of supported in-session slash commands."""
    table = Table(title="Available Slash Commands", border_style="cyan", show_header=True)
    table.add_column("Command", style="bold cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Example", style="dim")

    table.add_row("/help", "Display this guide of available commands", "/help")
    table.add_row("/status", "Show active workspace, repo, and model configuration", "/status")
    table.add_row("/dir [path]", "View or switch allowed local directory boundary", "/dir ./src")
    table.add_row("/repo [id]", "View or switch target GitHub repository identifier", "/repo owner/repo")
    table.add_row("/version [v]", "Switch DocGen engine version (v1, v2, v3)", "/version v1")
    table.add_row("/reset", "Reset session memory and initialize new thread", "/reset")
    table.add_row("/clear", "Clear the terminal screen and reprint banner", "/clear")
    table.add_row("/exit, /quit", "Gracefully terminate the CLI session", "/exit")

    console.print(table)
    console.print()


def render_error(message: str, details: Optional[str] = None):
    """Renders a standard error message."""
    text = Text()
    text.append("❌ Error: ", style="bold red")
    text.append(message, style="white")
    if details:
        text.append(f"\n   Details: {details}", style="dim red")
    console.print(text)


def render_warning(message: str):
    """Renders a warning message."""
    console.print(f"[bold yellow]⚠️  Warning:[/bold yellow] {message}")


def render_success(message: str):
    """Renders a success message."""
    console.print(f"[bold green]✓ [/bold green]{message}")


def render_info(message: str):
    """Renders an info message."""
    console.print(f"[bold cyan]ℹ [/bold cyan]{message}")

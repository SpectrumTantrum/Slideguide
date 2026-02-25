"""Rich formatting helpers for consistent CLI output."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()
error_console = Console(stderr=True)


def banner() -> None:
    """Print the SlideGuide CLI banner."""
    title = Text("SlideGuide", style="bold cyan")
    subtitle = Text("AI-Powered Lecture Tutor", style="dim")
    text = Text.assemble(title, " - ", subtitle)
    console.print(Panel(text, border_style="cyan", padding=(0, 2)))


def step_header(step: int, total: int, message: str) -> None:
    """Print a numbered step header."""
    console.print(f"\n[bold cyan][{step}/{total}][/bold cyan] {message}")


def success(message: str) -> None:
    """Print a success message."""
    console.print(f"  [green]OK[/green] {message}")


def failure(message: str) -> None:
    """Print a failure message."""
    console.print(f"  [red]FAIL[/red] {message}")


def warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"  [yellow]WARN[/yellow] {message}")


def optional(message: str) -> None:
    """Print an optional/skipped item."""
    console.print(f"  [dim]SKIP[/dim] {message}")


def info(message: str) -> None:
    """Print an info message."""
    console.print(f"  [blue]INFO[/blue] {message}")


@contextmanager
def spinner(message: str) -> Generator[None, None, None]:
    """Context manager that shows a spinner while work is in progress."""
    with console.status(f"[cyan]{message}[/cyan]", spinner="dots"):
        yield

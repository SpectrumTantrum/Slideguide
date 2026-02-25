"""Prerequisite detection: check required tools are installed and meet minimum versions."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

from cli.config import (
    IS_LINUX,
    IS_MACOS,
    IS_WINDOWS,
    MIN_NODE_VERSION,
    MIN_PYTHON_VERSION,
    MIN_SUPABASE_VERSION,
)
from cli.utils.console import console
from cli.utils.system import get_command_version, parse_version


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class PrereqResult:
    """Result of a single prerequisite check."""

    name: str
    required: bool
    found: bool
    version: tuple[int, ...] | None = None
    meets_minimum: bool = False
    minimum_version: tuple[int, ...] | None = None
    install_hint: str = ""
    extra_info: str = ""


# ── Version formatting ────────────────────────────────────────────────────────


def _fmt_version(v: tuple[int, ...] | None) -> str:
    """Format a version tuple as a dotted string, or '?' if None."""
    if v is None:
        return "?"
    return ".".join(str(p) for p in v)


def _version_gte(found: tuple[int, ...], minimum: tuple[int, ...]) -> bool:
    """Return True if *found* >= *minimum* using tuple comparison."""
    # Pad to same length for fair comparison
    length = max(len(found), len(minimum))
    f = found + (0,) * (length - len(found))
    m = minimum + (0,) * (length - len(minimum))
    return f >= m


# ── Install hints ─────────────────────────────────────────────────────────────


def _python_install_hint() -> str:
    if IS_WINDOWS:
        return (
            "Install Python 3.11+:\n"
            "  winget install Python.Python.3.11\n"
            "  or download from https://www.python.org/downloads/"
        )
    if IS_MACOS:
        return (
            "Install Python 3.11+:\n"
            "  brew install python@3.11"
        )
    # Linux
    return (
        "Install Python 3.11+:\n"
        "  Ubuntu/Debian: sudo apt install python3.11\n"
        "  Fedora/RHEL:   sudo dnf install python3.11"
    )


def _node_install_hint() -> str:
    if IS_WINDOWS:
        return (
            "Install Node.js 18+:\n"
            "  winget install OpenJS.NodeJS.LTS\n"
            "  or download from https://nodejs.org/"
        )
    if IS_MACOS:
        return (
            "Install Node.js 18+:\n"
            "  brew install node@18"
        )
    return (
        "Install Node.js 18+:\n"
        "  Ubuntu/Debian: sudo apt install nodejs\n"
        "  Fedora/RHEL:   sudo dnf install nodejs\n"
        "  Or use nvm: https://github.com/nvm-sh/nvm"
    )


def _docker_install_hint() -> str:
    if IS_WINDOWS:
        return (
            "Install Docker Desktop:\n"
            "  winget install Docker.DockerDesktop\n"
            "  or download from https://www.docker.com/products/docker-desktop/"
        )
    if IS_MACOS:
        return (
            "Install Docker Desktop:\n"
            "  brew install --cask docker\n"
            "  or download from https://www.docker.com/products/docker-desktop/"
        )
    return (
        "Install Docker:\n"
        "  Ubuntu/Debian: sudo apt install docker.io && sudo systemctl enable docker\n"
        "  Fedora/RHEL:   sudo dnf install docker && sudo systemctl enable docker\n"
        "  Or install Docker Desktop: https://www.docker.com/products/docker-desktop/"
    )


def _supabase_install_hint() -> str:
    if IS_WINDOWS:
        return (
            "Install Supabase CLI 1.0+:\n"
            "  winget install Supabase.CLI\n"
            "  or npm install -g supabase\n"
            "  or download from https://github.com/supabase/cli/releases"
        )
    if IS_MACOS:
        return (
            "Install Supabase CLI 1.0+:\n"
            "  brew install supabase/tap/supabase"
        )
    return (
        "Install Supabase CLI 1.0+:\n"
        "  npm install -g supabase\n"
        "  or download from https://github.com/supabase/cli/releases"
    )


def _tesseract_install_hint() -> str:
    if IS_WINDOWS:
        return (
            "Install Tesseract OCR (optional, for image-based slide text extraction):\n"
            "  winget install UB-Mannheim.TesseractOCR\n"
            "  or download from https://github.com/UB-Mannheim/tesseract/wiki"
        )
    if IS_MACOS:
        return (
            "Install Tesseract OCR (optional, for image-based slide text extraction):\n"
            "  brew install tesseract"
        )
    return (
        "Install Tesseract OCR (optional, for image-based slide text extraction):\n"
        "  Ubuntu/Debian: sudo apt install tesseract-ocr\n"
        "  Fedora/RHEL:   sudo dnf install tesseract"
    )


# ── Individual checks ─────────────────────────────────────────────────────────


def _check_python() -> PrereqResult:
    """Check the running Python interpreter meets the minimum version."""
    version = (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    meets = _version_gte(version, MIN_PYTHON_VERSION)
    result = PrereqResult(
        name="Python",
        required=True,
        found=True,
        version=version,
        meets_minimum=meets,
        minimum_version=MIN_PYTHON_VERSION,
    )
    if not meets:
        result.install_hint = _python_install_hint()
    return result


def _check_node() -> PrereqResult:
    """Check that Node.js is installed and meets the minimum version."""
    raw = get_command_version("node")
    if raw is None:
        return PrereqResult(
            name="Node.js",
            required=True,
            found=False,
            minimum_version=MIN_NODE_VERSION,
            install_hint=_node_install_hint(),
        )
    version = parse_version(raw)
    if version is None:
        return PrereqResult(
            name="Node.js",
            required=True,
            found=True,
            minimum_version=MIN_NODE_VERSION,
            extra_info=f"Could not parse version from: {raw}",
            install_hint=_node_install_hint(),
        )
    meets = _version_gte(version, MIN_NODE_VERSION)
    result = PrereqResult(
        name="Node.js",
        required=True,
        found=True,
        version=version,
        meets_minimum=meets,
        minimum_version=MIN_NODE_VERSION,
    )
    if not meets:
        result.install_hint = _node_install_hint()
    return result


def _check_docker() -> PrereqResult:
    """Check that Docker is installed and the daemon is running.

    Uses ``docker info`` instead of ``docker --version`` to verify the daemon
    is actually reachable.  On Windows, if the binary exists but the daemon is
    not running, we look for Docker Desktop in the standard install path and
    suggest starting it.
    """
    exe = shutil.which("docker")
    if exe is None:
        return PrereqResult(
            name="Docker",
            required=True,
            found=False,
            install_hint=_docker_install_hint(),
        )

    # Binary exists — check daemon via `docker info`
    try:
        result = subprocess.run(
            [exe, "info"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        daemon_running = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        daemon_running = False

    if daemon_running:
        # Grab version string for display
        raw = get_command_version("docker")
        version = parse_version(raw) if raw else None
        return PrereqResult(
            name="Docker",
            required=True,
            found=True,
            version=version,
            meets_minimum=True,
            extra_info="daemon running",
        )

    # Daemon not running — build a helpful hint
    hint = _docker_install_hint()
    extra = "Docker binary found but daemon is not running."

    if IS_WINDOWS:
        # Check well-known Docker Desktop path
        docker_desktop_paths = [
            os.path.join(
                os.environ.get("PROGRAMFILES", r"C:\Program Files"),
                "Docker", "Docker", "Docker Desktop.exe",
            ),
        ]
        for path in docker_desktop_paths:
            if os.path.isfile(path):
                extra = (
                    "Docker binary found but daemon is not running.\n"
                    f"  Docker Desktop found at: {path}\n"
                    "  Please start Docker Desktop and wait for it to finish initializing."
                )
                break

    return PrereqResult(
        name="Docker",
        required=True,
        found=True,
        meets_minimum=False,
        install_hint=hint,
        extra_info=extra,
    )


def _check_supabase() -> PrereqResult:
    """Check that the Supabase CLI is installed and meets the minimum version."""
    raw = get_command_version("supabase")
    if raw is None:
        return PrereqResult(
            name="Supabase CLI",
            required=True,
            found=False,
            minimum_version=MIN_SUPABASE_VERSION,
            install_hint=_supabase_install_hint(),
        )
    version = parse_version(raw)
    if version is None:
        return PrereqResult(
            name="Supabase CLI",
            required=True,
            found=True,
            minimum_version=MIN_SUPABASE_VERSION,
            extra_info=f"Could not parse version from: {raw}",
            install_hint=_supabase_install_hint(),
        )
    meets = _version_gte(version, MIN_SUPABASE_VERSION)
    result = PrereqResult(
        name="Supabase CLI",
        required=True,
        found=True,
        version=version,
        meets_minimum=meets,
        minimum_version=MIN_SUPABASE_VERSION,
    )
    if not meets:
        result.install_hint = _supabase_install_hint()
    return result


def _check_tesseract() -> PrereqResult:
    """Check for Tesseract OCR (optional dependency)."""
    raw = get_command_version("tesseract")
    if raw is None:
        return PrereqResult(
            name="Tesseract OCR",
            required=False,
            found=False,
            install_hint=_tesseract_install_hint(),
        )
    version = parse_version(raw)
    return PrereqResult(
        name="Tesseract OCR",
        required=False,
        found=True,
        version=version,
        meets_minimum=True,
    )


# ── Public API ─────────────────────────────────────────────────────────────────


def check_all_prerequisites() -> list[PrereqResult]:
    """Run all prerequisite checks and return structured results."""
    return [
        _check_python(),
        _check_node(),
        _check_docker(),
        _check_supabase(),
        _check_tesseract(),
    ]


def print_results(results: list[PrereqResult]) -> None:
    """Print prerequisite check results as a Rich table."""
    from rich.table import Table

    table = Table(title="Prerequisite Check", show_lines=False)
    table.add_column("Tool", style="bold")
    table.add_column("Required", justify="center")
    table.add_column("Status", justify="center")
    table.add_column("Version", justify="center")
    table.add_column("Minimum", justify="center")
    table.add_column("Notes")

    for r in results:
        req_str = "Yes" if r.required else "Optional"

        if not r.found:
            status = "[red]Not found[/red]"
        elif r.meets_minimum:
            status = "[green]OK[/green]"
        else:
            status = "[yellow]Version too low[/yellow]"

        version_str = _fmt_version(r.version) if r.found else "-"
        min_str = _fmt_version(r.minimum_version) if r.minimum_version else "-"

        notes = r.extra_info or ""

        table.add_row(r.name, req_str, status, version_str, min_str, notes)

    console.print()
    console.print(table)

    # Print install hints for anything that failed
    failures = [r for r in results if r.install_hint]
    if failures:
        console.print()
        for r in failures:
            label = "[red]MISSING[/red]" if not r.found else "[yellow]ACTION NEEDED[/yellow]"
            console.print(f"  {label} [bold]{r.name}[/bold]")
            for line in r.install_hint.splitlines():
                console.print(f"    {line}")
            console.print()

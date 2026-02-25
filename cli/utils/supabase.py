"""Supabase CLI wrapper: start, stop, status, reset, and output parsing."""

from __future__ import annotations

import re
import shutil
import subprocess

from cli.config import DEFAULT_SUPABASE_STUDIO_PORT, PROJECT_ROOT
from cli.utils.console import console, failure, spinner, success
from cli.utils.system import open_browser


def _find_supabase_binary() -> str:
    """Locate the ``supabase`` CLI binary via *shutil.which*.

    Raises
    ------
    FileNotFoundError
        If the binary is not on ``PATH``.
    """
    exe = shutil.which("supabase")
    if exe is None:
        raise FileNotFoundError(
            "Could not find the 'supabase' CLI on PATH. "
            "Install it via: https://supabase.com/docs/guides/cli/getting-started"
        )
    return exe


# ── Status ──────────────────────────────────────────────────────────────────


def is_supabase_running() -> bool:
    """Return *True* if the local Supabase stack is currently running.

    Runs ``supabase status`` in the project root and checks the exit code.
    """
    exe = shutil.which("supabase")
    if exe is None:
        return False
    try:
        result = subprocess.run(
            [exe, "status"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


# ── Start / Stop ────────────────────────────────────────────────────────────


def start_supabase() -> dict[str, str]:
    """Start the local Supabase stack and return connection details.

    Returns a dict with keys: ``api_url``, ``anon_key``,
    ``service_role_key``, ``db_url``.

    Raises
    ------
    RuntimeError
        If ``supabase start`` fails.
    """
    exe = _find_supabase_binary()
    with spinner("Starting Supabase (this may take a minute on first run)..."):
        try:
            result = subprocess.run(
                [exe, "start"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=300,  # first run can pull Docker images
            )
        except subprocess.TimeoutExpired as exc:
            failure("Supabase start timed out after 5 minutes")
            raise RuntimeError("supabase start timed out") from exc
        except OSError as exc:
            failure(f"Failed to execute supabase: {exc}")
            raise RuntimeError(f"Failed to execute supabase: {exc}") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        failure(f"supabase start failed (exit {result.returncode})")
        if stderr:
            console.print(f"  [dim]{stderr}[/dim]")
        raise RuntimeError(f"supabase start failed: {stderr}")

    parsed = parse_supabase_output(result.stdout)
    success("Supabase is running")
    return parsed


def stop_supabase() -> None:
    """Stop the local Supabase stack."""
    exe = _find_supabase_binary()
    with spinner("Stopping Supabase..."):
        try:
            result = subprocess.run(
                [exe, "stop"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            failure("Supabase stop timed out")
            return
        except OSError as exc:
            failure(f"Failed to execute supabase: {exc}")
            return

    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        failure(f"supabase stop failed (exit {result.returncode})")
        if stderr:
            console.print(f"  [dim]{stderr}[/dim]")
    else:
        success("Supabase stopped")


# ── Database reset ──────────────────────────────────────────────────────────


def reset_database() -> None:
    """Run ``supabase db reset`` to drop and recreate the local database.

    Prompts the user for confirmation before proceeding.
    """
    if not console.input(
        "  [yellow]This will destroy all local data. Continue?[/yellow] [dim](y/N)[/dim] "
    ).strip().lower().startswith("y"):
        console.print("  [dim]Aborted.[/dim]")
        return

    exe = _find_supabase_binary()
    with spinner("Resetting database..."):
        try:
            result = subprocess.run(
                [exe, "db", "reset"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            failure("Database reset timed out")
            return
        except OSError as exc:
            failure(f"Failed to execute supabase: {exc}")
            return

    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        failure(f"supabase db reset failed (exit {result.returncode})")
        if stderr:
            console.print(f"  [dim]{stderr}[/dim]")
    else:
        success("Database reset complete")


# ── Studio ──────────────────────────────────────────────────────────────────


def open_studio() -> None:
    """Open the Supabase Studio dashboard in the default browser."""
    url = f"http://localhost:{DEFAULT_SUPABASE_STUDIO_PORT}"
    open_browser(url)
    success(f"Opened Supabase Studio at {url}")


# ── Output parsing ──────────────────────────────────────────────────────────

# Patterns match the key-value lines printed by ``supabase start``.
# Keys have variable leading whitespace; values follow the colon.
_FIELD_PATTERNS: dict[str, re.Pattern[str]] = {
    "api_url": re.compile(r"API URL:\s*(.+)"),
    "anon_key": re.compile(r"anon key:\s*(.+)"),
    "service_role_key": re.compile(r"service_role key:\s*(.+)"),
    "db_url": re.compile(r"DB URL:\s*(.+)"),
}


def parse_supabase_output(stdout: str) -> dict[str, str]:
    """Extract connection details from ``supabase start`` output.

    Parameters
    ----------
    stdout:
        The full standard output captured from ``supabase start``.

    Returns
    -------
    dict[str, str]
        A dict with keys ``api_url``, ``anon_key``, ``service_role_key``,
        and ``db_url``.  Missing keys will have an empty-string value.
    """
    result: dict[str, str] = {}
    for key, pattern in _FIELD_PATTERNS.items():
        match = pattern.search(stdout)
        result[key] = match.group(1).strip() if match else ""
    return result

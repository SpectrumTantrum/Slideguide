"""``slideguide setup`` — interactive first-time project setup."""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import Annotated

import typer

from cli.config import ENV_FILE, FRONTEND_DIR, PROJECT_ROOT, SETUP_MARKER
from cli.utils.console import (
    banner,
    console,
    failure,
    info,
    spinner,
    step_header,
    success,
    warning,
)
from cli.utils.env_builder import interactive_configure, load_existing_env, write_env
from cli.utils.health import check_supabase_db
from cli.utils.prereqs import check_all_prerequisites, print_results
from cli.utils.supabase import is_supabase_running, start_supabase


TOTAL_STEPS = 6


def setup(
    non_interactive: Annotated[
        bool,
        typer.Option("--non-interactive", "-y", help="Skip interactive prompts"),
    ] = False,
    skip_prereqs: Annotated[
        bool,
        typer.Option("--skip-prereqs", help="Skip prerequisite checks"),
    ] = False,
    provider: Annotated[
        str | None,
        typer.Option(help="LLM provider: openrouter or lmstudio"),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option(help="API key for the selected provider"),
    ] = None,
) -> None:
    """Run the full SlideGuide first-time setup."""
    banner()

    # ── Step 1: Check prerequisites ─────────────────────────────────────────
    if skip_prereqs:
        step_header(1, TOTAL_STEPS, "Check prerequisites [dim](skipped)[/dim]")
    else:
        step_header(1, TOTAL_STEPS, "Check prerequisites")
        results = check_all_prerequisites()
        print_results(results)

        missing_required = [
            r for r in results if r.required and (not r.found or not r.meets_minimum)
        ]
        if missing_required:
            warning(
                "Some required prerequisites are missing or outdated. "
                "Setup will continue, but later steps may fail."
            )

    # ── Step 2: Configure environment ───────────────────────────────────────
    step_header(2, TOTAL_STEPS, "Configure environment")
    env_values = interactive_configure(
        non_interactive=non_interactive,
        provider=provider,
        api_key=api_key,
    )

    # ── Step 3: Setup database ──────────────────────────────────────────────
    step_header(3, TOTAL_STEPS, "Setup database")
    if is_supabase_running():
        info("Supabase is already running")
        # Still grab keys via `supabase start` which is idempotent
        keys = start_supabase()
    else:
        info("Starting Supabase...")
        keys = start_supabase()

    # Merge Supabase keys into .env
    existing = load_existing_env(ENV_FILE)
    if keys.get("api_url"):
        existing["SUPABASE_URL"] = keys["api_url"]
    if keys.get("anon_key"):
        existing["SUPABASE_ANON_KEY"] = keys["anon_key"]
    if keys.get("service_role_key"):
        existing["SUPABASE_SERVICE_ROLE_KEY"] = keys["service_role_key"]

    # Also merge any values that were set during the configure step
    for k, v in env_values.items():
        if k not in ("SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY"):
            existing[k] = v

    from cli.config import ENV_EXAMPLE

    write_env(ENV_FILE, existing, ENV_EXAMPLE)
    success("Supabase keys written to .env")

    # ── Step 4: Install backend dependencies ────────────────────────────────
    step_header(4, TOTAL_STEPS, "Install backend dependencies")
    with spinner("Installing Python packages..."):
        backend_result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".[dev]"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
    if backend_result.returncode == 0:
        success("Backend dependencies installed")
    else:
        failure("Backend dependency installation failed")
        if backend_result.stderr:
            console.print(f"  [dim]{backend_result.stderr.strip()[-500:]}[/dim]")

    # ── Step 5: Install frontend dependencies ───────────────────────────────
    step_header(5, TOTAL_STEPS, "Install frontend dependencies")
    npm_path = shutil.which("npm")
    if npm_path is None:
        failure("npm not found on PATH — skipping frontend install")
    else:
        with spinner("Installing Node.js packages..."):
            frontend_result = subprocess.run(
                [npm_path, "install"],
                cwd=str(FRONTEND_DIR),
                capture_output=True,
                text=True,
            )
        if frontend_result.returncode == 0:
            success("Frontend dependencies installed")
        else:
            failure("Frontend dependency installation failed")
            if frontend_result.stderr:
                console.print(f"  [dim]{frontend_result.stderr.strip()[-500:]}[/dim]")

    # ── Step 6: Verify installation ─────────────────────────────────────────
    step_header(6, TOTAL_STEPS, "Verify installation")

    # Check backend import
    import_result = subprocess.run(
        [sys.executable, "-c", "import backend"],
        capture_output=True,
        text=True,
    )
    if import_result.returncode == 0:
        success("Backend module imports successfully")
    else:
        failure("Backend module failed to import")

    # Check database connection
    if check_supabase_db():
        success("Database is reachable")
    else:
        failure("Database is not reachable")

    # ── Write setup marker ──────────────────────────────────────────────────
    SETUP_MARKER.parent.mkdir(parents=True, exist_ok=True)
    SETUP_MARKER.touch()

    # ── Final banner ────────────────────────────────────────────────────────
    console.print()
    console.print("[bold green]Setup complete![/bold green]")
    console.print()
    console.print("Next steps:")
    console.print("  [cyan]slideguide start[/cyan]   Start the backend and frontend servers")
    console.print("  [cyan]slideguide status[/cyan]  Check the status of all services")
    console.print("  [cyan]slideguide doctor[/cyan]  Run diagnostics if something isn't working")
    console.print()

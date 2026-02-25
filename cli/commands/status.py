"""``slideguide status`` and ``slideguide doctor`` — service status and diagnostics."""

from __future__ import annotations

import subprocess
import sys

from rich.table import Table

from cli.config import (
    DEFAULT_BACKEND_PORT,
    DEFAULT_FRONTEND_PORT,
    DEFAULT_SUPABASE_API_PORT,
    ENV_FILE,
    FRONTEND_DIR,
)
from cli.utils.console import console, failure, info, success
from cli.utils.health import check_health, check_provider_status, check_supabase_db
from cli.utils.prereqs import check_all_prerequisites, print_results
from cli.utils.processes import get_running_services
from cli.utils.supabase import is_supabase_running
from cli.utils.system import is_port_open


# ── Required env vars for the doctor check ──────────────────────────────────

_REQUIRED_ENV_VARS = ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY")
_PROVIDER_KEY_VARS = ("OPENROUTER_API_KEY", "LMSTUDIO_BASE_URL")


def _load_env_keys() -> dict[str, str]:
    """Read the .env file and return a dict of key=value pairs (values may be empty)."""
    result: dict[str, str] = {}
    if not ENV_FILE.exists():
        return result
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip().strip("\"'")
    return result


# ── status command ───────────────────────────────────────────────────────────


def status() -> None:
    """Show status of all SlideGuide services."""
    services = get_running_services()

    table = Table(title="SlideGuide Services", show_lines=False)
    table.add_column("Service", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Port", justify="center")
    table.add_column("PID", justify="center")
    table.add_column("URL")

    # ── Supabase ────────────────────────────────────────────────────────────
    supabase_running = is_supabase_running()
    table.add_row(
        "Supabase",
        "[green]running[/green]" if supabase_running else "[red]stopped[/red]",
        str(DEFAULT_SUPABASE_API_PORT),
        "-",
        f"http://127.0.0.1:{DEFAULT_SUPABASE_API_PORT}" if supabase_running else "-",
    )

    # ── Backend ─────────────────────────────────────────────────────────────
    backend_info = services.get("backend")
    backend_port = backend_info["port"] if backend_info and backend_info.get("port") else DEFAULT_BACKEND_PORT
    backend_url = f"http://127.0.0.1:{backend_port}"
    backend_healthy = check_health(f"{backend_url}/health/live")
    backend_running = backend_info is not None or backend_healthy

    table.add_row(
        "Backend",
        "[green]running[/green]" if backend_running else "[red]stopped[/red]",
        str(backend_port),
        str(backend_info["pid"]) if backend_info else "-",
        backend_url if backend_running else "-",
    )

    # ── Frontend ────────────────────────────────────────────────────────────
    frontend_info = services.get("frontend")
    frontend_port = frontend_info["port"] if frontend_info and frontend_info.get("port") else DEFAULT_FRONTEND_PORT
    frontend_url = f"http://127.0.0.1:{frontend_port}"
    frontend_reachable = is_port_open(frontend_port)
    frontend_running = frontend_info is not None or frontend_reachable

    table.add_row(
        "Frontend",
        "[green]running[/green]" if frontend_running else "[red]stopped[/red]",
        str(frontend_port),
        str(frontend_info["pid"]) if frontend_info else "-",
        frontend_url if frontend_running else "-",
    )

    console.print()
    console.print(table)

    # ── Provider info (only if backend is up) ───────────────────────────────
    if backend_running:
        provider = check_provider_status(backend_url)
        if provider:
            console.print()
            console.print("[bold]Provider Configuration[/bold]")
            for key, value in provider.items():
                console.print(f"  {key}: [cyan]{value}[/cyan]")
        console.print()


# ── doctor command ───────────────────────────────────────────────────────────


def doctor() -> None:
    """Run comprehensive health checks."""
    console.print()
    console.print("[bold cyan]SlideGuide Doctor[/bold cyan]")
    console.print("[dim]Running comprehensive health checks...[/dim]")

    all_ok = True

    # ── 1. Prerequisites ────────────────────────────────────────────────────
    console.print()
    console.print("[bold]1. Prerequisites[/bold]")
    results = check_all_prerequisites()
    print_results(results)

    missing_required = [
        r for r in results if r.required and (not r.found or not r.meets_minimum)
    ]
    if missing_required:
        all_ok = False

    # ── 2. Configuration ────────────────────────────────────────────────────
    console.print()
    console.print("[bold]2. Configuration[/bold]")

    if ENV_FILE.exists():
        success(".env file exists")
        env_keys = _load_env_keys()

        # Check required variables
        for var in _REQUIRED_ENV_VARS:
            if env_keys.get(var):
                success(f"{var} is set")
            else:
                failure(f"{var} is not set")
                info(f"  Fix: Add {var}=<value> to {ENV_FILE}")
                all_ok = False

        # Check for at least one provider key
        has_provider = any(env_keys.get(var) for var in _PROVIDER_KEY_VARS)
        if has_provider:
            success("At least one provider key is configured")
        else:
            failure("No provider key found")
            info(
                "  Fix: Set OPENROUTER_API_KEY or LMSTUDIO_BASE_URL in "
                f"{ENV_FILE}"
            )
            all_ok = False
    else:
        failure(f".env file not found at {ENV_FILE}")
        info("  Fix: Run [cyan]slideguide setup[/cyan] to create it")
        all_ok = False

    # ── 3. Services ─────────────────────────────────────────────────────────
    console.print()
    console.print("[bold]3. Services[/bold]")

    if is_supabase_running():
        success("Supabase is running")
    else:
        failure("Supabase is not running")
        info("  Fix: Run [cyan]supabase start[/cyan] or [cyan]slideguide start[/cyan]")
        all_ok = False

    backend_url = f"http://127.0.0.1:{DEFAULT_BACKEND_PORT}"
    if check_health(f"{backend_url}/health/live"):
        success("Backend is healthy")
    else:
        failure("Backend is not reachable")
        info(
            "  Fix: Run [cyan]slideguide start[/cyan] or "
            "[cyan]uvicorn backend.main:app --reload --port 8000[/cyan]"
        )
        all_ok = False

    frontend_url = f"http://127.0.0.1:{DEFAULT_FRONTEND_PORT}"
    if is_port_open(DEFAULT_FRONTEND_PORT):
        success(f"Frontend is reachable at {frontend_url}")
    else:
        failure("Frontend is not reachable")
        info(
            "  Fix: Run [cyan]slideguide start[/cyan] or "
            "[cyan]cd frontend && npm run dev[/cyan]"
        )
        all_ok = False

    # ── 4. Database ─────────────────────────────────────────────────────────
    console.print()
    console.print("[bold]4. Database[/bold]")

    if check_supabase_db():
        success("Supabase PostgreSQL is accepting connections")
    else:
        failure("Supabase PostgreSQL is not reachable")
        info(
            "  Fix: Ensure Supabase is running with "
            "[cyan]supabase start[/cyan]"
        )
        all_ok = False

    # ── 5. Dependencies ────────────────────────────────────────────────────
    console.print()
    console.print("[bold]5. Dependencies[/bold]")

    # Check frontend node_modules
    node_modules = FRONTEND_DIR / "node_modules"
    if node_modules.is_dir():
        success("Frontend node_modules/ exists")
    else:
        failure("Frontend node_modules/ not found")
        info(
            "  Fix: Run [cyan]cd frontend && npm install[/cyan] or "
            "[cyan]slideguide setup[/cyan]"
        )
        all_ok = False

    # Check backend package is installed
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import backend"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            success("Backend package is installed")
        else:
            failure("Backend package is not installed")
            info(
                "  Fix: Run [cyan]pip install -e '.[dev]'[/cyan] or "
                "[cyan]slideguide setup[/cyan]"
            )
            all_ok = False
    except (subprocess.TimeoutExpired, OSError):
        failure("Could not check backend package")
        info("  Fix: Run [cyan]pip install -e '.[dev]'[/cyan]")
        all_ok = False

    # ── Summary ─────────────────────────────────────────────────────────────
    console.print()
    if all_ok:
        console.print("[bold green]All checks passed![/bold green]")
    else:
        console.print(
            "[bold yellow]Some checks failed.[/bold yellow] "
            "Review the issues above and apply the suggested fixes."
        )
    console.print()

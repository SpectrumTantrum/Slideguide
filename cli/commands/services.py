"""Service lifecycle commands: start, stop, restart, dev, and logs."""

from __future__ import annotations

import shutil
import signal
import sys
from typing import Annotated

import typer
from rich.table import Table

from cli.config import (
    DEFAULT_BACKEND_PORT,
    DEFAULT_FRONTEND_PORT,
    FRONTEND_DIR,
    IS_WINDOWS,
    PROJECT_ROOT,
    SETUP_MARKER,
)
from cli.utils.console import console, failure, info, spinner, success, warning
from cli.utils.health import wait_for_health
from cli.utils.processes import (
    get_running_services,
    start_service,
    stop_all_services,
    stop_service,
    stream_logs,
)
from cli.utils.supabase import is_supabase_running, start_supabase, stop_supabase
from cli.utils.system import find_process_on_port, is_port_free, open_browser


# ── Helpers ──────────────────────────────────────────────────────────────────


def _check_setup_complete() -> bool:
    """Return True if initial setup has been completed; warn otherwise."""
    if SETUP_MARKER.exists():
        return True
    warning(
        "Initial setup has not been completed. "
        "Run [bold]slideguide setup[/bold] first."
    )
    return False


def _check_ports_free(*ports: tuple[str, int]) -> bool:
    """Verify that all listed ports are available.

    *ports* is a sequence of ``(service_name, port_number)`` tuples.
    Returns True if every port is free; prints diagnostics and returns
    False otherwise.
    """
    all_free = True
    for name, port in ports:
        if not is_port_free(port):
            occupant = find_process_on_port(port)
            detail = f" (used by {occupant})" if occupant else ""
            failure(f"Port {port} required by {name} is already in use{detail}")
            all_free = False
    return all_free


def _print_service_table(
    backend_port: int,
    frontend_port: int,
) -> None:
    """Print a summary table with all service URLs."""
    table = Table(title="SlideGuide Services", border_style="cyan")
    table.add_column("Service", style="bold")
    table.add_column("URL", style="green")

    table.add_row("Frontend", f"http://localhost:{frontend_port}")
    table.add_row("Backend API", f"http://127.0.0.1:{backend_port}")
    table.add_row("API Docs", f"http://127.0.0.1:{backend_port}/docs")
    table.add_row("Health", f"http://127.0.0.1:{backend_port}/health/live")

    console.print()
    console.print(table)
    console.print()


# ── Commands ─────────────────────────────────────────────────────────────────


def start(
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Don't auto-open browser")
    ] = False,
    backend_port: Annotated[
        int, typer.Option("--backend-port", help="Backend port")
    ] = DEFAULT_BACKEND_PORT,
    frontend_port: Annotated[
        int, typer.Option("--frontend-port", help="Frontend port")
    ] = DEFAULT_FRONTEND_PORT,
) -> None:
    """Start all SlideGuide services (Supabase, backend, frontend)."""

    # 1. Check setup is complete
    _check_setup_complete()

    # 2. Check ports are free
    if not _check_ports_free(
        ("backend", backend_port),
        ("frontend", frontend_port),
    ):
        raise typer.Exit(1)

    # 3. Ensure Supabase is running
    if not is_supabase_running():
        info("Supabase is not running — starting it now...")
        try:
            start_supabase()
        except RuntimeError:
            failure("Could not start Supabase. Check Docker and try again.")
            raise typer.Exit(1)
    else:
        success("Supabase already running")

    # 4. Start backend
    info("Starting backend...")
    backend_pid = start_service(
        "backend",
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--reload", "--port", str(backend_port)],
        cwd=PROJECT_ROOT,
        port=backend_port,
    )
    success(f"Backend started (PID {backend_pid})")

    # 5. Start frontend
    npm_path = shutil.which("npm")
    if npm_path is None:
        failure("npm not found on PATH. Please install Node.js.")
        # Stop the backend we just started before exiting
        stop_service("backend")
        raise typer.Exit(1)

    info("Starting frontend...")
    frontend_pid = start_service(
        "frontend",
        [npm_path, "run", "dev"],
        cwd=FRONTEND_DIR,
        port=frontend_port,
    )
    success(f"Frontend started (PID {frontend_pid})")

    # 6. Wait for backend health
    with spinner("Waiting for backend to become healthy..."):
        healthy = wait_for_health(f"http://127.0.0.1:{backend_port}/health/live")

    if healthy:
        success("Backend is healthy")
    else:
        warning("Backend did not respond in time — it may still be starting up")

    # 7. Print summary table
    _print_service_table(backend_port, frontend_port)

    # 8. Open browser
    if not no_browser:
        open_browser(f"http://localhost:{frontend_port}")
        info(f"Opened browser at http://localhost:{frontend_port}")

    # 9. Install signal handlers for graceful shutdown
    def _shutdown_handler(signum: int, frame: object) -> None:
        console.print("\n[yellow]Shutting down...[/yellow]")
        stop_all_services()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown_handler)
    if IS_WINDOWS:
        signal.signal(signal.SIGBREAK, _shutdown_handler)  # type: ignore[attr-defined]

    # 10. Stream combined logs until interrupted
    stream_logs(["backend", "frontend"])

    # If stream_logs returns (e.g. no services), clean up
    stop_all_services()


def stop(
    all_services: Annotated[
        bool, typer.Option("--all", help="Also stop Supabase")
    ] = False,
) -> None:
    """Stop running SlideGuide services."""

    running = get_running_services()

    if not running and not all_services:
        info("No tracked services are running.")
        return

    # Stop frontend first, then backend (reverse startup order)
    for name in ("frontend", "backend"):
        if name in running:
            stop_service(name)

    # Stop any other tracked services not named above
    for name in running:
        if name not in ("frontend", "backend"):
            stop_service(name)

    if all_services:
        info("Stopping Supabase...")
        stop_supabase()

    success("All services stopped")


def restart(
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Don't auto-open browser")
    ] = False,
    backend_port: Annotated[
        int, typer.Option("--backend-port", help="Backend port")
    ] = DEFAULT_BACKEND_PORT,
    frontend_port: Annotated[
        int, typer.Option("--frontend-port", help="Frontend port")
    ] = DEFAULT_FRONTEND_PORT,
) -> None:
    """Restart all SlideGuide services."""
    info("Stopping services...")
    stop(all_services=False)
    info("Starting services...")
    start(
        no_browser=no_browser,
        backend_port=backend_port,
        frontend_port=frontend_port,
    )


def dev(
    backend_port: Annotated[
        int, typer.Option("--backend-port", help="Backend port")
    ] = DEFAULT_BACKEND_PORT,
    frontend_port: Annotated[
        int, typer.Option("--frontend-port", help="Frontend port")
    ] = DEFAULT_FRONTEND_PORT,
) -> None:
    """Start services in development mode (debug logging, no browser)."""

    # 1. Check setup is complete
    if not _check_setup_complete():
        info("Continuing anyway — some features may not work.")

    # 2. Check ports are free
    if not _check_ports_free(
        ("backend", backend_port),
        ("frontend", frontend_port),
    ):
        raise typer.Exit(1)

    # 3. Ensure Supabase is running
    if not is_supabase_running():
        info("Supabase is not running — starting it now...")
        try:
            start_supabase()
        except RuntimeError:
            failure("Could not start Supabase. Check Docker and try again.")
            raise typer.Exit(1)
    else:
        success("Supabase already running")

    debug_env = {"SLIDEGUIDE_DEBUG": "1"}

    # 4. Start backend with debug env
    info("Starting backend (debug mode)...")
    backend_pid = start_service(
        "backend",
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--reload", "--port", str(backend_port)],
        cwd=PROJECT_ROOT,
        env=debug_env,
        port=backend_port,
    )
    success(f"Backend started (PID {backend_pid})")

    # 5. Start frontend with debug env
    npm_path = shutil.which("npm")
    if npm_path is None:
        failure("npm not found on PATH. Please install Node.js.")
        stop_service("backend")
        raise typer.Exit(1)

    info("Starting frontend (debug mode)...")
    frontend_pid = start_service(
        "frontend",
        [npm_path, "run", "dev"],
        cwd=FRONTEND_DIR,
        env=debug_env,
        port=frontend_port,
    )
    success(f"Frontend started (PID {frontend_pid})")

    # 6. Wait for backend health
    with spinner("Waiting for backend to become healthy..."):
        healthy = wait_for_health(f"http://127.0.0.1:{backend_port}/health/live")

    if healthy:
        success("Backend is healthy")
    else:
        warning("Backend did not respond in time — it may still be starting up")

    # 7. Print summary table
    _print_service_table(backend_port, frontend_port)
    info("Debug mode active — SLIDEGUIDE_DEBUG=1")

    # 8. Signal handlers for graceful shutdown
    def _shutdown_handler(signum: int, frame: object) -> None:
        console.print("\n[yellow]Shutting down...[/yellow]")
        stop_all_services()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown_handler)
    if IS_WINDOWS:
        signal.signal(signal.SIGBREAK, _shutdown_handler)  # type: ignore[attr-defined]

    # 9. Stream combined logs until interrupted
    stream_logs(["backend", "frontend"])

    # If stream_logs returns, clean up
    stop_all_services()


def logs() -> None:
    """Stream logs from running SlideGuide services."""

    running = get_running_services()
    if not running:
        info("No tracked services are running.")
        raise typer.Exit(0)

    service_names = list(running.keys())
    info(f"Streaming logs for: {', '.join(service_names)}")
    stream_logs(service_names)

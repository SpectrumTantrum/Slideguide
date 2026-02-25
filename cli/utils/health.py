"""Health-check polling utilities for SlideGuide services."""

from __future__ import annotations

import time
from typing import Any

import httpx

from cli.config import DEFAULT_BACKEND_PORT, DEFAULT_SUPABASE_DB_PORT
from cli.utils.system import is_port_open


# ── Single health check ──────────────────────────────────────────────────────


def check_health(url: str, timeout: float = 5.0) -> bool:
    """Return True if *url* responds with HTTP 200, False otherwise."""
    try:
        response = httpx.get(url, timeout=timeout)
        return response.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


# ── Polling with exponential backoff ──────────────────────────────────────────


def wait_for_health(
    url: str,
    timeout: float = 30.0,
    interval: float = 1.0,
) -> bool:
    """Poll *url* until it returns HTTP 200, or *timeout* seconds elapse.

    Uses exponential backoff starting at *interval* seconds, capped at 5 s
    between retries.

    Returns True if the service became healthy, False on timeout.
    """
    deadline = time.monotonic() + timeout
    delay = interval

    while time.monotonic() < deadline:
        if check_health(url, timeout=min(delay, 5.0)):
            return True
        # Sleep for the lesser of the current delay or the remaining time
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(delay, remaining))
        delay = min(delay * 2, 5.0)

    return False


# ── Supabase DB (PostgreSQL) port check ───────────────────────────────────────


def check_supabase_db(
    host: str = "127.0.0.1",
    port: int = DEFAULT_SUPABASE_DB_PORT,
) -> bool:
    """Return True if the Supabase PostgreSQL port is accepting connections."""
    return is_port_open(port, host=host)


# ── Provider status from backend ──────────────────────────────────────────────


def check_provider_status(
    backend_url: str = f"http://127.0.0.1:{DEFAULT_BACKEND_PORT}",
) -> dict[str, Any] | None:
    """Query the backend for the active LLM provider configuration.

    Returns the parsed JSON dict on success, or ``None`` if the request fails.
    """
    try:
        response = httpx.get(f"{backend_url}/api/settings/provider", timeout=5.0)
        if response.status_code == 200:
            return response.json()  # type: ignore[no-any-return]
        return None
    except (httpx.HTTPError, OSError, ValueError):
        return None

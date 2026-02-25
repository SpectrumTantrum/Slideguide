"""Cross-platform OS utilities: ports, versions, processes, filesystem."""

from __future__ import annotations

import re
import shutil
import socket
import subprocess
import webbrowser
from pathlib import Path

from cli.config import IS_WINDOWS


# ── Port utilities ──────────────────────────────────────────────────────────


def is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a port is free to bind on (works on all platforms)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def is_port_open(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    """Check if a service is listening on a port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def find_process_on_port(port: int) -> str | None:
    """Return a description of the process using the given port, or None."""
    try:
        if IS_WINDOWS:
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    pid = parts[-1]
                    # Look up process name
                    task = subprocess.run(
                        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    name = task.stdout.strip().split(",")[0].strip('"') if task.stdout.strip() else "unknown"
                    return f"{name} (PID {pid})"
        else:
            # Unix: try ss first, fall back to lsof
            for cmd in [
                ["ss", "-tlnp", f"sport = :{port}"],
                ["lsof", "-i", f":{port}", "-sTCP:LISTEN", "-t"],
            ]:
                if shutil.which(cmd[0]):
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    if result.stdout.strip():
                        return result.stdout.strip().splitlines()[0]
    except (subprocess.TimeoutExpired, FileNotFoundError, IndexError):
        pass
    return None


# ── Version utilities ───────────────────────────────────────────────────────

_VERSION_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")


def parse_version(version_str: str) -> tuple[int, ...] | None:
    """Extract (major, minor, patch) from a version string. Returns None on failure."""
    match = _VERSION_RE.search(version_str)
    if not match:
        return None
    major, minor = int(match.group(1)), int(match.group(2))
    patch = int(match.group(3)) if match.group(3) else 0
    return (major, minor, patch)


def get_command_version(cmd: str, flag: str = "--version") -> str | None:
    """Run `cmd --version` and return the raw output, or None if not found."""
    exe = shutil.which(cmd)
    if not exe:
        return None
    try:
        result = subprocess.run(
            [exe, flag],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout.strip() or result.stderr.strip()
        return output if output else None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


# ── Process utilities ───────────────────────────────────────────────────────


def is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    if IS_WINDOWS:
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return exit_code.value == STILL_ACTIVE
            return False
        finally:
            kernel32.CloseHandle(handle)
    else:
        import os

        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


# ── Filesystem / browser ───────────────────────────────────────────────────


def open_browser(url: str) -> None:
    """Open a URL in the user's default browser."""
    webbrowser.open(url)


def copy_env_example(src: Path, dst: Path) -> None:
    """Copy .env.example to .env (cross-platform)."""
    shutil.copy2(src, dst)

"""Cross-platform process management: start, stop, track, and stream services."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from cli.config import IS_WINDOWS, LOG_DIR, PID_FILE, STATE_DIR
from cli.utils.console import console, failure, success
from cli.utils.system import is_pid_alive


# ── PID storage ───────────────────────────────────────────────────────────────


def _ensure_dirs() -> None:
    """Create state and log directories if they don't exist."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_pids() -> dict:
    """Load the PID registry from disk. Returns empty dict on missing/corrupt file."""
    try:
        return json.loads(PID_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_pids(data: dict) -> None:
    """Persist the full PID registry to disk."""
    _ensure_dirs()
    PID_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_pid(name: str, pid: int, port: int | None) -> None:
    """Record a single service entry in the PID registry."""
    pids = load_pids()
    pids[name] = {
        "pid": pid,
        "port": port,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    save_pids(pids)


def remove_pid(name: str) -> None:
    """Remove a service entry from the PID registry."""
    pids = load_pids()
    pids.pop(name, None)
    save_pids(pids)


def get_service_info(name: str) -> dict | None:
    """Return the stored info dict for *name*, or None if not tracked."""
    return load_pids().get(name)


def get_running_services() -> dict:
    """Return all tracked services, pruning entries whose PIDs are no longer alive."""
    pids = load_pids()
    alive: dict = {}
    changed = False
    for name, entry in pids.items():
        if is_pid_alive(entry["pid"]):
            alive[name] = entry
        else:
            changed = True
    if changed:
        save_pids(alive)
    return alive


# ── Starting processes ────────────────────────────────────────────────────────


def start_service(
    name: str,
    cmd: list[str],
    cwd: Path | None = None,
    env: dict | None = None,
    port: int | None = None,
) -> int:
    """Start a subprocess for *name* and return its PID.

    * On Windows the process is placed in a new process group so it does not
      receive console signals from the parent.
    * On Unix a new session is created.
    * stdout / stderr are redirected to per-service log files under LOG_DIR.
    """
    _ensure_dirs()

    # Resolve the executable — critical for Windows where npm is actually npm.cmd
    resolved = shutil.which(cmd[0])
    if resolved is not None:
        cmd = [resolved, *cmd[1:]]

    # Build merged environment
    merged_env: dict[str, str] | None = None
    if env is not None:
        merged_env = {**os.environ, **env}

    # Open log files
    stdout_log = LOG_DIR / f"{name}.log"
    stderr_log = LOG_DIR / f"{name}.err.log"
    stdout_fh = open(stdout_log, "a", encoding="utf-8")  # noqa: SIM115
    stderr_fh = open(stderr_log, "a", encoding="utf-8")  # noqa: SIM115

    kwargs: dict = {
        "stdout": stdout_fh,
        "stderr": stderr_fh,
        "cwd": str(cwd) if cwd else None,
        "env": merged_env,
    }

    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(cmd, **kwargs)  # noqa: S603
    except Exception:
        stdout_fh.close()
        stderr_fh.close()
        raise

    save_pid(name, proc.pid, port)
    return proc.pid


# ── Stopping processes ────────────────────────────────────────────────────────


def _stop_windows(pid: int) -> bool:
    """Attempt graceful then forceful termination on Windows."""
    # Graceful: CTRL_BREAK_EVENT to the process group
    try:
        os.kill(pid, signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
    except OSError:
        pass

    # Wait up to 3 seconds for graceful exit
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if not is_pid_alive(pid):
            return True
        time.sleep(0.2)

    # Forceful: taskkill the entire process tree
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Final check
    time.sleep(0.3)
    return not is_pid_alive(pid)


def _stop_unix(pid: int) -> bool:
    """Attempt graceful then forceful termination on Unix."""
    try:
        pgid = os.getpgid(pid)
    except OSError:
        # Process already gone
        return True

    # Graceful: SIGTERM to the process group
    try:
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        pass

    # Wait up to 5 seconds
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not is_pid_alive(pid):
            return True
        time.sleep(0.2)

    # Forceful: SIGKILL the whole group
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass

    time.sleep(0.3)
    return not is_pid_alive(pid)


def stop_service(name: str) -> bool:
    """Stop the service identified by *name*. Returns True if successfully stopped."""
    svc_info = get_service_info(name)
    if svc_info is None:
        failure(f"No tracked service named '{name}'")
        return False

    pid = svc_info["pid"]

    if not is_pid_alive(pid):
        remove_pid(name)
        success(f"{name} (PID {pid}) already exited")
        return True

    if IS_WINDOWS:
        stopped = _stop_windows(pid)
    else:
        stopped = _stop_unix(pid)

    if stopped:
        remove_pid(name)
        success(f"{name} stopped (PID {pid})")
    else:
        failure(f"Could not stop {name} (PID {pid})")

    return stopped


def stop_all_services() -> None:
    """Stop every service in the PID registry."""
    pids = load_pids()
    if not pids:
        console.print("  No tracked services to stop.")
        return
    for name in list(pids):
        stop_service(name)


# ── Log streaming ─────────────────────────────────────────────────────────────

# Color palette for service prefixes (Rich markup)
_SERVICE_COLORS: dict[str, str] = {
    "backend": "cyan",
    "frontend": "magenta",
    "supabase": "green",
}


def _tail_log(service: str, stop_event: threading.Event) -> None:
    """Continuously read new lines from a service log file and print them."""
    log_path = LOG_DIR / f"{service}.log"
    color = _SERVICE_COLORS.get(service, "white")
    prefix = f"[bold {color}][{service}][/bold {color}]"

    # Wait for the file to appear (up to 10 s)
    waited = 0.0
    while not log_path.exists() and waited < 10.0:
        if stop_event.is_set():
            return
        time.sleep(0.3)
        waited += 0.3

    if not log_path.exists():
        console.print(f"{prefix} [dim]log file not found[/dim]")
        return

    with open(log_path, encoding="utf-8", errors="replace") as fh:
        # Seek to end so we only stream new output
        fh.seek(0, 2)
        while not stop_event.is_set():
            line = fh.readline()
            if line:
                console.print(f"{prefix} {line.rstrip()}")
            else:
                time.sleep(0.1)


def stream_logs(services: list[str]) -> None:
    """Stream log output from *services* until the user presses Ctrl+C."""
    if not services:
        console.print("  No services specified for log streaming.")
        return

    stop_event = threading.Event()
    threads: list[threading.Thread] = []

    for svc in services:
        t = threading.Thread(target=_tail_log, args=(svc, stop_event), daemon=True)
        t.start()
        threads.append(t)

    console.print("[dim]Streaming logs — press Ctrl+C to stop[/dim]\n")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=2.0)

"""CLI configuration: paths, defaults, constants."""

from __future__ import annotations

import platform
from pathlib import Path

# ── Platform ────────────────────────────────────────────────────────────────
IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"
PLATFORM = platform.system()


# ── Project root detection ──────────────────────────────────────────────────
def _find_project_root() -> Path:
    """Walk up from this file to find the directory containing pyproject.toml."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root (no pyproject.toml found)")


PROJECT_ROOT = _find_project_root()

# ── State directory ─────────────────────────────────────────────────────────
STATE_DIR = PROJECT_ROOT / ".slideguide"
PID_FILE = STATE_DIR / "pids.json"
SETUP_MARKER = STATE_DIR / "setup_complete"
LOG_DIR = STATE_DIR / "logs"

# ── Key project paths ──────────────────────────────────────────────────────
FRONTEND_DIR = PROJECT_ROOT / "frontend"
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"

# ── Default ports ───────────────────────────────────────────────────────────
DEFAULT_BACKEND_PORT = 8000
DEFAULT_FRONTEND_PORT = 3000
DEFAULT_SUPABASE_API_PORT = 54321
DEFAULT_SUPABASE_DB_PORT = 54322
DEFAULT_SUPABASE_STUDIO_PORT = 54323

# ── Minimum versions ───────────────────────────────────────────────────────
MIN_PYTHON_VERSION = (3, 11)
MIN_NODE_VERSION = (18, 0)
MIN_SUPABASE_VERSION = (1, 0)

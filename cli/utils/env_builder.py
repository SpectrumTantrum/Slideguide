"""Interactive .env configuration builder for SlideGuide."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import httpx
import typer
from dotenv import dotenv_values

from cli.config import ENV_EXAMPLE, ENV_FILE
from cli.utils.console import console, failure, info, success, warning

# Variables that are filled in automatically by `slideguide setup` after
# running `supabase start`, so we never prompt for them interactively.
_SUPABASE_KEYS = {
    "SUPABASE_URL",
    "SUPABASE_ANON_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
}

# Regex that matches a KEY=VALUE line, optionally followed by an inline comment.
# Captures:  (1) variable name  (2) value (may be empty)  (3) inline comment
_ENV_LINE_RE = re.compile(
    r"^(?P<name>[A-Z_][A-Z0-9_]*)=(?P<value>[^#]*)(?:#\s*(?P<comment>.*))?$"
)


# ── Data model ──────────────────────────────────────────────────────────────


@dataclass
class EnvVar:
    """A single environment variable parsed from .env.example."""

    name: str
    default: str
    comment: str


# ── Parsing helpers ─────────────────────────────────────────────────────────


def parse_env_example(path: Path) -> list[EnvVar]:
    """Parse .env.example and return a list of discovered variables.

    Each variable carries its default value (right-hand side, stripped) and
    any inline comment that appeared after ``#`` on the same line.
    """
    variables: list[EnvVar] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        match = _ENV_LINE_RE.match(line)
        if match:
            name = match.group("name")
            value = match.group("value").strip()
            comment = (match.group("comment") or "").strip()
            variables.append(EnvVar(name=name, default=value, comment=comment))
    return variables


def load_existing_env(path: Path) -> dict[str, str]:
    """Read an existing ``.env`` file and return its key-value pairs.

    Returns an empty dict when the file does not exist.
    """
    if not path.exists():
        return {}
    return {k: v for k, v in dotenv_values(path).items() if v is not None}


# ── Validation ──────────────────────────────────────────────────────────────


def validate_openrouter_key(key: str) -> bool:
    """Validate an OpenRouter API key via the ``/auth/key`` endpoint.

    Returns ``True`` when the key is accepted, ``False`` otherwise.
    """
    try:
        response = httpx.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {key}"},
            timeout=10,
        )
        return response.status_code == 200
    except httpx.HTTPError:
        return False


# ── Interactive flow ────────────────────────────────────────────────────────


def _ask_provider(
    existing: dict[str, str],
    non_interactive: bool,
    provider_override: str | None,
) -> str:
    """Determine the provider choice (``openrouter`` or ``lmstudio``).

    In non-interactive mode the *provider_override* or the existing value (or
    the default ``openrouter``) is used without prompting.
    """
    default = existing.get("LLM_PROVIDER", "openrouter")
    if provider_override:
        return provider_override
    if non_interactive:
        return default

    console.print("\n[bold]Provider selection[/bold]")
    console.print("  1) [cyan]openrouter[/cyan] — cloud models via OpenRouter + OpenAI embeddings")
    console.print("  2) [cyan]lmstudio[/cyan]   — fully local models via LM Studio")

    choice = typer.prompt(
        "Choose a provider (1 or 2)",
        default="1" if default == "openrouter" else "2",
    ).strip()

    return "lmstudio" if choice == "2" else "openrouter"


def _ask_api_keys(
    provider: str,
    existing: dict[str, str],
    non_interactive: bool,
    api_key_override: str | None,
) -> dict[str, str]:
    """Prompt for API keys required by the selected provider."""
    keys: dict[str, str] = {}

    if provider == "openrouter":
        # -- OpenRouter key ------------------------------------------------
        default_or = existing.get("OPENROUTER_API_KEY", "")
        if api_key_override:
            or_key = api_key_override
        elif non_interactive:
            or_key = default_or
        else:
            or_key = typer.prompt(
                "OpenRouter API key",
                default=default_or or None,
                hide_input=True,
            )

        if or_key:
            keys["OPENROUTER_API_KEY"] = or_key
            # Optionally validate
            if not non_interactive:
                info("Validating OpenRouter key...")
                if validate_openrouter_key(or_key):
                    success("OpenRouter key is valid")
                else:
                    warning("Could not validate key (may still work)")

        # -- OpenAI key (embeddings) ---------------------------------------
        default_oai = existing.get("OPENAI_API_KEY", "")
        if non_interactive:
            oai_key = default_oai
        else:
            oai_key = typer.prompt(
                "OpenAI API key (for embeddings)",
                default=default_oai or None,
                hide_input=True,
            )

        if oai_key:
            keys["OPENAI_API_KEY"] = oai_key

    else:
        # -- LM Studio base URL -------------------------------------------
        default_url = existing.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        if non_interactive:
            lm_url = default_url
        else:
            lm_url = typer.prompt("LM Studio base URL", default=default_url)
        keys["LMSTUDIO_BASE_URL"] = lm_url

    return keys


def interactive_configure(
    env_example: Path = ENV_EXAMPLE,
    env_file: Path = ENV_FILE,
    non_interactive: bool = False,
    provider: str | None = None,
    api_key: str | None = None,
) -> dict[str, str]:
    """Run the interactive .env configuration flow.

    Steps
    -----
    1. Parse ``.env.example`` for variable definitions and defaults.
    2. Load any pre-existing ``.env`` values so they can be offered as
       defaults during prompts.
    3. Ask the user which provider to use (``openrouter`` / ``lmstudio``).
    4. Prompt for relevant API keys (skips keys not needed by the chosen
       provider).
    5. Supabase keys are *always* skipped — they are injected later by the
       ``setup`` command after ``supabase start``.
    6. Build the final values dict, write ``.env``, and return the dict.

    Parameters
    ----------
    env_example:
        Path to the ``.env.example`` template.
    env_file:
        Destination path for the generated ``.env``.
    non_interactive:
        When ``True``, never prompt — use defaults, *provider*, and
        *api_key* arguments directly.
    provider:
        Override provider selection (``"openrouter"`` or ``"lmstudio"``).
    api_key:
        Override the primary API key for the chosen provider (OpenRouter
        API key when ``provider="openrouter"``).

    Returns
    -------
    dict[str, str]
        The final set of environment variables written to ``.env``.
    """
    template_vars = parse_env_example(env_example)
    existing = load_existing_env(env_file)

    # ── 1. Provider selection ────────────────────────────────────────────
    chosen_provider = _ask_provider(existing, non_interactive, provider)

    if chosen_provider == "openrouter":
        provider_values = {
            "LLM_PROVIDER": "openrouter",
            "EMBEDDING_PROVIDER": "openai",
            "VISION_PROVIDER": "openrouter",
        }
    else:
        provider_values = {
            "LLM_PROVIDER": "lmstudio",
            "EMBEDDING_PROVIDER": "lmstudio",
            "VISION_PROVIDER": "lmstudio",
        }

    info(f"Provider set to [cyan]{chosen_provider}[/cyan]")

    # ── 2. API keys ─────────────────────────────────────────────────────
    key_values = _ask_api_keys(chosen_provider, existing, non_interactive, api_key)

    # ── 3. Merge: template defaults → existing .env → interactive input ─
    values: dict[str, str] = {}
    for var in template_vars:
        if var.name in _SUPABASE_KEYS:
            # Preserve any existing Supabase values but never prompt for them
            if var.name in existing:
                values[var.name] = existing[var.name]
            else:
                values[var.name] = var.default
            continue

        # Priority: explicit interactive input > existing > template default
        if var.name in provider_values:
            values[var.name] = provider_values[var.name]
        elif var.name in key_values:
            values[var.name] = key_values[var.name]
        elif var.name in existing:
            values[var.name] = existing[var.name]
        else:
            values[var.name] = var.default

    # ── 4. Write ────────────────────────────────────────────────────────
    write_env(env_file, values, env_example)
    success(f".env written to {env_file}")

    return values


# ── File writer ─────────────────────────────────────────────────────────────


def write_env(path: Path, values: dict[str, str], template_path: Path) -> None:
    """Write a ``.env`` file using ``.env.example`` as the structural template.

    Every line from the template is reproduced as-is, except that
    ``KEY=<template_default>`` lines are replaced with the corresponding value
    from *values*.  Comments, blank lines, and section headers are preserved
    verbatim so the resulting file keeps the same layout as the template.
    """
    lines: list[str] = []
    for raw_line in template_path.read_text(encoding="utf-8").splitlines():
        match = _ENV_LINE_RE.match(raw_line.strip())
        if match:
            name = match.group("name")
            comment_part = match.group("comment") or ""
            new_value = values.get(name, match.group("value").strip())
            if comment_part:
                # Re-align the inline comment roughly where it was
                kv = f"{name}={new_value}"
                # Pad to at least column 42 so comments line up
                padded = kv.ljust(42)
                lines.append(f"{padded}# {comment_part}")
            else:
                lines.append(f"{name}={new_value}")
        else:
            # Comment line, blank line, or section header — keep as-is
            lines.append(raw_line)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

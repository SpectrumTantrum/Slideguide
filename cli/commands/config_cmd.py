"""``slideguide config`` — configuration management commands."""

from __future__ import annotations

from dotenv import dotenv_values
from rich.table import Table

import typer

from cli.config import (
    DEFAULT_BACKEND_PORT,
    DEFAULT_FRONTEND_PORT,
    ENV_EXAMPLE,
    ENV_FILE,
)
from cli.utils.console import console, failure, info, success, warning
from cli.utils.env_builder import (
    interactive_configure,
    load_existing_env,
    validate_openrouter_key,
    write_env,
)
from cli.utils.system import is_port_free

app = typer.Typer(no_args_is_help=True)

# Keys whose values should be partially redacted in output.
_SECRET_SUBSTRINGS = ("KEY", "SECRET", "PASSWORD")


def _redact(key: str, value: str) -> str:
    """Return a redacted version of *value* if *key* looks like a secret."""
    is_secret = any(s in key.upper() for s in _SECRET_SUBSTRINGS)
    if not is_secret:
        return value

    if not value or value.startswith("your_"):
        return "[dim]not set[/dim]"

    if len(value) <= 8:
        return value[:2] + "..." + value[-2:]

    return value[:4] + "..." + value[-4:]


@app.command()
def show() -> None:
    """Display current .env configuration (secrets are redacted)."""
    if not ENV_FILE.exists():
        warning(f".env file not found at {ENV_FILE}")
        info("Run [cyan]slideguide setup[/cyan] to create one.")
        return

    values = dotenv_values(ENV_FILE)

    table = Table(title=".env Configuration", show_lines=False)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value")

    for key, value in values.items():
        display_value = _redact(key, value or "")
        table.add_row(key, display_value)

    console.print()
    console.print(table)
    console.print()


@app.command()
def edit() -> None:
    """Re-run the interactive configuration wizard with current values pre-filled."""
    interactive_configure()


@app.command()
def provider(
    provider_name: str = typer.Argument(
        help="Provider to switch to: 'openrouter' or 'lmstudio'",
    ),
) -> None:
    """Quick-switch all providers (LLM, embedding, vision) at once."""
    provider_name = provider_name.lower().strip()

    if provider_name not in ("openrouter", "lmstudio"):
        failure(f"Unknown provider [bold]{provider_name}[/bold]")
        info("Valid providers: [cyan]openrouter[/cyan], [cyan]lmstudio[/cyan]")
        raise typer.Exit(code=1)

    existing = load_existing_env(ENV_FILE)

    if provider_name == "openrouter":
        existing["LLM_PROVIDER"] = "openrouter"
        existing["EMBEDDING_PROVIDER"] = "openai"
        existing["VISION_PROVIDER"] = "openrouter"
    else:
        existing["LLM_PROVIDER"] = "lmstudio"
        existing["EMBEDDING_PROVIDER"] = "lmstudio"
        existing["VISION_PROVIDER"] = "lmstudio"

    write_env(ENV_FILE, existing, ENV_EXAMPLE)
    success(f"All providers switched to [cyan]{provider_name}[/cyan]")


@app.command()
def validate() -> None:
    """Validate the current configuration (required vars, API keys, ports)."""
    if not ENV_FILE.exists():
        failure(".env file not found")
        info("Run [cyan]slideguide setup[/cyan] to create one.")
        raise typer.Exit(code=1)

    values = load_existing_env(ENV_FILE)
    all_ok = True

    # ── Required variables ──────────────────────────────────────────────────
    info("Checking required environment variables...")
    required_vars = ["SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"]

    for var in required_vars:
        val = values.get(var, "")
        if not val or val.startswith("your_"):
            failure(f"{var} is missing or not configured")
            all_ok = False
        else:
            success(f"{var} is set")

    # ── Provider-specific keys ──────────────────────────────────────────────
    llm_provider = values.get("LLM_PROVIDER", "openrouter")

    if llm_provider == "openrouter":
        info("Checking OpenRouter provider keys...")

        or_key = values.get("OPENROUTER_API_KEY", "")
        if not or_key or or_key.startswith("your_"):
            failure("OPENROUTER_API_KEY is missing or not configured")
            all_ok = False
        else:
            success("OPENROUTER_API_KEY is set")
            info("Validating OpenRouter API key...")
            if validate_openrouter_key(or_key):
                success("OpenRouter API key is valid")
            else:
                warning("Could not validate OpenRouter key (may still work)")

        oai_key = values.get("OPENAI_API_KEY", "")
        if not oai_key or oai_key.startswith("your_"):
            failure("OPENAI_API_KEY is missing (needed for embeddings with OpenRouter)")
            all_ok = False
        else:
            success("OPENAI_API_KEY is set")

    elif llm_provider == "lmstudio":
        info("Checking LM Studio provider keys...")

        lm_url = values.get("LMSTUDIO_BASE_URL", "")
        if not lm_url:
            failure("LMSTUDIO_BASE_URL is missing")
            all_ok = False
        else:
            success(f"LMSTUDIO_BASE_URL is set ({lm_url})")

    # ── Port availability ───────────────────────────────────────────────────
    info("Checking port availability...")

    for name, port in [
        ("Backend", DEFAULT_BACKEND_PORT),
        ("Frontend", DEFAULT_FRONTEND_PORT),
    ]:
        if is_port_free(port):
            success(f"{name} port {port} is available")
        else:
            warning(f"{name} port {port} is in use")

    # ── Summary ─────────────────────────────────────────────────────────────
    console.print()
    if all_ok:
        success("Configuration looks good!")
    else:
        failure("Some configuration issues were found. See above for details.")
        raise typer.Exit(code=1)

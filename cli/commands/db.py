"""``slideguide db`` — database management commands."""

from __future__ import annotations

import typer

from cli.utils.console import info
from cli.utils.supabase import open_studio, reset_database

app = typer.Typer(no_args_is_help=True)


@app.command()
def reset() -> None:
    """Drop and recreate the local Supabase database."""
    reset_database()


@app.command()
def migrate() -> None:
    """Re-run all Supabase migrations (equivalent to ``supabase db reset``).

    This destroys existing data and re-applies every migration file in
    ``supabase/migrations/`` from scratch.
    """
    info("This will re-run all migrations in supabase/migrations/")
    info("Existing data will be destroyed and recreated from migrations.")
    reset_database()


@app.command()
def studio() -> None:
    """Open the Supabase Studio dashboard in your browser."""
    open_studio()

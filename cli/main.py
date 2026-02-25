"""SlideGuide CLI — single entry point for all project commands."""

from __future__ import annotations

import typer

from cli.commands import config_cmd, db, services, setup, status, test

app = typer.Typer(
    name="slideguide",
    help="SlideGuide — AI-Powered Lecture Tutor CLI",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# ── Subcommand groups ───────────────────────────────────────────────────────
app.add_typer(db.app, name="db", help="Database management commands")
app.add_typer(config_cmd.app, name="config", help="Configuration management")

# ── Top-level commands (registered from modules) ───────────────────────────
app.command()(setup.setup)
app.command()(services.start)
app.command()(services.stop)
app.command()(services.restart)
app.command()(services.dev)
app.command()(services.logs)
app.command()(status.status)
app.command()(status.doctor)
app.command()(test.test)


if __name__ == "__main__":
    app()

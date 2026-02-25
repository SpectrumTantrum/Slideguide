"""``slideguide test`` — run the project test suite."""

from __future__ import annotations

import subprocess
import sys
from typing import Annotated

import typer

from cli.config import PROJECT_ROOT
from cli.utils.console import console, info


def test(
    coverage: Annotated[
        bool,
        typer.Option("--coverage", help="Run with coverage"),
    ] = False,
    args: Annotated[
        list[str] | None,
        typer.Argument(help="Additional pytest arguments"),
    ] = None,
) -> None:
    """Run the test suite."""
    cmd = [sys.executable, "-m", "pytest"]

    if coverage:
        cmd.extend(["--cov=backend", "--cov-report=term-missing"])

    if args:
        cmd.extend(args)

    info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
    )

    raise typer.Exit(code=result.returncode)

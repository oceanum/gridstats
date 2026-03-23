"""Command-line interface for gridstats."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="gridstats",
    help="Compute gridded statistics on oceanographic datasets.",
    add_completion=False,
)


@app.command("run")
def run(
    config: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to a YAML pipeline configuration file.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
) -> None:
    """Run a stats pipeline from a YAML configuration file."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    import gridstats  # trigger plugin registration

    from gridstats.pipeline import Pipeline

    Pipeline.from_yaml(config).run()


@app.command("list-stats")
def list_stats() -> None:
    """List all registered stat functions."""
    import gridstats  # trigger plugin registration

    from gridstats.registry import list_stats as _list

    names = _list()
    if names:
        typer.echo("\n".join(names))
    else:
        typer.echo("No stat functions registered.")


def main() -> None:
    """Entry point for the gridstats CLI."""
    app()

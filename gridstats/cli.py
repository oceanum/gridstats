"""Command-line interface for gridstats."""
from __future__ import annotations

from pathlib import Path

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
) -> None:
    """Run a stats pipeline from a YAML configuration file."""
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

"""Command-line interface for gridstats."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="gridstats",
    help="Compute gridded statistics on oceanographic datasets.",
    add_completion=False,
)


#: Environment variable holding an inline YAML config, used when no path is given.
CONFIG_ENV_VAR = "CONFIG"


@app.command("run")
def run(
    config: Optional[Path] = typer.Argument(
        None,
        exists=True,
        readable=True,
        help=(
            "Path to a YAML pipeline configuration file. If omitted, the config "
            f"is read from the ${CONFIG_ENV_VAR} environment variable instead."
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
) -> None:
    """Run a stats pipeline from a YAML config file or the $CONFIG env var.

    Provide either a path argument or, for deployments that inject the config
    as an environment variable (e.g. Argo on k8s), set $CONFIG to the inline
    YAML document. The path argument takes precedence when both are present.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    import gridstats  # trigger plugin registration

    from gridstats.pipeline import Pipeline

    if config is not None:
        pipeline = Pipeline.from_yaml(config)
    else:
        env_config = os.environ.get(CONFIG_ENV_VAR)
        if not env_config:
            raise typer.BadParameter(
                f"No config provided: pass a path argument or set ${CONFIG_ENV_VAR}.",
                param_hint="CONFIG",
            )
        pipeline = Pipeline.from_yaml_string(env_config)

    pipeline.run()


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

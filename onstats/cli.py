"""Console script for onstats."""
import logging.config
import os
import sys

import click
import yaml

from oncore.git import fetch_gitlab_file
from oncore import LOGGING_CONFIG
from onstats.kmzstats_new import KMZ
from onstats.stats import Stats


logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def _load_config(config):
    """Load config from local file, gitlab or env."""
    if os.path.isfile(config):
        instance = yaml.load(open(config), Loader=yaml.Loader)
    elif "gitlab:" in config:
        config = fetch_gitlab_file(config)
        instance = yaml.load(config, Loader=yaml.Loader)
    else:
        instance = yaml.load(config, Loader=yaml.Loader)
    return instance


@click.group()
def main():
    pass


@main.command()
@click.argument("config", envvar="CONFIG")
def gridstats(config):
    """Console script for gridstats."""
    instance = _load_config(config)
    instance()


@main.command()
@click.argument("config", envvar="CONFIG")
@click.option(
    "-o",
    "--outdir",
    help="Output directory for saving files",
    default="./kmz-stats",
    show_default=True,
)
@click.option(
    "-k",
    "--kmzfile",
    help="Output name for kmz",
    default="gridstats.kmz",
    show_default=True,
)
@click.option(
    "-p",
    "--pixels",
    help="Pixels DPI",
    default=512,
    type=int,
    show_default=True,
)
def kmz(config, outdir, kmzfile, pixels):
    """Console script for kmz."""
    kmz = KMZ(config=config, outdir=outdir, kmzfile=kmzfile, pixels=pixels)
    kmz.run()
    return kmz

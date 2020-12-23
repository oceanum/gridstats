"""Console script for onstats."""
import logging
import os
import sys

import click
import yaml

from oncore.git import fetch_gitlab_file
from onstats.kmzstats_new import KMZ
from onstats.stats import Stats

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


@click.group()
def main():
    pass


@main.command()
@click.argument("config", envvar="CONFIG")
@click.option(
    "-l",
    "--load",
    help="Load after each stat call, slower but memory-efficient",
    default=False,
    type=bool,
    show_default=True,
)
def gridstats(config, load):
    """Console script for gridstats."""
    if os.path.isfile(config):
        conf = yaml.load(open(config), Loader=yaml.Loader)
    elif "gitlab:" in config:
        config = fetch_gitlab_file(config)
        conf = yaml.load(config, Loader=yaml.Loader)
    else:
        conf = yaml.load(config, Loader=yaml.Loader)

    stats = Stats(**conf["init"])

    for call in conf.get("calls", []):
        method = call["method"]
        kwargs = call.get("kwargs", {})
        logger.info(f"Stat.{method}({kwargs})")
        getattr(stats, method)(**kwargs)
        if load:
            logger.info(f"Trigerring computation for call: {method}")
            stats._load()
    return stats


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
    "-p", "--pixels", help="Pixels DPI", default=512, type=int, show_default=True,
)
def kmz(config, outdir, kmzfile, pixels):
    """Console script for kmz."""
    kmz = KMZ(config=config, outdir=outdir, kmzfile=kmzfile, pixels=pixels)
    kmz.run()
    return kmz

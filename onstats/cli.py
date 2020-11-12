"""Console script for onstats."""
import sys
import click
import yaml
import logging

from oncore.git import fetch_gitlab_file

from onstats.stats import Stats
from onstats.kmzstats import KMZ


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


@click.group()
def main():
    pass


@main.command()
@click.argument("config")
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
    if config.startswith("gitlab"):
        conf = yaml.load(fetch_gitlab_file(config), Loader=yaml.Loader)
    else:
        conf = yaml.load(open(config), Loader=yaml.Loader)

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
@click.argument("config")
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

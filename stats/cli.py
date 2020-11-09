# -*- coding: utf-8 -*-

"""Console script for stats."""
import sys
import click
import yaml
import logging

from oncore.git import fetch_gitlab_file

from stats.stats import Stats


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


@click.command()
@click.argument("config")
@click.option(
    "-l",
    "--load",
    help="Load after each stat call, slower but memory-efficient",
    default=False,
    type=bool,
    show_default=True,
)
def main(config, load):
    """Console script for stats."""
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

"""Top-level package for gridstats."""

__author__ = "Oceanum Developers"
__email__ = "developers@oceanum.science"
__version__ = "2.0.0"

# Import loaders and ops so their @register_* decorators run, populating
# the registry before any Pipeline or CLI code runs.
import gridstats.loaders  # noqa: F401
import gridstats.ops  # noqa: F401

# Discover and register any third-party plugins declared via entry points.
from gridstats.registry import _load_entrypoint_plugins

_load_entrypoint_plugins()

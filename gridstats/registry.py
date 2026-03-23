"""Registry for stat functions and data loaders.

Third-party packages can extend gridstats without modifying library code by
declaring entry points in their package metadata:

    [project.entry-points."gridstats.stats"]
    my_stat = "my_package.module:my_stat_function"

    [project.entry-points."gridstats.loaders"]
    my_loader = "my_package.module:MyLoaderClass"

Built-in ops and loaders self-register via the @register_stat and
@register_loader decorators when their modules are imported.
"""
from __future__ import annotations

import logging
from typing import Callable

import xarray as xr

logger = logging.getLogger(__name__)

# Type aliases
StatFn = Callable[..., xr.Dataset]
LoaderCls = type

_STATS: dict[str, StatFn] = {}
_LOADERS: dict[str, LoaderCls] = {}


def register_stat(name: str) -> Callable[[StatFn], StatFn]:
    """Decorator to register a stat function under a given name."""

    def decorator(func: StatFn) -> StatFn:
        if name in _STATS:
            logger.warning("Stat '%s' is already registered; overwriting.", name)
        _STATS[name] = func
        return func

    return decorator


def register_loader(name: str) -> Callable[[LoaderCls], LoaderCls]:
    """Decorator to register a data loader class under a given name."""

    def decorator(cls: LoaderCls) -> LoaderCls:
        if name in _LOADERS:
            logger.warning("Loader '%s' is already registered; overwriting.", name)
        _LOADERS[name] = cls
        return cls

    return decorator


def get_stat(name: str) -> StatFn:
    """Return the registered stat function for the given name.

    Raises:
        KeyError: If no stat is registered under that name.
    """
    try:
        return _STATS[name]
    except KeyError:
        available = list(_STATS)
        raise KeyError(
            f"Stat '{name}' not registered. Available: {available}"
        ) from None


def get_loader(name: str) -> LoaderCls:
    """Return the registered loader class for the given name.

    Raises:
        KeyError: If no loader is registered under that name.
    """
    try:
        return _LOADERS[name]
    except KeyError:
        available = list(_LOADERS)
        raise KeyError(
            f"Loader '{name}' not registered. Available: {available}"
        ) from None


def list_stats() -> list[str]:
    """Return sorted names of all registered stat functions."""
    return sorted(_STATS)


def list_loaders() -> list[str]:
    """Return sorted names of all registered loaders."""
    return sorted(_LOADERS)


def _load_entrypoint_plugins() -> None:
    """Discover and register plugins declared via package entry points."""
    from importlib.metadata import entry_points

    for ep in entry_points(group="gridstats.stats"):
        try:
            func = ep.load()
            register_stat(ep.name)(func)
            logger.debug("Loaded stat plugin '%s' from entry point.", ep.name)
        except Exception as exc:
            logger.warning("Failed to load stat plugin '%s': %s", ep.name, exc)

    for ep in entry_points(group="gridstats.loaders"):
        try:
            cls = ep.load()
            register_loader(ep.name)(cls)
            logger.debug("Loaded loader plugin '%s' from entry point.", ep.name)
        except Exception as exc:
            logger.warning("Failed to load loader plugin '%s': %s", ep.name, exc)

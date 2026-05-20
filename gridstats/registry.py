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

import functools
import logging
from typing import Callable

import xarray as xr

logger = logging.getLogger(__name__)

# Type aliases
StatFn = Callable[..., xr.Dataset]
DerivedFn = Callable[..., xr.DataArray]
LoaderCls = type

_STATS: dict[str, StatFn] = {}
_DERIVED: dict[str, DerivedFn] = {}
_LOADERS: dict[str, LoaderCls] = {}


def register_stat(name: str) -> Callable[[StatFn], StatFn]:
    """Decorator to register a stat function under a given name.

    The registered wrapper transparently accepts a ``DataArray`` in addition to
    the standard ``Dataset``: the array is promoted to a single-variable Dataset
    before calling the underlying function, then the result is unwrapped back to
    a ``DataArray`` when the output contains exactly one variable.  When the
    output contains multiple variables (e.g. grouped results that added a month
    dimension are not affected — those stay as Dataset) the Dataset is returned
    as-is.  Dataset inputs pass straight through with no overhead.
    """

    def decorator(func: StatFn) -> StatFn:
        if name in _STATS:
            logger.warning("Stat '%s' is already registered; overwriting.", name)

        @functools.wraps(func)
        def wrapper(data: xr.Dataset | xr.DataArray, *args, **kwargs):
            if isinstance(data, xr.DataArray):
                var_name = data.name or "_var"
                result = func(data.to_dataset(name=var_name), *args, **kwargs)
                if isinstance(result, xr.Dataset) and len(result.data_vars) == 1:
                    return result[next(iter(result.data_vars))]
                return result
            return func(data, *args, **kwargs)

        _STATS[name] = wrapper
        return wrapper

    return decorator


def register_derived(name: str) -> Callable[[DerivedFn], DerivedFn]:
    """Decorator to register a derived variable function under a given name."""

    def decorator(func: DerivedFn) -> DerivedFn:
        if name in _DERIVED:
            logger.warning("Derived '%s' is already registered; overwriting.", name)
        _DERIVED[name] = func
        return func

    return decorator


def get_derived(name: str) -> DerivedFn:
    """Return the registered derived variable function for the given name.

    Raises:
        KeyError: If no derived function is registered under that name.
    """
    try:
        return _DERIVED[name]
    except KeyError:
        available = list(_DERIVED)
        raise KeyError(
            f"Derived variable '{name}' not registered. Available: {available}"
        ) from None


def list_derived() -> list[str]:
    """Return sorted names of all registered derived variable functions."""
    return sorted(_DERIVED)


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

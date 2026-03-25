"""Pydantic configuration models for gridstats."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Source configs — one concrete class per loader type
# ---------------------------------------------------------------------------

class _BaseSourceConfig(BaseModel):
    """Shared pre-processing fields for all source types."""

    mapping: dict[str, str] = {}
    sel: dict[str, Any] = {}
    isel: dict[str, Any] = {}
    chunks: dict[str, int] = {}


class XarraySourceConfig(_BaseSourceConfig):
    """Load data with xarray (local files, cloud storage, any xarray engine)."""

    type: Literal["xarray"]
    urlpath: str
    engine: str = "zarr"
    open_kwargs: dict[str, Any] = {}


class IntakeSourceConfig(_BaseSourceConfig):
    """Load data from an intake-forecast catalog."""

    type: Literal["intake"]
    catalog: str
    dataset_id: str


#: Discriminated union — Pydantic selects the concrete type from the ``type`` field.
SourceConfig = Annotated[
    Union[XarraySourceConfig, IntakeSourceConfig],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Other config models
# ---------------------------------------------------------------------------

class OutputConfig(BaseModel):
    """Configuration for pipeline output."""

    outfile: str
    updir: str | None = None


class ClusterConfig(BaseModel):
    """Dask cluster configuration.

    Defaults are tuned for cloud environments where CPUs are virtual (hyperthreaded).
    ``threads_per_worker=2`` maps two vCPUs to one worker process (one physical core's
    worth of compute), giving fewer workers with more memory each — important for
    memory-intensive operations like quantile.
    """

    enabled: bool = False
    n_workers: int | None = None
    threads_per_worker: int = 2
    processes: bool = True


class CallConfig(BaseModel):
    """Configuration for a single stat call.

    Any extra fields are forwarded as keyword arguments to the stat function.
    """

    model_config = ConfigDict(extra="allow")

    func: str
    data_vars: list[str] | Literal["all"] = "all"
    dim: str = "time"
    group: str | None = None
    chunks: dict[str, int] = {}
    tiles: dict[str, int] = {}
    use_dask_cluster: bool = True
    use_flox: bool = True
    nsector: int | None = None
    dir_var: str = "dpm"
    suffix: str | None = None

    def extra_kwargs(self) -> dict[str, Any]:
        """Return function-specific kwargs (all fields beyond the base schema)."""
        return self.model_extra or {}


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    source: SourceConfig | None = None
    # Placeholder for future multi-source support.
    # When implemented, each named source will map to a separate Zarr group
    # in the output, so that datasets on different grids can coexist without
    # requiring interpolation.
    sources: dict[str, SourceConfig] | None = None
    output: OutputConfig
    calls: list[CallConfig]
    cluster: ClusterConfig = ClusterConfig()
    metadata: dict[str, Any] = {}

    @model_validator(mode="after")
    def _check_sources(self) -> PipelineConfig:
        has_source = self.source is not None
        has_sources = self.sources is not None
        if not has_source and not has_sources:
            raise ValueError("Either 'source' or 'sources' must be provided.")
        if has_source and has_sources:
            raise ValueError("Provide either 'source' or 'sources', not both.")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load and validate a pipeline config from a YAML file."""
        raw = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(raw)

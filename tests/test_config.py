"""Tests for onstats.config."""
import textwrap

import pytest
from pydantic import ValidationError

from onstats.config import (
    CallConfig,
    ClusterConfig,
    IntakeSourceConfig,
    OutputConfig,
    PipelineConfig,
    XarraySourceConfig,
)


class TestXarraySourceConfig:
    def test_basic(self):
        c = XarraySourceConfig(type="xarray", urlpath="gs://bucket/data.zarr")
        assert c.urlpath == "gs://bucket/data.zarr"
        assert c.engine == "zarr"
        assert c.mapping == {}

    def test_custom_engine(self):
        c = XarraySourceConfig(type="xarray", urlpath="data.nc", engine="netcdf4")
        assert c.engine == "netcdf4"

    def test_mapping(self):
        c = XarraySourceConfig(type="xarray", urlpath="data.zarr", mapping={"tps": "tp"})
        assert c.mapping == {"tps": "tp"}

    def test_slice_dict(self):
        c = XarraySourceConfig(type="xarray", urlpath="data.zarr", slice_dict={"sel": {"latitude": 0}})
        assert "sel" in c.slice_dict

    def test_chunks(self):
        c = XarraySourceConfig(type="xarray", urlpath="data.zarr", chunks={"time": 100})
        assert c.chunks == {"time": 100}

    def test_missing_urlpath_raises(self):
        with pytest.raises(ValidationError):
            XarraySourceConfig(type="xarray")


class TestIntakeSourceConfig:
    def test_basic(self):
        c = IntakeSourceConfig(
            type="intake", catalog="gs://bucket/catalog.yml", dataset_id="wave_nz"
        )
        assert c.catalog == "gs://bucket/catalog.yml"
        assert c.dataset_id == "wave_nz"

    def test_missing_catalog_raises(self):
        with pytest.raises(ValidationError):
            IntakeSourceConfig(type="intake", dataset_id="wave_nz")

    def test_missing_dataset_id_raises(self):
        with pytest.raises(ValidationError):
            IntakeSourceConfig(type="intake", catalog="gs://bucket/catalog.yml")


class TestSourceConfigDiscriminator:
    """The SourceConfig discriminated union selects the right type from 'type'."""

    def test_xarray_via_pipeline(self):
        config = PipelineConfig(
            source={"type": "xarray", "urlpath": "data.zarr"},
            output={"outfile": "out.zarr"},
            calls=[{"func": "mean"}],
        )
        assert isinstance(config.source, XarraySourceConfig)
        assert config.source.urlpath == "data.zarr"

    def test_intake_via_pipeline(self):
        config = PipelineConfig(
            source={"type": "intake", "catalog": "cat.yml", "dataset_id": "wave_nz"},
            output={"outfile": "out.zarr"},
            calls=[{"func": "mean"}],
        )
        assert isinstance(config.source, IntakeSourceConfig)
        assert config.source.dataset_id == "wave_nz"

    def test_unknown_type_raises(self):
        with pytest.raises(ValidationError):
            PipelineConfig(
                source={"type": "unknown", "urlpath": "data.zarr"},
                output={"outfile": "out.zarr"},
                calls=[{"func": "mean"}],
            )

    def test_missing_type_raises(self):
        with pytest.raises(ValidationError):
            PipelineConfig(
                source={"urlpath": "data.zarr"},
                output={"outfile": "out.zarr"},
                calls=[{"func": "mean"}],
            )


class TestCallConfig:
    def test_defaults(self):
        c = CallConfig(func="mean")
        assert c.dim == "time"
        assert c.data_vars == "all"
        assert c.group is None
        assert c.chunks == {}
        assert c.tiles == {}
        assert c.use_dask_cluster is True
        assert c.nsector is None
        assert c.dir_var == "dpm"
        assert c.suffix is None

    def test_extra_kwargs(self):
        c = CallConfig(func="quantile", q=[0.5, 0.95], data_vars=["hs"])
        assert c.extra_kwargs() == {"q": [0.5, 0.95]}

    def test_no_extra_kwargs(self):
        c = CallConfig(func="mean", dim="time")
        assert c.extra_kwargs() == {}

    def test_data_vars_list(self):
        c = CallConfig(func="mean", data_vars=["hs", "tp"])
        assert c.data_vars == ["hs", "tp"]

    def test_tiles(self):
        c = CallConfig(func="rpv", tiles={"latitude": 20, "longitude": 20})
        assert c.tiles == {"latitude": 20, "longitude": 20}

    def test_nsector(self):
        c = CallConfig(func="mean", nsector=8, dir_var="dpm")
        assert c.nsector == 8
        assert c.dir_var == "dpm"

    def test_suffix(self):
        c = CallConfig(func="mean", suffix="_custom")
        assert c.suffix == "_custom"


class TestClusterConfig:
    def test_defaults(self):
        c = ClusterConfig()
        assert c.enabled is False
        assert c.n_workers is None
        assert c.threads_per_worker == 1
        assert c.processes is True

    def test_enabled(self):
        c = ClusterConfig(enabled=True, n_workers=4)
        assert c.enabled is True
        assert c.n_workers == 4


class TestPipelineConfig:
    def test_basic(self):
        config = PipelineConfig(
            source={"type": "xarray", "urlpath": "data.zarr"},
            output={"outfile": "out.zarr"},
            calls=[{"func": "mean"}],
        )
        assert config.source.urlpath == "data.zarr"
        assert config.output.outfile == "out.zarr"
        assert len(config.calls) == 1

    def test_requires_source_or_sources(self):
        with pytest.raises(ValidationError, match="Either 'source' or 'sources'"):
            PipelineConfig(
                output={"outfile": "out.zarr"},
                calls=[{"func": "mean"}],
            )

    def test_no_both_sources(self):
        with pytest.raises(ValidationError, match="not both"):
            PipelineConfig(
                source={"type": "xarray", "urlpath": "data.zarr"},
                sources={"wave": {"type": "xarray", "urlpath": "waves.zarr"}},
                output={"outfile": "out.zarr"},
                calls=[{"func": "mean"}],
            )

    def test_sources_placeholder(self):
        config = PipelineConfig(
            sources={"wave": {"type": "xarray", "urlpath": "waves.zarr"}},
            output={"outfile": "out.zarr"},
            calls=[{"func": "mean"}],
        )
        assert "wave" in config.sources
        assert isinstance(config.sources["wave"], XarraySourceConfig)

    def test_cluster_defaults(self):
        config = PipelineConfig(
            source={"type": "xarray", "urlpath": "data.zarr"},
            output={"outfile": "out.zarr"},
            calls=[{"func": "mean"}],
        )
        assert config.cluster.enabled is False

    def test_from_yaml(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            source:
              type: xarray
              urlpath: gs://bucket/data.zarr
              engine: zarr
              mapping:
                tps: tp
            output:
              outfile: ./results.zarr
            calls:
              - func: mean
                dim: time
                data_vars: [hs, tp]
              - func: quantile
                dim: time
                q: [0.5, 0.95]
                data_vars: [hs]
        """)
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml_content)

        config = PipelineConfig.from_yaml(config_file)

        assert isinstance(config.source, XarraySourceConfig)
        assert config.source.urlpath == "gs://bucket/data.zarr"
        assert config.source.mapping == {"tps": "tp"}
        assert len(config.calls) == 2
        assert config.calls[0].func == "mean"
        assert config.calls[1].extra_kwargs() == {"q": [0.5, 0.95]}

    def test_from_yaml_intake(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            source:
              type: intake
              catalog: /catalogs/oceanum.yaml
              dataset_id: wave_nz
            output:
              outfile: ./results.zarr
            calls:
              - func: mean
        """)
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml_content)

        config = PipelineConfig.from_yaml(config_file)
        assert isinstance(config.source, IntakeSourceConfig)
        assert config.source.catalog == "/catalogs/oceanum.yaml"
        assert config.source.dataset_id == "wave_nz"

    def test_metadata(self):
        config = PipelineConfig(
            source={"type": "xarray", "urlpath": "data.zarr"},
            output={"outfile": "out.zarr"},
            calls=[{"func": "mean"}],
            metadata={"institution": "Test"},
        )
        assert config.metadata == {"institution": "Test"}

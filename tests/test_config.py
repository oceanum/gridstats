"""Tests for onstats.config."""
import textwrap

import pytest
from pydantic import ValidationError

from onstats.config import (
    CallConfig,
    ClusterConfig,
    OutputConfig,
    PipelineConfig,
    SourceConfig,
)


class TestSourceConfig:
    def test_urlpath(self):
        c = SourceConfig(urlpath="gs://bucket/data.zarr")
        assert c.urlpath == "gs://bucket/data.zarr"
        assert c.engine == "zarr"
        assert c.mapping == {}

    def test_catalog(self):
        c = SourceConfig(catalog="gs://bucket/catalog.yml", dataset_id="wave_nz")
        assert c.catalog == "gs://bucket/catalog.yml"
        assert c.dataset_id == "wave_nz"

    def test_missing_source_raises(self):
        with pytest.raises(ValidationError, match="Provide either"):
            SourceConfig()

    def test_catalog_without_dataset_id_raises(self):
        with pytest.raises(ValidationError, match="Provide either"):
            SourceConfig(catalog="gs://bucket/catalog.yml")

    def test_both_urlpath_and_catalog_raises(self):
        with pytest.raises(ValidationError, match="not both"):
            SourceConfig(
                urlpath="gs://bucket/data.zarr",
                catalog="gs://bucket/catalog.yml",
                dataset_id="wave_nz",
            )

    def test_mapping(self):
        c = SourceConfig(urlpath="data.zarr", mapping={"tps": "tp"})
        assert c.mapping == {"tps": "tp"}

    def test_slice_dict(self):
        c = SourceConfig(urlpath="data.zarr", slice_dict={"sel": {"latitude": 0}})
        assert "sel" in c.slice_dict

    def test_chunks(self):
        c = SourceConfig(urlpath="data.zarr", chunks={"time": 100})
        assert c.chunks == {"time": 100}


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
            source={"urlpath": "data.zarr"},
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
                source={"urlpath": "data.zarr"},
                sources={"wave": {"urlpath": "waves.zarr"}},
                output={"outfile": "out.zarr"},
                calls=[{"func": "mean"}],
            )

    def test_sources_placeholder(self):
        config = PipelineConfig(
            sources={"wave": {"urlpath": "waves.zarr"}},
            output={"outfile": "out.zarr"},
            calls=[{"func": "mean"}],
        )
        assert "wave" in config.sources

    def test_cluster_defaults(self):
        config = PipelineConfig(
            source={"urlpath": "data.zarr"},
            output={"outfile": "out.zarr"},
            calls=[{"func": "mean"}],
        )
        assert config.cluster.enabled is False

    def test_from_yaml(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            source:
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

        assert config.source.urlpath == "gs://bucket/data.zarr"
        assert config.source.mapping == {"tps": "tp"}
        assert len(config.calls) == 2
        assert config.calls[0].func == "mean"
        assert config.calls[1].extra_kwargs() == {"q": [0.5, 0.95]}

    def test_metadata(self):
        config = PipelineConfig(
            source={"urlpath": "data.zarr"},
            output={"outfile": "out.zarr"},
            calls=[{"func": "mean"}],
            metadata={"institution": "Test"},
        )
        assert config.metadata == {"institution": "Test"}

"""Tests for onstats.pipeline."""
import textwrap

import numpy as np
import pytest
import xarray as xr

import onstats.loaders  # ensure loaders are registered
import onstats.ops  # ensure ops are registered


@pytest.fixture
def sample_ds():
    rng = np.random.default_rng(0)
    nt, ny, nx = 24 * 365, 3, 3
    return xr.Dataset(
        {
            "hs": (
                ["time", "latitude", "longitude"],
                (rng.random((nt, ny, nx)) * 3 + 0.5).astype("float32"),
            ),
            "tp": (
                ["time", "latitude", "longitude"],
                (rng.random((nt, ny, nx)) * 8 + 4).astype("float32"),
            ),
        },
        coords={
            "time": xr.date_range("2020-01-01", periods=nt, freq="1h"),
            "latitude": [-40.0, -39.0, -38.0],
            "longitude": [170.0, 171.0, 172.0],
        },
    )


@pytest.fixture
def netcdf_source(tmp_path, sample_ds):
    path = tmp_path / "source.nc"
    sample_ds.to_netcdf(path)
    return str(path)


@pytest.fixture
def pipeline_config(tmp_path, netcdf_source):
    from onstats.config import CallConfig, OutputConfig, PipelineConfig, XarraySourceConfig

    return PipelineConfig(
        source=XarraySourceConfig(type="xarray", urlpath=netcdf_source, engine="netcdf4"),
        output=OutputConfig(outfile=str(tmp_path / "out.nc")),
        calls=[
            CallConfig(func="mean", dim="time", data_vars=["hs"]),
        ],
    )


class TestPipeline:
    def test_run_produces_output_file(self, pipeline_config):
        from onstats.pipeline import Pipeline

        p = Pipeline(pipeline_config)
        dsout = p.run()
        assert "hs_mean" in dsout
        from pathlib import Path

        assert Path(pipeline_config.output.outfile).exists()

    def test_run_variable_suffix(self, pipeline_config):
        from onstats.pipeline import Pipeline

        dsout = Pipeline(pipeline_config).run()
        assert "hs_mean" in dsout

    def test_run_custom_suffix(self, tmp_path, netcdf_source):
        from onstats.config import CallConfig, OutputConfig, PipelineConfig, XarraySourceConfig
        from onstats.pipeline import Pipeline

        config = PipelineConfig(
            source=XarraySourceConfig(type="xarray", urlpath=netcdf_source, engine="netcdf4"),
            output=OutputConfig(outfile=str(tmp_path / "out.nc")),
            calls=[CallConfig(func="mean", dim="time", data_vars=["hs"], suffix="_avg")],
        )
        dsout = Pipeline(config).run()
        assert "hs_avg" in dsout

    def test_run_multiple_calls(self, tmp_path, netcdf_source):
        from onstats.config import CallConfig, OutputConfig, PipelineConfig, XarraySourceConfig
        from onstats.pipeline import Pipeline

        config = PipelineConfig(
            source=XarraySourceConfig(type="xarray", urlpath=netcdf_source, engine="netcdf4"),
            output=OutputConfig(outfile=str(tmp_path / "out.nc")),
            calls=[
                CallConfig(func="mean", dim="time", data_vars=["hs"]),
                CallConfig(func="max", dim="time", data_vars=["hs"]),
            ],
        )
        dsout = Pipeline(config).run()
        assert "hs_mean" in dsout
        assert "hs_max" in dsout

    def test_run_group_monthly(self, tmp_path, netcdf_source):
        from onstats.config import CallConfig, OutputConfig, PipelineConfig, XarraySourceConfig
        from onstats.pipeline import Pipeline

        config = PipelineConfig(
            source=XarraySourceConfig(type="xarray", urlpath=netcdf_source, engine="netcdf4"),
            output=OutputConfig(outfile=str(tmp_path / "out.nc")),
            calls=[CallConfig(func="mean", dim="time", data_vars=["hs"], group="month")],
        )
        dsout = Pipeline(config).run()
        assert "hs_mean_month" in dsout
        assert "month" in dsout["hs_mean_month"].dims

    def test_run_zarr_output(self, tmp_path, netcdf_source):
        from onstats.config import CallConfig, OutputConfig, PipelineConfig, XarraySourceConfig
        from onstats.pipeline import Pipeline

        config = PipelineConfig(
            source=XarraySourceConfig(type="xarray", urlpath=netcdf_source, engine="netcdf4"),
            output=OutputConfig(outfile=str(tmp_path / "out.zarr")),
            calls=[CallConfig(func="mean", dim="time", data_vars=["hs"])],
        )
        Pipeline(config).run()
        import os

        assert os.path.isdir(str(tmp_path / "out.zarr"))

    def test_from_yaml(self, tmp_path, netcdf_source):
        from onstats.pipeline import Pipeline

        yaml_content = textwrap.dedent(f"""\
            source:
              type: xarray
              urlpath: {netcdf_source}
              engine: netcdf4
            output:
              outfile: {tmp_path}/out.nc
            calls:
              - func: mean
                dim: time
                data_vars: [hs]
        """)
        cfg_file = tmp_path / "config.yml"
        cfg_file.write_text(yaml_content)
        dsout = Pipeline.from_yaml(cfg_file).run()
        assert "hs_mean" in dsout

    def test_multi_source_raises_not_implemented(self, tmp_path):
        from onstats.config import CallConfig, OutputConfig, PipelineConfig, XarraySourceConfig
        from onstats.pipeline import Pipeline

        config = PipelineConfig(
            sources={"wave": XarraySourceConfig(type="xarray", urlpath="waves.zarr")},
            output=OutputConfig(outfile=str(tmp_path / "out.nc")),
            calls=[CallConfig(func="mean")],
        )
        with pytest.raises(NotImplementedError, match="Multi-source"):
            Pipeline(config).run()

    def test_apply_tiled(self, tmp_path, netcdf_source):
        from onstats.config import CallConfig, OutputConfig, PipelineConfig, XarraySourceConfig
        from onstats.pipeline import Pipeline

        config = PipelineConfig(
            source=XarraySourceConfig(type="xarray", urlpath=netcdf_source, engine="netcdf4"),
            output=OutputConfig(outfile=str(tmp_path / "out.nc")),
            calls=[
                CallConfig(
                    func="mean",
                    dim="time",
                    data_vars=["hs"],
                    tiles={"latitude": 2},
                )
            ],
        )
        dsout = Pipeline(config).run()
        assert "hs_mean" in dsout
        # Should cover the same grid as without tiling
        assert dsout["hs_mean"].latitude.size == 3

"""Tests for onstats.loaders."""
import numpy as np
import pytest
import xarray as xr

from onstats.config import XarraySourceConfig
from onstats.loaders.xarray import XarrayLoader


@pytest.fixture
def sample_dataset():
    return xr.Dataset(
        {
            "hs": (
                ["time", "latitude", "longitude"],
                np.random.rand(10, 5, 5).astype("float32"),
            ),
            "tps": (
                ["time", "latitude", "longitude"],
                np.random.rand(10, 5, 5).astype("float32"),
            ),
        },
        coords={
            "time": xr.date_range("2020-01-01", periods=10, freq="1h"),
            "latitude": np.linspace(-40, -35, 5),
            "longitude": np.linspace(170, 175, 5),
        },
    )


@pytest.fixture
def netcdf_file(tmp_path, sample_dataset):
    path = tmp_path / "test.nc"
    sample_dataset.to_netcdf(path)
    return path


@pytest.fixture
def zarr_store(tmp_path, sample_dataset):
    path = tmp_path / "test.zarr"
    sample_dataset.to_zarr(path)
    return str(path)


class TestXarrayLoader:
    def test_load_netcdf(self, netcdf_file):
        config = XarraySourceConfig(type="xarray", urlpath=str(netcdf_file), engine="netcdf4")
        dset = XarrayLoader().load(config)
        assert "hs" in dset
        assert "tps" in dset

    def test_load_zarr(self, zarr_store):
        config = XarraySourceConfig(type="xarray", urlpath=zarr_store, engine="zarr")
        dset = XarrayLoader().load(config)
        assert "hs" in dset

    def test_mapping_renames_variable(self, netcdf_file):
        config = XarraySourceConfig(
            type="xarray", urlpath=str(netcdf_file), engine="netcdf4", mapping={"tps": "tp"}
        )
        dset = XarrayLoader().load(config)
        assert "tp" in dset
        assert "tps" not in dset

    def test_missing_mapping_key_ignored(self, netcdf_file):
        config = XarraySourceConfig(
            type="xarray", urlpath=str(netcdf_file), engine="netcdf4", mapping={"nonexistent": "tp"}
        )
        dset = XarrayLoader().load(config)
        assert "hs" in dset

    def test_slice_dict_applied(self, netcdf_file, sample_dataset):
        lat0 = float(sample_dataset.latitude[1])
        lat1 = float(sample_dataset.latitude[3])
        config = XarraySourceConfig(
            type="xarray",
            urlpath=str(netcdf_file),
            engine="netcdf4",
            slice_dict={"sel": {"latitude": slice(lat0, lat1)}},
        )
        dset = XarrayLoader().load(config)
        assert dset.latitude.size < sample_dataset.latitude.size

    def test_chunks_applied(self, netcdf_file):
        config = XarraySourceConfig(
            type="xarray",
            urlpath=str(netcdf_file),
            engine="netcdf4",
            chunks={"time": 5},
        )
        dset = XarrayLoader().load(config)
        assert dset.chunks["time"] == (5, 5)

    def test_open_kwargs_forwarded(self, netcdf_file):
        config = XarraySourceConfig(
            type="xarray",
            urlpath=str(netcdf_file),
            engine="netcdf4",
            open_kwargs={"decode_times": False},
        )
        dset = XarrayLoader().load(config)
        assert "hs" in dset

    def test_dataset_is_lazy(self, netcdf_file):
        config = XarraySourceConfig(
            type="xarray", urlpath=str(netcdf_file), engine="netcdf4", chunks={"time": 10}
        )
        dset = XarrayLoader().load(config)
        import dask.array as da

        assert isinstance(dset["hs"].data, da.Array)

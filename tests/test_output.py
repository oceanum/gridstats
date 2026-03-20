"""Tests for onstats.output."""
import numpy as np
import pytest
import xarray as xr

from onstats.output import (
    finalise,
    set_global_attributes,
    set_variable_attributes,
    write,
    write_netcdf,
    write_zarr,
)


@pytest.fixture
def source_ds():
    return xr.Dataset(
        coords={"time": xr.date_range("2020-01-01", periods=3, freq="1h")}
    )


@pytest.fixture
def dsout():
    return xr.Dataset(
        {
            "hs_mean": (
                ["latitude", "longitude"],
                np.random.rand(3, 3).astype("float32"),
            ),
            "tp_max": (
                ["latitude", "longitude"],
                np.random.rand(3, 3).astype("float32"),
            ),
        },
        coords={
            "latitude": [-40.0, -39.0, -38.0],
            "longitude": [170.0, 171.0, 172.0],
        },
    )


class TestSetVariableAttributes:
    def test_known_variable_gets_attrs(self, dsout):
        out = set_variable_attributes(dsout)
        assert "standard_name" in out["hs_mean"].attrs
        assert "units" in out["hs_mean"].attrs

    def test_unknown_variable_no_error(self):
        ds = xr.Dataset({"foo_bar": (["x"], [1.0])})
        out = set_variable_attributes(ds)  # should not raise
        assert "foo_bar" in out

    def test_extra_metadata_applied(self, dsout):
        extra = {"data_vars": {"hs": {"standard_name": "custom_hs", "units": "ft"}}}
        out = set_variable_attributes(dsout, extra)
        assert "custom_hs" in out["hs_mean"].attrs["standard_name"]


class TestSetGlobalAttributes:
    def test_sets_institution(self, source_ds, dsout):
        out = set_global_attributes(source_ds, dsout)
        assert out.attrs["institution"] == "Oceanum"

    def test_sets_time_coverage(self, source_ds, dsout):
        out = set_global_attributes(source_ds, dsout)
        assert "time_coverage_start" in out.attrs
        assert "time_coverage_end" in out.attrs


class TestFinalise:
    def test_sorts_descending_dims(self, source_ds):
        ds = xr.Dataset(
            {"hs_mean": (["latitude"], [1.0, 2.0, 3.0])},
            coords={"latitude": [10.0, 5.0, 0.0]},
        )
        out = finalise(ds, source_ds)
        assert list(out.latitude.values) == [0.0, 5.0, 10.0]

    def test_float64_cast_to_float32(self, source_ds):
        ds = xr.Dataset({"hs_mean": (["x"], np.array([1.0, 2.0], dtype="float64"))})
        out = finalise(ds, source_ds)
        assert out["hs_mean"].dtype == np.float32

    def test_chunks_applied(self, source_ds, dsout):
        out = finalise(dsout, source_ds, chunks={"latitude": 2})
        assert out.chunks["latitude"] == (2, 1)


class TestWriters:
    def test_write_netcdf(self, tmp_path, dsout, source_ds):
        path = str(tmp_path / "out.nc")
        write_netcdf(dsout, path)
        loaded = xr.open_dataset(path)
        assert "hs_mean" in loaded

    def test_write_zarr(self, tmp_path, dsout, source_ds):
        path = str(tmp_path / "out.zarr")
        write_zarr(dsout, path)
        loaded = xr.open_zarr(path)
        assert "hs_mean" in loaded

    def test_write_dispatch_nc(self, tmp_path, dsout):
        path = str(tmp_path / "out.nc")
        write(dsout, path)
        assert (tmp_path / "out.nc").exists()

    def test_write_dispatch_zarr(self, tmp_path, dsout):
        path = str(tmp_path / "out.zarr")
        write(dsout, path)
        assert (tmp_path / "out.zarr").is_dir()

    def test_write_bad_extension_raises(self, dsout):
        with pytest.raises(ValueError, match="'.nc' or '.zarr'"):
            write(dsout, "output.hdf5")

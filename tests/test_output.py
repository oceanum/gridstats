"""Tests for gridstats.output."""
import numpy as np
import pytest
import xarray as xr

from gridstats.config import NotnullMaskConfig, ThresholdMaskConfig
from gridstats.output import (
    _build_mask,
    finalise,
    set_global_attributes,
    set_variable_attributes,
    upload,
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

    def test_extra_attrs_override_defaults(self, source_ds, dsout):
        out = set_global_attributes(source_ds, dsout, extra_attrs={"institution": "NIWA", "project": "MyProject"})
        assert out.attrs["institution"] == "NIWA"
        assert out.attrs["project"] == "MyProject"
        assert out.attrs["source"] == "gridstats"  # default still present


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


class TestBuildMask:
    @pytest.fixture
    def spatial_source(self):
        data = np.full((3, 3, 3), np.nan)
        data[:, 1:, 1:] = 1.0  # non-null inland; first row/col stays NaN
        return xr.Dataset(
            {"hs": (["time", "latitude", "longitude"], data)},
            coords={
                "time": [0, 1, 2],
                "latitude": [-40.0, -39.0, -38.0],
                "longitude": [170.0, 171.0, 172.0],
            },
        )

    def test_notnull_no_isel(self, spatial_source):
        cfg = NotnullMaskConfig(type="notnull", var="hs")
        mask = _build_mask(spatial_source, cfg)
        assert mask.dims == ("time", "latitude", "longitude")
        assert mask.dtype == bool

    def test_notnull_with_isel_reduces_dims(self, spatial_source):
        cfg = NotnullMaskConfig(type="notnull", var="hs", isel={"time": 0})
        mask = _build_mask(spatial_source, cfg)
        assert "time" not in mask.dims
        assert mask.shape == (3, 3)

    def test_notnull_correct_values(self, spatial_source):
        cfg = NotnullMaskConfig(type="notnull", var="hs", isel={"time": 0})
        mask = _build_mask(spatial_source, cfg)
        assert not mask.values[0, 0]   # NaN point masked
        assert mask.values[1, 1]       # valid point kept

    def test_threshold_gt(self, spatial_source):
        cfg = ThresholdMaskConfig(type="threshold", var="hs", isel={"time": 1}, operator="gt", value=0.5)
        mask = _build_mask(spatial_source, cfg)
        assert mask.shape == (3, 3)
        assert mask.values[1, 1]       # 1.0 > 0.5
        assert not mask.values[0, 0]   # NaN > 0.5 is False

    def test_threshold_lt(self, spatial_source):
        cfg = ThresholdMaskConfig(type="threshold", var="hs", isel={"time": 1}, operator="lt", value=2.0)
        mask = _build_mask(spatial_source, cfg)
        assert mask.values[1, 1]       # 1.0 < 2.0


class TestFinaliseWithMask:
    @pytest.fixture
    def source_with_nans(self):
        data = np.array([[[1.0, np.nan], [np.nan, 1.0]]])
        return xr.Dataset(
            {"hs": (["time", "latitude", "longitude"], data)},
            coords={
                "time": [0],
                "latitude": [-40.0, -39.0],
                "longitude": [170.0, 171.0],
            },
        )

    def test_notnull_mask_applied(self, source_with_nans):
        dsout = xr.Dataset(
            {"hs_mean": (["latitude", "longitude"], np.ones((2, 2), dtype="float32"))},
            coords={"latitude": [-40.0, -39.0], "longitude": [170.0, 171.0]},
        )
        cfg = NotnullMaskConfig(type="notnull", var="hs", isel={"time": 0})
        out = finalise(dsout, source_with_nans, mask_config=cfg)
        assert not np.isnan(out["hs_mean"].values[0, 0])   # hs=1.0 → kept
        assert np.isnan(out["hs_mean"].values[0, 1])        # hs=NaN → masked

    def test_no_mask_leaves_data_unchanged(self, source_with_nans):
        dsout = xr.Dataset(
            {"hs_mean": (["latitude", "longitude"], np.ones((2, 2), dtype="float32"))},
            coords={"latitude": [-40.0, -39.0], "longitude": [170.0, 171.0]},
        )
        out = finalise(dsout, source_with_nans)
        assert not np.any(np.isnan(out["hs_mean"].values))

    def test_mask_broadcasts_over_extra_dims(self, source_with_nans):
        dsout = xr.Dataset(
            {
                "hs_mean": (
                    ["time", "latitude", "longitude"],
                    np.ones((5, 2, 2), dtype="float32"),
                )
            },
            coords={
                "time": range(5),
                "latitude": [-40.0, -39.0],
                "longitude": [170.0, 171.0],
            },
        )
        cfg = NotnullMaskConfig(type="notnull", var="hs", isel={"time": 0})
        out = finalise(dsout, source_with_nans, mask_config=cfg)
        # All time steps at the NaN location should be masked
        assert np.all(np.isnan(out["hs_mean"].values[:, 0, 1]))


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

    def test_write_zarr_append_creates_store_when_missing(self, tmp_path, dsout):
        path = str(tmp_path / "out.zarr")
        write_zarr(dsout, path, append=True)
        loaded = xr.open_zarr(path)
        assert "hs_mean" in loaded

    def test_write_zarr_append_adds_new_variable(self, tmp_path, dsout):
        path = str(tmp_path / "out.zarr")
        ds1 = dsout[["hs_mean"]]
        ds2 = dsout[["tp_max"]]
        write_zarr(ds1, path, append=True)
        write_zarr(ds2, path, append=True)
        loaded = xr.open_zarr(path, consolidated=False)
        assert "hs_mean" in loaded
        assert "tp_max" in loaded

    def test_write_zarr_append_overwrites_existing_variable(self, tmp_path):
        path = str(tmp_path / "out.zarr")
        ds1 = xr.Dataset(
            {"hs_mean": (["x"], np.array([1.0, 2.0], dtype="float32"))},
            coords={"x": [0, 1]},
        )
        ds2 = xr.Dataset(
            {"hs_mean": (["x"], np.array([9.0, 8.0], dtype="float32"))},
            coords={"x": [0, 1]},
        )
        write_zarr(ds1, path, append=True)
        write_zarr(ds2, path, append=True)
        loaded = xr.open_zarr(path, consolidated=False)
        np.testing.assert_array_equal(loaded["hs_mean"].values, [9.0, 8.0])

    def test_write_zarr_consolidate(self, tmp_path, dsout):
        path = str(tmp_path / "out.zarr")
        write_zarr(dsout, path, consolidate=True)
        import zarr
        store = zarr.open_consolidated(path)
        assert "hs_mean" in store


class TestUpload:
    def test_upload_zarr_dir(self, tmp_path, dsout):
        src = str(tmp_path / "out.zarr")
        write_zarr(dsout, src)
        updir = tmp_path / "remote"
        dest = upload(src, str(updir))
        assert dest == str(updir / "out.zarr")
        loaded = xr.open_zarr(dest)
        assert "hs_mean" in loaded

    def test_upload_netcdf_file(self, tmp_path, dsout):
        src = str(tmp_path / "out.nc")
        write_netcdf(dsout, src)
        updir = tmp_path / "remote"
        dest = upload(src, str(updir))
        assert dest == str(updir / "out.nc")
        loaded = xr.open_dataset(dest)
        assert "hs_mean" in loaded

    def test_upload_strips_trailing_slash(self, tmp_path, dsout):
        src = str(tmp_path / "out.zarr")
        write_zarr(dsout, src)
        dest = upload(src, str(tmp_path / "remote") + "/")
        assert dest == str(tmp_path / "remote" / "out.zarr")

    def test_upload_skips_remote_outfile(self, tmp_path):
        dest = upload("gs://bucket/stats/out.zarr", str(tmp_path / "remote"))
        assert dest == "gs://bucket/stats/out.zarr"

    def test_upload_missing_source_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            upload(str(tmp_path / "nope.zarr"), str(tmp_path / "remote"))

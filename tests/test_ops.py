"""Tests for gridstats.ops."""
import numpy as np
import pytest
import xarray as xr

import gridstats.ops  # trigger registration


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def ds():
    """Small dataset (10 timesteps, 3×3 grid) used across all tests."""
    rng = np.random.default_rng(42)
    nt, ny, nx = 24 * 365, 3, 3  # 1 year hourly
    return xr.Dataset(
        {
            "hs": (
                ["time", "latitude", "longitude"],
                (rng.random((nt, ny, nx)) * 4 + 0.5).astype("float32"),
            ),
            "tp": (
                ["time", "latitude", "longitude"],
                (rng.random((nt, ny, nx)) * 10 + 4).astype("float32"),
            ),
            "dpm": (
                ["time", "latitude", "longitude"],
                (rng.random((nt, ny, nx)) * 360).astype("float32"),
            ),
        },
        coords={
            "time": xr.date_range("2020-01-01", periods=nt, freq="1h"),
            "latitude": [-40.0, -39.0, -38.0],
            "longitude": [170.0, 171.0, 172.0],
        },
    )


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------

class TestAggregations:
    def test_mean(self, ds):
        from gridstats.ops.aggregations import mean
        out = mean(ds, dim="time")
        assert "hs" in out
        assert "time" not in out.dims

    def test_mean_group(self, ds):
        from gridstats.ops.aggregations import mean
        out = mean(ds[["hs"]], dim="time", group="month")
        assert "month" in out.dims
        assert out.month.size == 12

    def test_max(self, ds):
        from gridstats.ops.aggregations import max
        out = max(ds, dim="time")
        assert float(out["hs"].min()) >= float(ds["hs"].min())

    def test_min(self, ds):
        from gridstats.ops.aggregations import min
        out = min(ds, dim="time")
        assert float(out["hs"].max()) <= float(ds["hs"].max())

    def test_std(self, ds):
        from gridstats.ops.aggregations import std
        out = std(ds, dim="time")
        assert "hs" in out
        assert float(out["hs"].min()) >= 0

    def test_count(self, ds):
        from gridstats.ops.aggregations import count
        out = count(ds, dim="time")
        assert int(out["hs"].max()) == ds.time.size

    def test_quantile(self, ds):
        from gridstats.ops.aggregations import quantile
        out = quantile(ds[["hs"]], dim="time", q=[0.5, 0.95])
        assert "quantile" in out.dims
        assert out["quantile"].size == 2

    def test_pcount_full_data(self, ds):
        from gridstats.ops.aggregations import pcount
        out = pcount(ds, dim="time")
        assert float(out["hs"].min()) == pytest.approx(100.0)

    def test_pcount_with_nans(self, ds):
        from gridstats.ops.aggregations import pcount
        ds_nan = ds.copy()
        ds_nan["hs"].values[:5] = np.nan
        out = pcount(ds_nan[["hs"]], dim="time")
        assert float(out["hs"].max()) < 100.0


# ---------------------------------------------------------------------------
# Exceedance
# ---------------------------------------------------------------------------

class TestExceedance:
    def test_exceedance_basic(self, ds):
        from gridstats.ops.exceedance import exceedance
        out = exceedance(ds[["hs"]], dim="time", threshold=2.0)
        assert "hs_2" in out
        val = float(out["hs_2"].mean())
        assert 0 <= val <= 1

    def test_nonexceedance_basic(self, ds):
        from gridstats.ops.exceedance import nonexceedance
        out = nonexceedance(ds[["hs"]], dim="time", threshold=2.0)
        assert "hs_2" in out

    def test_exceedance_plus_nonexceedance_sum_to_one(self, ds):
        from gridstats.ops.exceedance import exceedance, nonexceedance
        exc = exceedance(ds[["hs"]], dim="time", threshold=2.0)
        non = nonexceedance(ds[["hs"]], dim="time", threshold=2.0)
        total = float((exc["hs_2"] + non["hs_2"]).mean())
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_exceedance_group(self, ds):
        from gridstats.ops.exceedance import exceedance
        out = exceedance(ds[["hs"]], dim="time", threshold=2.0, group="month")
        assert "month" in out.dims


# ---------------------------------------------------------------------------
# RPV
# ---------------------------------------------------------------------------

class TestRpv:
    def test_rpv_output_shape(self, ds):
        from gridstats.ops.rpv import rpv
        out = rpv(ds[["hs"]], dim="time", return_periods=[10, 100])
        assert "period" in out.dims
        assert out.period.size == 2

    def test_rpv_bad_distribution_raises(self, ds):
        from gridstats.ops.rpv import rpv
        with pytest.raises(ValueError, match="not found"):
            rpv(ds[["hs"]], dim="time", distribution="not_a_dist")

    def test_rpv_values_increasing(self, ds):
        from gridstats.ops.rpv import rpv
        out = rpv(ds[["hs"]], dim="time", return_periods=[10, 100])
        # 100-year value should be >= 10-year value
        v10 = float(out["hs"].sel(period=10).mean())
        v100 = float(out["hs"].sel(period=100).mean())
        assert v100 >= v10


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------

class TestDistribution:
    def test_distribution3(self, ds):
        from gridstats.ops.distribution import distribution3
        out = distribution3(
            ds,
            dim="time",
            var1="hs", var2="tp", var3="dpm",
            bins1={"start": 0, "step": 1.0},
            bins2={"start": 0, "step": 2.0},
            bins3={"start": 0, "stop": 360, "step": 90},
        )
        assert "dist" in out
        assert "hs" in out.dims
        assert "tp" in out.dims
        assert "dpm" in out.dims

    def test_distribution2(self, ds):
        from gridstats.ops.distribution import distribution2
        out = distribution2(
            ds,
            dim="time",
            var1="hs", var2="dpm",
            bins1={"start": 0, "step": 1.0},
            bins2={"start": 0, "stop": 360, "step": 90},
        )
        assert "dist2" in out

    def test_distribution3_group(self, ds):
        from gridstats.ops.distribution import distribution3
        out = distribution3(
            ds,
            dim="time",
            var1="hs", var2="tp", var3="dpm",
            bins1={"start": 0, "step": 1.0},
            bins2={"start": 0, "step": 2.0},
            bins3={"start": 0, "stop": 360, "step": 90},
            group="month",
        )
        assert "month" in out.dims


# ---------------------------------------------------------------------------
# Directional
# ---------------------------------------------------------------------------

class TestDirectional:
    def test_statdir(self, ds):
        from gridstats.ops.directional import statdir
        out = statdir(ds, funcs=["mean"], dir_var="dpm", nsector=4, dim="time")
        assert "direction" in out.dims
        assert out.direction.size == 4

    def test_statdir_missing_dir_var_raises(self, ds):
        from gridstats.ops.directional import statdir
        with pytest.raises(ValueError, match="not found"):
            statdir(ds, funcs=["mean"], dir_var="nonexistent", dim="time")


# ---------------------------------------------------------------------------
# Probability
# ---------------------------------------------------------------------------

class TestProbability:
    def test_range_probability(self, ds):
        from gridstats.ops.probability import range_probability
        out = range_probability(
            ds,
            dim="time",
            data_ranges=[
                {"var": "hs", "start": 1.0, "stop": None, "left": "closed", "right": "closed"},
                {"var": "hs", "start": None, "stop": 2.0, "left": "closed", "right": "open"},
            ],
        )
        assert "hs_1_to_max" in out
        assert "hs_min_to_2" in out
        assert 0 <= float(out["hs_1_to_max"].mean()) <= 1

    def test_range_probability_custom_label(self, ds):
        from gridstats.ops.probability import range_probability
        out = range_probability(
            ds,
            dim="time",
            data_ranges=[{"var": "hs", "start": 2.0, "stop": None, "label": "hs_above_2"}],
        )
        assert "hs_above_2" in out


# ---------------------------------------------------------------------------
# Derived variables
# ---------------------------------------------------------------------------

class TestDerivedVariables:
    @pytest.fixture
    def ds_wind(self):
        """Dataset with u/v wind components."""
        rng = np.random.default_rng(0)
        nt, ny, nx = 24, 3, 3
        return xr.Dataset(
            {
                "uwnd": (["time", "lat", "lon"], rng.uniform(-10, 10, (nt, ny, nx)).astype("float32")),
                "vwnd": (["time", "lat", "lon"], rng.uniform(-10, 10, (nt, ny, nx)).astype("float32")),
                "ucur": (["time", "lat", "lon"], rng.uniform(-2, 2, (nt, ny, nx)).astype("float32")),
                "vcur": (["time", "lat", "lon"], rng.uniform(-2, 2, (nt, ny, nx)).astype("float32")),
                "cloud_cover": (["time", "lat", "lon"], rng.uniform(0, 1, (nt, ny, nx)).astype("float32")),
                "fp": (["time", "lat", "lon"], rng.uniform(0.05, 0.25, (nt, ny, nx)).astype("float32")),
                "hs_sea": (["time", "lat", "lon"], rng.uniform(0, 5, (nt, ny, nx)).astype("float32")),
                "hs_sw1": (["time", "lat", "lon"], rng.uniform(0, 3, (nt, ny, nx)).astype("float32")),
                "lp_sw1": (["time", "lat", "lon"], rng.uniform(50, 400, (nt, ny, nx)).astype("float32")),
            },
            coords={
                "time": xr.date_range("2020-01-01", periods=nt, freq="1h"),
                "lat": [0.0, 1.0, 2.0],
                "lon": [100.0, 101.0, 102.0],
            },
        )

    def test_wspd(self, ds_wind):
        from gridstats.ops.derived import wspd
        out = wspd(ds_wind)
        assert out.dims == ("time", "lat", "lon")
        # Speed must be non-negative
        assert float(out.min()) >= 0.0
        # Check values match manual calculation at one point
        u0 = float(ds_wind["uwnd"].isel(time=0, lat=0, lon=0))
        v0 = float(ds_wind["vwnd"].isel(time=0, lat=0, lon=0))
        expected = float(np.sqrt(u0**2 + v0**2))
        assert abs(float(out.isel(time=0, lat=0, lon=0)) - expected) < 1e-4

    def test_wspd_custom_var_names(self, ds_wind):
        from gridstats.ops.derived import wspd
        ds = ds_wind.rename({"uwnd": "u10", "vwnd": "v10"})
        out = wspd(ds, uwnd="u10", vwnd="v10")
        assert float(out.min()) >= 0.0

    def test_wdir_range(self, ds_wind):
        from gridstats.ops.derived import wdir
        out = wdir(ds_wind)
        assert float(out.min()) >= 0.0
        assert float(out.max()) < 360.0

    def test_cspd(self, ds_wind):
        from gridstats.ops.derived import cspd
        out = cspd(ds_wind)
        assert float(out.min()) >= 0.0

    def test_cdir_range(self, ds_wind):
        from gridstats.ops.derived import cdir
        out = cdir(ds_wind)
        assert float(out.min()) >= 0.0
        assert float(out.max()) < 360.0

    def test_tp(self, ds_wind):
        from gridstats.ops.derived import tp
        out = tp(ds_wind)
        # Period should be the inverse of frequency (0.05–0.25 Hz → 4–20 s)
        assert float(out.min()) > 3.0
        assert float(out.max()) < 21.0

    def test_douglas_sea_range(self, ds_wind):
        from gridstats.ops.derived import douglas_sea
        out = douglas_sea(ds_wind)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 9.0

    def test_douglas_swell_range(self, ds_wind):
        from gridstats.ops.derived import douglas_swell
        out = douglas_swell(ds_wind)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 9.0

    def test_clear_sky(self, ds_wind):
        from gridstats.ops.derived import clear_sky
        out = clear_sky(ds_wind)
        assert out.dtype == bool

    def test_covered_sky(self, ds_wind):
        from gridstats.ops.derived import covered_sky
        out = covered_sky(ds_wind)
        assert out.dtype == bool

    def test_derived_with_dask(self, ds_wind):
        from gridstats.ops.derived import wspd
        ds_dask = ds_wind.chunk({"time": 6})
        out = wspd(ds_dask)
        assert hasattr(out.data, "dask")
        result = out.compute()
        assert float(result.min()) >= 0.0

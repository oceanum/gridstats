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
# Mode
# ---------------------------------------------------------------------------

class TestMode:
    # Bins suited for Douglas-style integer scale 0–9
    DOUGLAS_BINS = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]

    def _ds_with_known_mode(self, mode_val: float, n: int = 100) -> xr.Dataset:
        """Dataset where `x` has a clear mode at `mode_val` (integer-valued)."""
        rng = np.random.default_rng(0)
        vals = rng.integers(0, 10, size=(n,)).astype("float32")
        # Overwrite half the values with the target mode
        vals[: n // 2] = mode_val
        return xr.Dataset(
            {"x": (["time"], vals)},
            coords={"time": np.arange(n)},
        )

    def test_returns_correct_mode(self):
        from gridstats.ops.aggregations import mode
        ds = self._ds_with_known_mode(3.0)
        out = mode(ds, dim="time", bins=self.DOUGLAS_BINS)
        assert float(out["x"]) == pytest.approx(3.0)

    def test_mode_reduces_time_dim(self, ds):
        from gridstats.ops.aggregations import mode
        out = mode(ds[["hs"]], dim="time", bins=np.arange(0, 6.5, 0.5).tolist())
        assert "time" not in out.dims
        assert out["hs"].shape == (3, 3)

    def test_all_nan_returns_nan(self):
        from gridstats.ops.aggregations import mode
        ds = xr.Dataset(
            {"x": (["time"], np.full(20, np.nan, dtype="float32"))},
            coords={"time": np.arange(20)},
        )
        out = mode(ds, dim="time", bins=self.DOUGLAS_BINS)
        assert np.isnan(float(out["x"]))

    def test_weight_var_excluded_from_output(self):
        from gridstats.ops.aggregations import mode
        rng = np.random.default_rng(1)
        ds = xr.Dataset(
            {
                "x": (["time"], rng.integers(0, 10, 50).astype("float32")),
                "w": (["time"], rng.random(50).astype("float32")),
            },
            coords={"time": np.arange(50)},
        )
        out = mode(ds, dim="time", bins=self.DOUGLAS_BINS, weight_var="w")
        assert "x" in out
        assert "w" not in out

    def test_weighted_mode_shifts_result(self):
        from gridstats.ops.aggregations import mode
        # 60 % occurrences at degree 2, but weight concentrates on degree 7
        n = 100
        vals = np.array([2.0] * 60 + [7.0] * 40, dtype="float32")
        wts = np.array([0.1] * 60 + [10.0] * 40, dtype="float32")
        ds = xr.Dataset(
            {"x": (["time"], vals), "w": (["time"], wts)},
            coords={"time": np.arange(n)},
        )
        unweighted = mode(ds[["x"]], dim="time", bins=self.DOUGLAS_BINS)
        weighted = mode(ds, dim="time", bins=self.DOUGLAS_BINS, weight_var="w")
        assert float(unweighted["x"]) == pytest.approx(2.0)
        assert float(weighted["x"]) == pytest.approx(7.0)

    def test_group_month_adds_month_dim(self, ds):
        from gridstats.ops.aggregations import mode
        out = mode(
            ds[["hs"]], dim="time", bins=np.arange(0, 6.5, 0.5).tolist(), group="month"
        )
        assert "month" in out.dims
        assert out.month.size == 12

    def test_mode_value_within_bins(self, ds):
        from gridstats.ops.aggregations import mode
        bins = np.arange(0, 6.5, 0.5).tolist()
        out = mode(ds[["hs"]], dim="time", bins=bins)
        val = float(out["hs"].mean())
        assert bins[0] <= val <= bins[-1]


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

# ---------------------------------------------------------------------------
# DataArray input coercion (registry wrapper)
# ---------------------------------------------------------------------------

class TestDataArrayInput:
    """Stat functions accept DataArray and return DataArray transparently."""

    def _da(self, name="hs"):
        rng = np.random.default_rng(7)
        nt, ny, nx = 48, 3, 3
        return xr.DataArray(
            (rng.random((nt, ny, nx)) * 4 + 0.5).astype("float32"),
            dims=["time", "latitude", "longitude"],
            coords={
                "time": xr.date_range("2020-01-01", periods=nt, freq="1h"),
                "latitude": [-40.0, -39.0, -38.0],
                "longitude": [170.0, 171.0, 172.0],
            },
            name=name,
        )

    def test_mean_returns_dataarray(self):
        from gridstats.ops.aggregations import mean
        out = mean(self._da(), dim="time")
        assert isinstance(out, xr.DataArray)
        assert out.name == "hs"
        assert "time" not in out.dims

    def test_max_returns_dataarray(self):
        from gridstats.ops.aggregations import max
        out = max(self._da(), dim="time")
        assert isinstance(out, xr.DataArray)

    def test_exceedance_returns_dataarray(self):
        from gridstats.ops.exceedance import exceedance
        out = exceedance(self._da(), dim="time", threshold=2.0)
        assert isinstance(out, xr.DataArray)
        assert float(out.min()) >= 0 and float(out.max()) <= 1

    def test_nonexceedance_returns_dataarray(self):
        from gridstats.ops.exceedance import nonexceedance
        out = nonexceedance(self._da(), dim="time", threshold=2.0)
        assert isinstance(out, xr.DataArray)

    def test_mode_returns_dataarray(self):
        from gridstats.ops.aggregations import mode
        out = mode(self._da(), dim="time", bins=np.arange(0, 6.5, 0.5).tolist())
        assert isinstance(out, xr.DataArray)

    def test_unnamed_dataarray_works(self):
        from gridstats.ops.aggregations import mean
        da = self._da(name=None)
        assert da.name is None
        out = mean(da, dim="time")
        assert isinstance(out, xr.DataArray)

    def test_group_month_returns_dataarray(self):
        from gridstats.ops.aggregations import mean
        out = mean(self._da(), dim="time", group="month")
        assert isinstance(out, xr.DataArray)
        assert "month" in out.dims

    def test_dataset_input_unchanged(self):
        from gridstats.ops.aggregations import mean
        ds = xr.Dataset({"hs": self._da()})
        out = mean(ds, dim="time")
        assert isinstance(out, xr.Dataset)


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
        from gridstats.derived.wind import wspd
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
        from gridstats.derived.wind import wspd
        ds = ds_wind.rename({"uwnd": "u10", "vwnd": "v10"})
        out = wspd(ds, uwnd="u10", vwnd="v10")
        assert float(out.min()) >= 0.0

    def test_wdir_range(self, ds_wind):
        from gridstats.derived.wind import wdir
        out = wdir(ds_wind)
        assert float(out.min()) >= 0.0
        assert float(out.max()) < 360.0

    def test_cspd(self, ds_wind):
        from gridstats.derived.current import cspd
        out = cspd(ds_wind)
        assert float(out.min()) >= 0.0

    def test_cdir_range(self, ds_wind):
        from gridstats.derived.current import cdir
        out = cdir(ds_wind)
        assert float(out.min()) >= 0.0
        assert float(out.max()) < 360.0

    def test_tp(self, ds_wind):
        from gridstats.derived.wave import tp
        out = tp(ds_wind)
        # Period should be the inverse of frequency (0.05–0.25 Hz → 4–20 s)
        assert float(out.min()) > 3.0
        assert float(out.max()) < 21.0

    def test_douglas_sea_range(self, ds_wind):
        from gridstats.derived.wave import douglas_sea
        out = douglas_sea(ds_wind)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 9.0

    def test_douglas_swell_range(self, ds_wind):
        from gridstats.derived.wave import douglas_swell
        out = douglas_swell(ds_wind)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 9.0

    def test_clear_sky(self, ds_wind):
        from gridstats.derived.sky import clear_sky
        out = clear_sky(ds_wind)
        assert out.dtype == bool

    def test_covered_sky(self, ds_wind):
        from gridstats.derived.sky import covered_sky
        out = covered_sky(ds_wind)
        assert out.dtype == bool

    def test_derived_with_dask(self, ds_wind):
        from gridstats.derived.wind import wspd
        ds_dask = ds_wind.chunk({"time": 6})
        out = wspd(ds_dask)
        assert hasattr(out.data, "dask")
        result = out.compute()
        assert float(result.min()) >= 0.0

    def test_uorb_seabed(self, ds_wind):
        """Near-bed orbital velocity is positive and physically reasonable."""
        from gridstats.derived.uorb import uorb
        ds_wind = ds_wind.assign({"hs": ds_wind["hs_sea"], "tp": ds_wind["fp"].pipe(lambda x: 1.0 / x)})
        out = uorb(ds_wind, hs="hs", tp="tp", depth=20.0, z=0.0)
        assert float(out.min()) >= 0.0
        assert out.attrs["units"] == "m/s"

    def test_uorb_deep_water_decays_with_depth(self, ds_wind):
        """Velocity in deep water decreases from surface to seabed."""
        from gridstats.derived.uorb import uorb
        ds = ds_wind.assign({
            "hs": ds_wind["hs_sea"].clip(min=0.5),
            "tp": xr.full_like(ds_wind["hs_sea"], 10.0),
        })
        u_bed = uorb(ds, hs="hs", tp="tp", depth=200.0, z=0.0)
        u_surf = uorb(ds, hs="hs", tp="tp", depth=200.0, z=200.0)
        # In deep water, surface velocity must exceed seabed velocity
        assert float(u_surf.mean()) > float(u_bed.mean())

    def test_uorb_shallow_water_uniform(self, ds_wind):
        """In very shallow water, velocity is nearly depth-uniform."""
        from gridstats.derived.uorb import uorb
        ds = ds_wind.assign({
            "hs": xr.full_like(ds_wind["hs_sea"], 0.5),
            "tp": xr.full_like(ds_wind["hs_sea"], 20.0),
        })
        u_bed = uorb(ds, hs="hs", tp="tp", depth=1.0, z=0.0)
        u_mid = uorb(ds, hs="hs", tp="tp", depth=1.0, z=0.5)
        # In shallow water, profile should be very uniform (ratio close to 1)
        ratio = float(u_mid.mean()) / float(u_bed.mean())
        assert 0.95 < ratio < 1.05

    def test_uorb_soulsby_formula(self):
        """Verify against the Soulsby (1997) eq. 2.44 analytical value."""
        import math
        from gridstats.derived.uorb import uorb, _wavenumber
        # Set up: Hs=2 m, Tp=10 s, depth=25 m, z=0 (seabed)
        Hs, Tp, h = 2.0, 10.0, 25.0
        omega = 2.0 * math.pi / Tp
        k_da = _wavenumber(xr.DataArray([omega]), xr.DataArray([float(h)]))
        k = float(k_da[0])
        U_expected = math.pi * Hs / (Tp * math.sinh(k * h))
        ds = xr.Dataset(
            {"hs": (["t"], [Hs]), "tp": (["t"], [Tp])},
            coords={"t": [0]},
        )
        out = uorb(ds, hs="hs", tp="tp", depth=h, z=0.0)
        assert abs(float(out[0]) - U_expected) < 1e-4

    def test_uorb_dask(self, ds_wind):
        """uorb is dask-compatible and produces the same result as eager mode."""
        from gridstats.derived.uorb import uorb
        ds = ds_wind.assign({
            "hs": ds_wind["hs_sea"].clip(min=0.1),
            "tp": xr.full_like(ds_wind["hs_sea"], 8.0),
        })
        ds_dask = ds.chunk({"time": 6})
        out_eager = uorb(ds, hs="hs", tp="tp", depth=30.0, z=0.0)
        out_dask = uorb(ds_dask, hs="hs", tp="tp", depth=30.0, z=0.0)
        assert hasattr(out_dask.data, "dask")
        np.testing.assert_allclose(out_dask.compute().values, out_eager.values, rtol=1e-5)

    def test_uorb_nan_for_zero_depth(self, ds_wind):
        """Zero or negative depth cells return NaN."""
        from gridstats.derived.uorb import uorb
        ds = ds_wind.assign({
            "hs": ds_wind["hs_sea"],
            "tp": xr.full_like(ds_wind["hs_sea"], 8.0),
        })
        out = uorb(ds, hs="hs", tp="tp", depth=0.0, z=0.0)
        assert np.all(np.isnan(out.values))

    def test_uorb_solver_exact_vs_explicit(self):
        """The two solvers agree to within the documented 0.2 % tolerance."""
        from gridstats.derived.uorb import uorb
        ds = xr.Dataset(
            {"hs": (["t"], [2.0]), "tp": (["t"], [10.0])},
            coords={"t": [0]},
        )
        u_explicit = float(uorb(ds, depth=25.0, z=0.0, solver="explicit")[0])
        u_exact    = float(uorb(ds, depth=25.0, z=0.0, solver="exact")[0])
        assert abs(u_explicit - u_exact) / u_exact < 0.002

    def test_uorb_invalid_solver_raises(self):
        from gridstats.derived.uorb import uorb
        ds = xr.Dataset({"hs": (["t"], [1.0]), "tp": (["t"], [8.0])}, coords={"t": [0]})
        with pytest.raises(ValueError, match="solver"):
            uorb(ds, depth=20.0, solver="bad")

    def test_uorb_reference_surface_equals_bed_conversion(self, ds_wind):
        """reference='surface' with z=0 should equal reference='bed' with z=depth."""
        from gridstats.derived.uorb import uorb
        h = 30.0
        ds_w = ds_wind.assign({
            "hs": ds_wind["hs_sea"],
            "tp": ds_wind["fp"].pipe(lambda x: 1.0 / x),
        })
        # z=0 from surface = still-water level = z=h from bed
        u_surface = uorb(ds_w, depth=h, z=0.0, reference="surface")
        u_bed_top = uorb(ds_w, depth=h, z=h, reference="bed")
        np.testing.assert_allclose(u_surface.values, u_bed_top.values, rtol=1e-5)

    def test_uorb_reference_surface_midwater(self, ds_wind):
        """reference='surface' with z=z0 equals reference='bed' with z=depth-z0."""
        from gridstats.derived.uorb import uorb
        h, z_surf = 40.0, 10.0
        ds_w = ds_wind.assign({
            "hs": ds_wind["hs_sea"],
            "tp": ds_wind["fp"].pipe(lambda x: 1.0 / x),
        })
        u_from_surface = uorb(ds_w, depth=h, z=z_surf, reference="surface")
        u_from_bed = uorb(ds_w, depth=h, z=h - z_surf, reference="bed")
        np.testing.assert_allclose(u_from_surface.values, u_from_bed.values, rtol=1e-5)

    def test_uorb_reference_surface_long_name(self, ds_wind):
        """long_name reflects 'below surface' when reference='surface'."""
        from gridstats.derived.uorb import uorb
        ds_w = ds_wind.assign({
            "hs": ds_wind["hs_sea"],
            "tp": ds_wind["fp"].pipe(lambda x: 1.0 / x),
        })
        out = uorb(ds_w, depth=20.0, z=5.0, reference="surface")
        assert "below surface" in out.attrs["long_name"]

    def test_uorb_invalid_reference_raises(self):
        from gridstats.derived.uorb import uorb
        ds = xr.Dataset({"hs": (["t"], [1.0]), "tp": (["t"], [8.0])}, coords={"t": [0]})
        with pytest.raises(ValueError, match="reference"):
            uorb(ds, depth=20.0, reference="bad")

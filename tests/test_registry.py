"""Tests for gridstats.registry."""
import pytest
import xarray as xr

from gridstats.registry import (
    _LOADERS,
    _STATS,
    get_loader,
    get_stat,
    list_loaders,
    list_stats,
    register_loader,
    register_stat,
)


@pytest.fixture(autouse=True)
def clean_registries():
    """Snapshot and restore registries around each test."""
    orig_stats = dict(_STATS)
    orig_loaders = dict(_LOADERS)
    yield
    _STATS.clear()
    _STATS.update(orig_stats)
    _LOADERS.clear()
    _LOADERS.update(orig_loaders)


class TestRegisterStat:
    def test_register_and_get(self):
        @register_stat("test_stat")
        def my_stat(data: xr.Dataset, **kwargs) -> xr.Dataset:
            return data

        assert get_stat("test_stat") is my_stat

    def test_decorator_returns_original_function(self):
        @register_stat("identity")
        def fn(data, **kwargs):
            return data

        ds = xr.Dataset({"x": 1})
        assert fn(ds) is ds

    def test_overwrite_warns(self, caplog):
        @register_stat("dup")
        def v1(data, **kwargs):
            return data

        with caplog.at_level("WARNING"):

            @register_stat("dup")
            def v2(data, **kwargs):
                return data

        assert "already registered" in caplog.text
        assert get_stat("dup") is v2

    def test_missing_raises_key_error(self):
        with pytest.raises(KeyError, match="not registered"):
            get_stat("nonexistent_xyz")

    def test_list_stats_sorted(self):
        @register_stat("zzz_stat")
        def z(data, **kwargs):
            return data

        @register_stat("aaa_stat")
        def a(data, **kwargs):
            return data

        names = list_stats()
        assert "zzz_stat" in names
        assert "aaa_stat" in names
        assert names == sorted(names)


class TestRegisterLoader:
    def test_register_and_get(self):
        @register_loader("test_loader")
        class MyLoader:
            def load(self, config):
                return xr.Dataset()

        assert get_loader("test_loader") is MyLoader

    def test_overwrite_warns(self, caplog):
        @register_loader("dup_loader")
        class L1:
            pass

        with caplog.at_level("WARNING"):

            @register_loader("dup_loader")
            class L2:
                pass

        assert "already registered" in caplog.text
        assert get_loader("dup_loader") is L2

    def test_missing_raises_key_error(self):
        with pytest.raises(KeyError, match="not registered"):
            get_loader("nonexistent_xyz")

    def test_list_loaders_sorted(self):
        @register_loader("loader_b")
        class B:
            pass

        @register_loader("loader_a")
        class A:
            pass

        names = list_loaders()
        assert "loader_a" in names
        assert "loader_b" in names
        assert names == sorted(names)

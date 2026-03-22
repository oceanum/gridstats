"""Tests for onstats.cli."""
import textwrap

import numpy as np
import pytest
import xarray as xr
from typer.testing import CliRunner

from onstats.cli import app


@pytest.fixture
def source_file(tmp_path):
    ds = xr.Dataset(
        {
            "hs": (
                ["time", "latitude", "longitude"],
                (np.random.rand(24 * 365, 3, 3) * 3 + 0.5).astype("float32"),
            )
        },
        coords={
            "time": xr.date_range("2020-01-01", periods=24 * 365, freq="1h"),
            "latitude": [-40.0, -39.0, -38.0],
            "longitude": [170.0, 171.0, 172.0],
        },
    )
    path = tmp_path / "source.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture
def config_file(tmp_path, source_file):
    out = tmp_path / "out.nc"
    content = textwrap.dedent(f"""\
        source:
          type: xarray
          urlpath: {source_file}
          engine: netcdf4
        output:
          outfile: {out}
        calls:
          - func: mean
            dim: time
            data_vars: [hs]
    """)
    cfg = tmp_path / "config.yml"
    cfg.write_text(content)
    return cfg


class TestCli:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output

    def test_run_command_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "config" in result.output.lower()

    def test_run_pipeline(self, config_file):
        runner = CliRunner()
        result = runner.invoke(app, ["run", str(config_file)])
        assert result.exit_code == 0, result.output

    def test_run_missing_config_fails(self):
        runner = CliRunner()
        result = runner.invoke(app, ["run", "nonexistent.yml"])
        assert result.exit_code != 0

    def test_list_stats(self):
        runner = CliRunner()
        result = runner.invoke(app, ["list-stats"])
        assert result.exit_code == 0
        assert "mean" in result.output

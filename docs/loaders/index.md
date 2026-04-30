# Data Loaders

Loaders open a data source and return a preprocessed `xr.Dataset`. The `type` field in the `source` config block selects which loader to use.

| Loader | `type` value | Description |
|---|---|---|
| [xarray](xarray.md) | `xarray` | Any format supported by `xarray.open_dataset`: NetCDF, Zarr, GRIB, HDF5, and more. Accepts local paths and fsspec URIs (`gs://`, `s3://`, `https://`). |
| [intake](intake.md) | `intake` | Dataset from an [intake-forecast](https://github.com/oceanum/intake-forecast) catalog. Requires `pip install "gridstats[extra]"`. |

For custom loaders see [Custom Plugins](../plugins.md).

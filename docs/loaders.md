# Data Loaders

Loaders open a data source and return a preprocessed `xr.Dataset`. The `type` field in the `source` block selects which loader to use.

---

## `type: xarray`

Opens any file format supported by `xarray.open_dataset`: NetCDF, Zarr, GRIB, HDF5, and more. Accepts local paths, `gs://`, `s3://`, `https://`, or any fsspec-compatible URI.

```yaml
source:
  type: xarray
  urlpath: /data/wave_hindcast.zarr
  engine: zarr
  chunks:
    time: 100
    latitude: 50
    longitude: 50
  mapping:
    Hs: hs
    Tp: tp
  sel:
    latitude: {start: -50, stop: -10}
    longitude: {start: 110, stop: 180}
```

| Field | Description |
|---|---|
| `urlpath` | Path or URL to the dataset. **Required.** |
| `engine` | xarray engine to use: `zarr` (default), `netcdf4`, `scipy`, `cfgrib`, … |
| `open_kwargs` | Extra keyword arguments forwarded verbatim to `xarray.open_dataset`. |
| `chunks` | Dask chunk sizes for lazy loading. |
| `mapping` | Dict of `{source_name: target_name}` variable renames applied after opening. |
| `sel` | Label-based selection. Values can be scalars, lists, or `{start, stop}` dicts (converted to `slice`). |
| `isel` | Integer-index selection. Same value types as `sel`. |

---

## `type: intake`

Opens a dataset from an [intake-forecast](https://github.com/oceanum/intake-forecast) catalog. Requires `pip install "gridstats[extra]"`.

```yaml
source:
  type: intake
  catalog: /catalogs/oceanum.yaml
  dataset_id: wave_global_era5
  chunks:
    time: 50
  mapping:
    significant_wave_height: hs
  sel:
    time: {start: "2000-01-01", stop: "2020-12-31"}
```

| Field | Description |
|---|---|
| `catalog` | Path or URL to the intake catalog file. **Required.** |
| `dataset_id` | Entry name within the catalog. **Required.** |
| `mapping`, `sel`, `isel`, `chunks` | Same preprocessing as the xarray loader. |

After loading, the intake loader delegates preprocessing (renaming and selection) to the xarray loader's `_preprocess` method.

---

## Custom loaders

See [Plugins](plugins.md) for how to implement and register your own loader.

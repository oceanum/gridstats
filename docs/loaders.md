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
  slice_dict:
    sel:
      latitude: slice(-50, -10)
      longitude: slice(110, 180)
```

| Field | Description |
|---|---|
| `urlpath` | Path or URL to the dataset. **Required.** |
| `engine` | xarray engine to use: `zarr` (default), `netcdf4`, `scipy`, `cfgrib`, … |
| `open_kwargs` | Extra keyword arguments forwarded verbatim to `xarray.open_dataset`. |
| `chunks` | Dask chunk sizes for lazy loading. |
| `mapping` | Dict of `{source_name: target_name}` variable renames applied after opening. |
| `slice_dict` | Dict of `{method: kwargs}` applied sequentially (e.g. `sel`, `isel`, `where`). |

---

## `type: intake`

Opens a dataset from an [intake-forecast](https://github.com/oceanum/intake-forecast) catalog. Requires the `intake-forecast` package.

```yaml
source:
  type: intake
  catalog: /catalogs/oceanum.yaml
  dataset_id: wave_global_era5
  chunks:
    time: 50
  mapping:
    significant_wave_height: hs
  slice_dict:
    sel:
      time: slice("2000-01-01", "2020-12-31")
```

| Field | Description |
|---|---|
| `catalog` | Path or URL to the intake catalog file. **Required.** |
| `dataset_id` | Entry name within the catalog. **Required.** |
| `chunks`, `mapping`, `slice_dict` | Same preprocessing as the xarray loader. |

After loading, the intake loader delegates preprocessing (renaming and slicing) to the xarray loader's `_preprocess` method.

---

## Custom loaders

See [Plugins](plugins.md) for how to implement and register your own loader.

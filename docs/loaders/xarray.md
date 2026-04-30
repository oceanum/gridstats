# xarray loader

Opens any file format supported by `xarray.open_dataset`: NetCDF, Zarr, GRIB, HDF5, and more. Accepts local paths, `gs://`, `s3://`, `https://`, or any fsspec-compatible URI.

## Fields

| Field | Description |
|---|---|
| `urlpath` | Path or URL to the dataset. **Required.** |
| `engine` | xarray engine to use: `zarr` (default), `netcdf4`, `scipy`, `cfgrib`, … |
| `open_kwargs` | Extra keyword arguments forwarded verbatim to `xarray.open_dataset`. |
| `chunks` | Dask chunk sizes for lazy loading. |
| `mapping` | Dict of `{source_name: target_name}` variable renames applied after opening. |
| `sel` | Label-based selection. Values can be scalars, lists, or `{start, stop}` dicts (converted to `slice`). |
| `isel` | Integer-index selection. Same value types as `sel`. |

## Example

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

## API reference

::: gridstats.loaders.xarray.XarrayLoader
    options:
      show_source: false
      heading_level: 3

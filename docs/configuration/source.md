# Source

Defines the single input dataset. The `type` field is required and selects the loader.

## `type: xarray`

Load from any file or URL supported by xarray (local NetCDF, Zarr, cloud storage, etc.).

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `"xarray"` | — | Selects the xarray loader. **Required.** |
| `urlpath` | string | — | Path or URI to the dataset. Accepts local paths, `gs://`, `s3://`, `https://`, etc. **Required.** |
| `engine` | string | `"zarr"` | xarray engine: `zarr`, `netcdf4`, `h5netcdf`, `scipy`, `cfgrib`, … |
| `open_kwargs` | dict | `{}` | Extra keyword arguments forwarded verbatim to `xarray.open_dataset`. |
| `mapping` | dict | `{}` | Rename variables on load: `{old_name: new_name}`. |
| `sel` | dict | `{}` | Label-based selection applied after load (see [sel / isel](#sel-isel) below). |
| `isel` | dict | `{}` | Index-based selection applied after load (see [sel / isel](#sel-isel) below). |
| `chunks` | dict | `{}` | Dask chunk sizes applied on open: `{dim: size}`. |

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
    latitude: {start: -50, stop: -30}
    time: {start: "2000-01-01", stop: "2020-12-31"}
```

## `type: intake`

Load from an [intake-forecast](https://github.com/oceanum/intake-forecast) catalog. Requires the `intake-forecast` package.

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `"intake"` | — | Selects the intake loader. **Required.** |
| `catalog` | string | — | Path or URI to the intake catalog YAML file. **Required.** |
| `dataset_id` | string | — | Entry name within the catalog. **Required.** |
| `mapping` | dict | `{}` | Rename variables on load. |
| `sel` | dict | `{}` | Label-based selection applied after load. |
| `isel` | dict | `{}` | Index-based selection applied after load. |
| `chunks` | dict | `{}` | Dask chunk sizes. |

```yaml
source:
  type: intake
  catalog: /catalogs/oceanum.yaml
  dataset_id: wave_global_era5
  chunks:
    time: 50
```

## `sel` / `isel`

`sel` selects by coordinate label, `isel` by integer index — exactly as in
[`xr.Dataset.sel`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.sel.html) and
[`xr.Dataset.isel`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.isel.html).

Each dimension value can be:

| Value type | Behaviour | Example |
|---|---|---|
| Scalar | Select a single coordinate value | `depth: 0.0` |
| List | Select those specific coordinate values | `level: [500, 850, 1000]` |
| `{start, stop}` dict | Select a **range** — converted to `slice(start, stop)` | `latitude: {start: -50, stop: -30}` |

!!! warning "Range dicts vs. lists"
    A `{start, stop}` dict is the only way to specify a range. A list like
    `[-50, -30]` selects **exactly those two coordinate values**, not all values
    between them — consistent with xarray's own behaviour.
    Either key may be omitted: `{stop: -30}` → `slice(None, -30)`.

```yaml
source:
  sel:
    latitude: {start: -50, stop: -30}         # range: all latitudes in [-50, -30]
    time: {start: "2000-01-01", stop: "2020-12-31"}
    depth: 0.0                                 # single value
    level: [500, 850, 1000]                    # exact list of values
  isel:
    member: 0                                  # first ensemble member by index
    x: {start: 100, stop: 200}                # index range
```

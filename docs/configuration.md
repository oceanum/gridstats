# Configuration reference

Pipelines are configured in YAML. The file is parsed with `yaml.safe_load` (no Python code execution) and validated against Pydantic models before any data is loaded.

## Top-level schema

```yaml
source:          # single data source (mutually exclusive with sources:)
  ...

sources:         # placeholder — multi-source support (coming soon)
  name:
    ...

output:
  ...

cluster:         # optional Dask cluster settings
  ...

metadata:        # optional: extra CF attributes to merge into the output
  ...

calls:
  - func: <name>
    ...
```

---

## `source`

Defines the single input dataset. The `type` field is required and selects the loader.

### `type: xarray`

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

### `type: intake`

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

### `sel` / `isel`

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

---

## `output`

| Field | Type | Default | Description |
|---|---|---|---|
| `outfile` | string | — | Output file path. Extension determines format: `.nc` for NetCDF4, `.zarr` for Zarr. Supports remote paths (`gs://`, `s3://`). |
| `updir` | string | `null` | *Deprecated.* Write directly to a remote path via `outfile` instead. |

---

## `cluster`

Optional Dask LocalCluster configuration. If `enabled: false` (the default), computations run in the main process without a cluster.

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Whether to start a local Dask cluster. |
| `n_workers` | int | `null` | Number of workers (defaults to number of CPUs). |
| `threads_per_worker` | int | `2` | Threads per worker. |
| `processes` | bool | `true` | Use separate processes (recommended for CPU-bound work). |

---

## `metadata`

Extra CF attributes merged into the output at finalisation. Supports the same keys as `attributes.yml`: `coords`, `data_vars`, `stats`.

```yaml
metadata:
  data_vars:
    hs:
      standard_name: sea_surface_wave_significant_height
      long_name: significant wave height
      units: m
```

---

## `calls`

An ordered list of stat operations. Each call is a dict with a `func` field naming a registered stat function, plus any function-specific parameters.

### Common fields (all calls)

| Field | Type | Default | Description |
|---|---|---|---|
| `func` | string | — | Name of a registered stat function. |
| `data_vars` | list or `"all"` | `"all"` | Variables to pass to the function. Use `"all"` for all dataset variables. |
| `dim` | string | `"time"` | Dimension to reduce along. |
| `group` | string | `null` | Time component for groupby: `month`, `season`, `year`. |
| `chunks` | dict | `{}` | Rechunk the data before this call: `{dim: size}`. |
| `tiles` | dict | `{}` | Process in spatial tiles: `{dim: tile_size}`. Useful for memory-intensive stats like `quantile` or `rpv`. |
| `use_dask_cluster` | bool | `true` | Whether to use the Dask cluster for this call (if `cluster.enabled: true`). |
| `use_flox` | bool | `true` | Whether to use [flox](https://flox.readthedocs.io) for groupby reductions. Efficient for most operations, but set to `false` for `quantile` on large grids — flox's quantile path uses ~2× the memory of the native xarray implementation. |
| `nsector` | int | `null` | If set, bin the data into this many directional sectors and apply the function to each. |
| `dir_var` | string | `"dpm"` | Directional variable used for sectorisation when `nsector` is set. |
| `suffix` | string | `null` | Override the auto-generated output variable suffix. Default: `_{func}` (plus `_{group}` and `_direc` if applicable). |

### Function-specific parameters

Any additional fields in a call dict are forwarded as keyword arguments to the stat function. See [Operations](ops/index.md) for the parameters accepted by each built-in function.

### Example with all common fields

```yaml
calls:
  - func: rpv
    data_vars: [hs]
    dim: time
    group: null
    chunks:
      time: -1        # full time axis in one chunk (required by rpv)
      latitude: 50
      longitude: 50
    tiles:
      latitude: 10
    use_dask_cluster: true
    use_flox: true
    nsector: 8
    dir_var: dpm
    suffix: _return_period
    # rpv-specific:
    return_periods: [10, 50, 100]
    percentile: 95
    distribution: gumbel_r
    duration: 24
```

---

## Multi-source (upcoming)

The `sources` field is a placeholder for a future feature. When implemented, each named source will map to a separate Zarr group in the output, allowing datasets on different grids to coexist without interpolation:

```yaml
# Not yet implemented — schema is validated but will raise NotImplementedError at runtime
sources:
  waves:
    type: xarray
    urlpath: gs://bucket/waves.zarr
  wind:
    type: xarray
    urlpath: gs://bucket/wind.zarr

output:
  outfile: ./combined.zarr   # will write combined.zarr/waves and combined.zarr/wind

calls:
  - source: waves
    func: mean
    data_vars: [hs]
  - source: wind
    func: mean
    data_vars: [wspd]
```

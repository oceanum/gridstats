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

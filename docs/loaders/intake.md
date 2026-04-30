# intake loader

Opens a dataset from an [intake-forecast](https://github.com/oceanum/intake-forecast) catalog. Requires `pip install "gridstats[extra]"`.

After loading, preprocessing (variable renaming, `sel`/`isel`) is delegated to the same logic as the xarray loader.

## Fields

| Field | Description |
|---|---|
| `catalog` | Path or URL to the intake catalog file. **Required.** |
| `dataset_id` | Entry name within the catalog. **Required.** |
| `mapping` | Dict of `{source_name: target_name}` variable renames. |
| `sel` | Label-based selection (scalars, lists, or `{start, stop}` dicts). |
| `isel` | Integer-index selection. |
| `chunks` | Dask chunk sizes for lazy loading. |

## Example

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

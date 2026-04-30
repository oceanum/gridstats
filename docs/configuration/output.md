# Output

Controls where and how results are written.

## Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `outfile` | string | — | Output file path. Extension determines format: `.nc` for NetCDF4, `.zarr` for Zarr. Supports remote paths (`gs://`, `s3://`). **Required.** |
| `global_attrs` | dict | `{}` | Global dataset attributes to add to or override the defaults. See [Global attributes](#global-attributes) below. |
| `append` | bool | `false` | Add variables to an existing Zarr store rather than overwriting it. See [Parallel Zarr writes](#parallel-zarr-writes) below. |
| `consolidate` | bool | `false` | Run `zarr.consolidate_metadata()` after writing. See [Parallel Zarr writes](#parallel-zarr-writes) below. |
| `updir` | string | `null` | *Deprecated.* Write directly to a remote path via `outfile` instead. |

## Global attributes

By default, gridstats writes the following global attributes to every output file:

| Attribute | Default value |
|---|---|
| `title` | `"Data stats"` |
| `institution` | `"Oceanum"` |
| `source` | `"gridstats"` |
| `date_created` | Today's date (UTC) |
| `time_coverage_start` | First timestamp in the source dataset |
| `time_coverage_end` | Last timestamp in the source dataset |
| `time_coverage_duration` | ISO 8601 duration |
| `time_coverage_resolution` | ISO 8601 timestep |

Any key in `global_attrs` is merged on top of these defaults, overriding the matching default or adding a new attribute:

```yaml
output:
  outfile: ./stats.zarr
  global_attrs:
    title: "New Zealand Wave Climatology 1980–2020"
    institution: "NIWA"
    project: "NZ-Waves-2025"
    references: "https://doi.org/10.xxxx/xxxxx"
```

## Parallel Zarr writes

When computing different statistics in separate parallel tasks and writing them all to the same Zarr archive, set `append: true` on each task. Each task writes only its own variables; all other variables in the store are left untouched.

```yaml
# Task A — computes hs stats
output:
  outfile: gs://my-bucket/stats.zarr
  append: true

# Task B — computes tp stats (runs in parallel with Task A)
output:
  outfile: gs://my-bucket/stats.zarr
  append: true

# Task C — consolidates metadata (runs after A and B complete)
output:
  outfile: gs://my-bucket/stats.zarr
  append: true
  consolidate: true
```

!!! warning "Parallel write safety"
    Different tasks **must** write different variables. If two tasks attempt to write the
    same variable simultaneously the result is undefined. Rerunning a single task
    (e.g., after a failure) is safe — the existing variable is deleted and rewritten.

!!! note "Reading before consolidation"
    Until the consolidation task runs, open the store with `consolidated=False`:

    ```python
    import xarray as xr
    ds = xr.open_zarr("gs://my-bucket/stats.zarr", consolidated=False)
    ```

    After `consolidate: true` has run, the standard `xr.open_zarr(...)` call works
    without any extra arguments.

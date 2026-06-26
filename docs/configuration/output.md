# Output

Controls where and how results are written.

## Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `outfile` | string | — | Output file path. Extension determines format: `.nc` for NetCDF4, `.zarr` for Zarr. Supports remote paths (`gs://`, `s3://`). **Required.** |
| `global_attrs` | dict | `{}` | Global dataset attributes to add to or override the defaults. See [Global attributes](#global-attributes) below. |
| `append` | bool | `false` | Add variables to an existing Zarr store rather than overwriting it. See [Parallel Zarr writes](#parallel-zarr-writes) below. |
| `consolidate` | bool | `false` | Run `zarr.consolidate_metadata()` after writing. See [Parallel Zarr writes](#parallel-zarr-writes) below. |
| `mask` | dict | `null` | Optional spatial mask applied to all output variables before writing. See [Masking](#masking) below. |
| `updir` | string | `null` | Optional remote directory to upload the written output to after the run. See [Uploading to a remote directory](#uploading-to-a-remote-directory) below. |

## Masking

An optional spatial mask can be applied to all output variables before writing. The mask is derived from a variable in the **source** dataset and broadcast automatically across any extra dimensions (e.g. time, quantile, direction) in the output.

Two mask types are supported, selected via the `type` field.

### `notnull` — mask where a variable is null

Keeps output values where the chosen source variable is non-null; sets everything else to NaN. The most common use case is deriving a land/ice mask from a single timestamp of a wave or depth variable.

```yaml
output:
  outfile: out.zarr
  mask:
    type: notnull
    var: hs          # source variable to test
    isel:            # optional: reduce to 2-D before testing
      time: 0
```

### `threshold` — mask by a numerical condition

Keeps output values where `var <operator> value` is true.

| `operator` | Condition |
|---|---|
| `gt` | `var > value` |
| `lt` | `var < value` |
| `ge` | `var >= value` |
| `le` | `var <= value` |

```yaml
output:
  outfile: out.zarr
  mask:
    type: threshold
    var: depth
    isel:
      time: 0
    operator: gt
    value: 0.0       # keep ocean points (depth > 0)
```

### `isel`

Both mask types accept an optional `isel` dict that reduces the source variable to a lower-dimensional slice before the mask is computed. This is the standard way to produce a 2-D `(latitude, longitude)` mask from a variable that also has a time dimension.

```yaml
isel:
  time: 0      # use the first time step
```

## Uploading to a remote directory

Set `updir` to have gridstats upload the finished output to a remote directory once the run completes. The output is copied under its basename, so `outfile: ./scratch/hs.zarr` with `updir: gs://my-bucket/stats` lands at `gs://my-bucket/stats/hs.zarr`. Any fsspec-supported target works (`gs://`, `s3://`, local paths), and Zarr stores are copied recursively.

```yaml
output:
  outfile: ./scratch/hs.zarr        # written to fast local disk first
  updir: gs://my-bucket/stats       # then uploaded here
```

This is useful for deployments (e.g. Argo on k8s) that compute to local scratch and then publish the result to object storage.

!!! note "Writing directly to remote storage"
    For Zarr output you usually don't need `updir` at all — point `outfile` straight at a
    `gs://` / `s3://` path and gridstats streams chunks to the bucket as it writes. Reach
    for `updir` only when you want to stage the output on local disk first (faster writes,
    or to keep partial results off remote storage until the run succeeds). When `outfile`
    is already a remote path, `updir` is ignored.

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

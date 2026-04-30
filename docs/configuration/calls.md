# Calls

An ordered list of stat operations. Each call is a dict with a `func` field naming a registered stat function, plus any function-specific parameters.

## Common fields

| Field | Type | Default | Description |
|---|---|---|---|
| `func` | string | — | Name of a registered stat function. **Required.** |
| `data_vars` | list or `"all"` | `"all"` | Variables to pass to the function. Use `"all"` for all dataset variables. |
| `derived_vars` | list | `[]` | Derived variables to compute before the stat runs. See [Derived Variables](../derived/index.md). |
| `dim` | string | `"time"` | Dimension to reduce along. |
| `group` | string | `null` | Time component for groupby: `month`, `season`, `year`. |
| `chunks` | dict | `{}` | Rechunk the data before this call: `{dim: size}`. |
| `tiles` | dict | `{}` | Process in spatial tiles: `{dim: tile_size}`. Useful for memory-intensive stats like `quantile` or `rpv`. |
| `use_dask_cluster` | bool | `true` | Whether to use the Dask cluster for this call (if `cluster.enabled: true`). |
| `use_flox` | bool | `true` | Whether to use [flox](https://flox.readthedocs.io) for groupby reductions. Efficient for most operations, but set to `false` for `quantile` on large grids — flox's quantile path uses ~2× the memory of the native xarray implementation. |
| `nsector` | int | `null` | If set, bin the data into this many directional sectors and apply the function to each. |
| `dir_var` | string | `"dpm"` | Directional variable used for sectorisation when `nsector` is set. |
| `suffix` | string | `null` | Override the auto-generated output variable suffix. Default: `_{func}` (plus `_{group}` and `_direc` if applicable). |

## Function-specific parameters

Any additional fields in a call dict are forwarded as keyword arguments to the stat function. See [Operations](../ops/index.md) for the parameters accepted by each built-in function.

## Example

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

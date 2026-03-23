# Operations overview

An **operation** (or "op") is a function that reduces an `xr.Dataset` along one or more dimensions and returns a result dataset. Operations are registered by name in the [registry](../api/registry.md) and referenced from config via the `func` field.

## Built-in operations

| `func` | Description |
|---|---|
| [`mean`](aggregations.md#gridstats.ops.aggregations.mean) | Arithmetic mean |
| [`max`](aggregations.md#gridstats.ops.aggregations.max) | Maximum value |
| [`min`](aggregations.md#gridstats.ops.aggregations.min) | Minimum value |
| [`std`](aggregations.md#gridstats.ops.aggregations.std) | Standard deviation |
| [`count`](aggregations.md#gridstats.ops.aggregations.count) | Count of non-NaN values |
| [`quantile`](aggregations.md#gridstats.ops.aggregations.quantile) | Quantiles at specified levels |
| [`pcount`](aggregations.md#gridstats.ops.aggregations.pcount) | Percentage of non-NaN values |
| [`exceedance`](exceedance.md#gridstats.ops.exceedance.exceedance) | Probability that values exceed a threshold |
| [`nonexceedance`](exceedance.md#gridstats.ops.exceedance.nonexceedance) | Probability that values stay below a threshold |
| [`rpv`](rpv.md#gridstats.ops.rpv.rpv) | Return period values via extreme-value analysis |
| [`distribution3`](distribution.md#gridstats.ops.distribution.distribution3) | 3-D joint histogram (Hs × Tp × Dir) |
| [`distribution2`](distribution.md#gridstats.ops.distribution.distribution2) | 2-D joint histogram (speed × direction) |
| [`distribution3_timestep`](distribution.md#gridstats.ops.distribution.distribution3_timestep) | Memory-efficient 3-D histogram (accumulated in time chunks) |
| [`statdir`](directional.md#gridstats.ops.directional.statdir) | Apply multiple functions over directional sectors |
| [`hmo`](frequency-domain.md#gridstats.ops.frequency_domain.hmo) | Frequency-domain Hs in spectral bands |
| [`range_probability`](probability.md#gridstats.ops.probability.range_probability) | Probability of values falling within specified ranges |
| [`winpow`](windpower.md#gridstats.ops.windpower.winpow) | Wind turbine power from wind speed |

## Output variable naming

Each operation's output variables are renamed with a suffix before merging into the output dataset. The default suffix is:

```
{variable}_{func}[_{group}][_direc]
```

Examples:

| Call | Input var | Output var |
|---|---|---|
| `func: mean` | `hs` | `hs_mean` |
| `func: mean, group: month` | `hs` | `hs_mean_month` |
| `func: rpv, nsector: 8` | `hs` | `hs_rpv_direc` |
| `func: rpv, suffix: _100yr` | `hs` | `hs_100yr` |

## Grouping

All operations that accept a `group` parameter will compute the statistic separately for each value of a time component. Supported groups:

| `group` | Output dimension | Values |
|---|---|---|
| `month` | `month` | 1–12 |
| `season` | `season` | DJF, MAM, JJA, SON |
| `year` | `year` | actual years present in the data |

## Directional sectorisation

Setting `nsector` on any call bins the data by direction before applying the function. For example, `nsector: 8` creates 8 × 45° sectors. The output gains a `direction` dimension with sector-centre values (0°, 45°, …, 315°).

The `dir_var` field (default: `dpm`) names the directional variable used for binning. It must be present in the dataset alongside the `data_vars`.

## Custom operations

See [Custom Plugins](../plugins.md) for how to register your own stat functions.

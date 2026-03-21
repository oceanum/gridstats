# Return Period Values

Estimates the value expected to be exceeded once every N years using peaks-over-threshold (POT) extreme value analysis.

## `rpv`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `return_periods` | list[float] | `[1,5,10,20,50,100,1000,10000]` | Return periods in years. |
| `percentile` | float | `95` | Percentile used to define the POT threshold. |
| `distribution` | str | `"gumbel_r"` | Any `scipy.stats` continuous distribution name. |
| `duration` | float | `24` | Minimum peak separation in hours. |
| `group` | str | `null` | Time grouping (rarely used with RPV). |

The output has a `period` dimension with the return period values (in years).

```yaml
- func: rpv
  dim: time
  data_vars: [hs]
  return_periods: [10, 25, 50, 100]
  percentile: 95
  distribution: gumbel_r
  duration: 24         # peaks must be at least 24 h apart
  chunks:
    time: -1           # RPV requires the full time axis in one chunk
    latitude: 30
    longitude: 30
  tiles:
    latitude: 10
```

!!! note
    `rpv` requires the entire time axis to be in a single chunk. Use `chunks: {time: -1}` together with `tiles` for large grids.

## Distributions

Any distribution available in `scipy.stats` can be used. Common choices:

| `distribution` | Use case |
|---|---|
| `gumbel_r` | Wave heights, wind speeds (Gumbel max) |
| `weibull_min` | Wind speed (Weibull) |
| `genpareto` | Generalised Pareto for POT data |
| `lognorm` | Log-normal |

## Method

1. A threshold is defined at the given `percentile` of the data.
2. Peaks above the threshold are extracted, requiring at least `duration` hours between peaks (to ensure independence).
3. The chosen `scipy.stats` distribution is fitted to the peaks.
4. Values at each return period are computed from the fitted distribution's inverse survival function.

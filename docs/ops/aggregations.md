# Aggregations

Standard reduction operations wrapping xarray's built-in methods. All support temporal grouping via `group`.

---

## `mean`

Arithmetic mean along `dim`.

```yaml
- func: mean
  dim: time
  data_vars: [hs, tp, wspd]
  group: month           # optional
```

---

## `max`

Maximum value along `dim`.

```yaml
- func: max
  dim: time
  data_vars: [hs]
  group: season
```

---

## `min`

Minimum value along `dim`.

```yaml
- func: min
  dim: time
  data_vars: [hs]
```

---

## `std`

Standard deviation along `dim`.

```yaml
- func: std
  dim: time
  data_vars: [hs, tp]
```

---

## `count`

Count of non-NaN values along `dim`. Useful as a data-availability metric.

```yaml
- func: count
  dim: time
  data_vars: [hs]
```

---

## `quantile`

Quantiles at one or more levels.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `q` | list[float] | — | Quantile levels in [0, 1]. |

The output has a `quantile` dimension.

```yaml
- func: quantile
  dim: time
  data_vars: [hs]
  q: [0.5, 0.75, 0.90, 0.95, 0.99]
  chunks:
    time: -1         # quantile requires the full time axis in one chunk
    latitude: 50
    longitude: 50
  tiles:
    latitude: 10     # process 10 rows at a time if memory is tight
```

!!! note
    `quantile` loads the entire time axis into memory per spatial chunk. Use `tiles` to limit peak memory usage on large grids.

---

## `pcount`

Percentage of non-NaN values (0–100). Indicates data coverage.

```yaml
- func: pcount
  dim: time
  data_vars: [hs, tp]
```

---

## API reference

::: onstats.ops.aggregations.mean
    options:
      show_source: false
      heading_level: 3

::: onstats.ops.aggregations.max
    options:
      show_source: false
      heading_level: 3

::: onstats.ops.aggregations.min
    options:
      show_source: false
      heading_level: 3

::: onstats.ops.aggregations.std
    options:
      show_source: false
      heading_level: 3

::: onstats.ops.aggregations.count
    options:
      show_source: false
      heading_level: 3

::: onstats.ops.aggregations.quantile
    options:
      show_source: false
      heading_level: 3

::: onstats.ops.aggregations.pcount
    options:
      show_source: false
      heading_level: 3

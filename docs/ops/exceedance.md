# Exceedance

Probability that a variable exceeds (or stays below) a threshold, optionally requiring the condition to persist for a minimum duration.

---

## `exceedance`

Probability that values are **≥ threshold** (and **≤ maxval** when set).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `threshold` | float | — | Lower bound for exceedance. |
| `maxval` | float | `inf` | Upper bound. |
| `inclusive` | bool | `true` | Use `>=` instead of `>`. |
| `duration` | str or list[str] | `"0h"` | Minimum continuous duration the condition must hold. |
| `group` | str | `null` | Time grouping (`month`, `season`, `year`). |

When `duration` is a list the output includes a `duration` dimension.

Output variable names are suffixed with the threshold value, e.g. `hs_2` for threshold 2.

```yaml
- func: exceedance
  dim: time
  data_vars: [hs]
  threshold: 2.5
  duration: 1d          # must persist ≥ 1 day

- func: exceedance
  dim: time
  data_vars: [hs]
  threshold: 4.0
  duration: [12h, 1d, 3d]   # multiple durations → 'duration' dimension
  group: year
```

---

## `nonexceedance`

Probability that values are **≤ threshold**.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `threshold` | float | — | Upper bound. |
| `inclusive` | bool | `true` | Use `<=` instead of `<`. |
| `duration` | str or list[str] | `"0h"` | Minimum continuous duration. |
| `group` | str | `null` | Time grouping. |

```yaml
- func: nonexceedance
  dim: time
  data_vars: [hs]
  threshold: 1.0
  duration: 3d
```

---

## Duration format

Duration strings use pandas `Timedelta` notation:

| String | Meaning |
|---|---|
| `"0h"` | No duration filtering (simple fraction of time steps) |
| `"6h"` | 6 hours |
| `"1d"` | 1 day |
| `"7d"` | 1 week |

---

## API reference

::: gridstats.ops.exceedance.exceedance
    options:
      show_source: false
      heading_level: 3

::: gridstats.ops.exceedance.nonexceedance
    options:
      show_source: false
      heading_level: 3

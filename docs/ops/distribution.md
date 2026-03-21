# Distributions

Joint histograms (counts) of two or three variables. Results are not normalised — they are raw integer counts that can be normalised downstream.

---

## Bin specification

All distribution functions accept bin specifications as either:

- A **dict** with `start`, `stop` (optional), `step` keys — equivalent to `numpy.arange(start, stop+step, step)`. If `stop` is omitted it is inferred from the data maximum.
- A **list** or array of explicit bin edges.

```yaml
bins1: {start: 0, step: 0.5}           # 0, 0.5, 1.0, ...  (stop from data)
bins2: {start: 0, stop: 20, step: 1.0} # 0, 1, 2, ..., 20
bins3: [0, 90, 180, 270, 360]           # explicit edges
```

---

## `distribution3`

3-D joint histogram over three variables (typically Hs × Tp × Dir).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `var1` | str | `"hs"` | First variable name. |
| `var2` | str | `"tp"` | Second variable name. |
| `var3` | str | `"dpm"` | Third variable name. |
| `bins1/2/3` | dict or list | — | Bin specifications for each variable. |
| `isdir1/2/3` | bool | `false/false/true` | Whether each variable is directional (values wrap at 360°). |
| `group` | str | `null` | Time grouping (e.g. `month` for monthly climatological distributions). |

Output variable: `dist` with dimensions `(var1, var2, var3)`.

```yaml
- func: distribution3
  dim: time
  data_vars: [hs, tp, dpm]
  var1: hs
  var2: tp
  var3: dpm
  bins1: {start: 0, step: 0.5}
  bins2: {start: 0, step: 1.0}
  bins3: {start: 0, stop: 360, step: 45}
  isdir3: true
  group: month
```

---

## `distribution2`

2-D joint histogram over two variables (typically speed × direction).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `var1` | str | `"wspd"` | First variable name. |
| `var2` | str | `"wdir"` | Second variable name. |
| `bins1/2` | dict or list | — | Bin specifications. |
| `isdir1/2` | bool | `false/true` | Whether each variable is directional. |
| `group` | str | `null` | Time grouping. |

Output variable: `dist2` with dimensions `(var1, var2)`.

```yaml
- func: distribution2
  dim: time
  data_vars: [wspd, wdir]
  var1: wspd
  var2: wdir
  bins1: {start: 0, step: 1.0}
  bins2: {start: 0, stop: 360, step: 22.5}
  isdir2: true
  group: month
```

---

## `distribution3_timestep`

Memory-efficient variant of `distribution3` that accumulates histogram counts in time chunks. Suitable for very long timeseries where loading the full dataset at once would exceed memory limits.

All parameters are the same as `distribution3`, plus:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `freq` | str | `"30d"` | Time chunk size as a pandas frequency string (e.g. `"10d"`, `"1ME"`). |

```yaml
- func: distribution3_timestep
  dim: time
  data_vars: [hs, tp, dpm]
  var1: hs
  var2: tp
  var3: dpm
  bins1: {start: 0, step: 0.5}
  bins2: {start: 0, step: 1.0}
  bins3: {start: 0, stop: 360, step: 45}
  group: month
  freq: 10d           # load 10 days at a time
```

!!! tip
    Prefer `distribution3` for datasets that fit comfortably in memory — it is faster because it avoids repeated I/O. Use `distribution3_timestep` only for multi-decade datasets.

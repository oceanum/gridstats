# Range Probability

Probability that a variable falls within a specified value range.

---

## `range_probability`

Computes the fraction of non-NaN time steps where a variable falls within `[start, stop]` (bounds configurable as open or closed).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data_ranges` | list[dict] | — | List of range specifications (see below). |

### Range specification

Each item in `data_ranges` is a dict with the following fields:

| Key | Type | Default | Description |
|---|---|---|---|
| `var` | str | — | Variable name in the input dataset. |
| `start` | float \| null | — | Lower bound. `null` means no lower bound. |
| `stop` | float \| null | — | Upper bound. `null` means no upper bound. |
| `left` | `"closed"` \| `"open"` | `"closed"` | Whether the lower bound is inclusive (`>=`) or exclusive (`>`). |
| `right` | `"closed"` \| `"open"` | `"closed"` | Whether the upper bound is inclusive (`<=`) or exclusive (`<`). |
| `label` | str | `{var}_{start}_to_{stop}` | Output variable name. |

### Output

One probability variable per range specification, with values in [0, 1].

---

### Examples

Probability of Hs between 1.5 m and 3.0 m (inclusive):

```yaml
- func: range_probability
  dim: time
  data_vars: [hs]
  data_ranges:
    - var: hs
      start: 1.5
      stop: 3.0
      label: hs_moderate
```

Multiple ranges with custom labels:

```yaml
- func: range_probability
  dim: time
  data_vars: [wspd]
  data_ranges:
    - var: wspd
      start: null
      stop: 3.0
      label: wspd_calm           # < 3 m/s (calm)
    - var: wspd
      start: 3.0
      stop: 10.0
      left: open                 # (3, 10]
      label: wspd_moderate
    - var: wspd
      start: 10.0
      stop: null
      left: open
      label: wspd_strong         # > 10 m/s
```

---

## API reference

::: gridstats.ops.probability.range_probability
    options:
      show_source: false
      heading_level: 3

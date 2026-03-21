# Wind Power

Estimates wind turbine power output from a wind speed variable.

---

## `winpow`

Converts wind speed to power using a third-order polynomial fitted to a reference turbine power curve, then applies cut-in, rated, and cut-out speed limits.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `turbine_power` | float | — | Rated power output in kW. **Required.** |
| `cutin` | float | `3.0` | Cut-in wind speed (m/s). Output is zero below this. |
| `rated` | float | `10.61` | Rated wind speed (m/s). Output is capped at `turbine_power` above this. |
| `cutout` | float | `25.0` | Cut-out wind speed (m/s). Output is zero above this. |
| `agg` | str \| null | `"mean"` | Aggregation applied after computing instantaneous power (`"mean"`, `"max"`, etc.). Set to `null` to return the full power time series. |
| `group` | str \| null | `null` | Time grouping (`month`, `season`, `year`). Only valid when `agg` is set. |

---

### Power curve

The polynomial is fitted to a reference 16 MW offshore turbine curve. The relationship between wind speed and output power follows:

```
P(v) = a·v + b·v² + c·v³ + d    for  cutin < v < rated
P(v) = turbine_power              for  rated ≤ v < cutout
P(v) = 0                          for  v ≤ cutin  or  v ≥ cutout
```

The polynomial coefficients are fitted internally — only `turbine_power` is needed to scale the output to your turbine's rated capacity.

---

### Examples

Mean power output for a 15 MW turbine:

```yaml
- func: winpow
  dim: time
  data_vars: [wspd]
  turbine_power: 15000   # kW
  agg: mean
```

Monthly mean and maximum power with custom speed limits:

```yaml
- func: winpow
  dim: time
  data_vars: [wspd]
  turbine_power: 8000
  cutin: 4.0
  rated: 12.0
  cutout: 25.0
  agg: mean
  group: month
```

Return the full instantaneous power time series (no aggregation):

```yaml
- func: winpow
  dim: time
  data_vars: [wspd]
  turbine_power: 15000
  agg: null
```

---

!!! note
    `winpow` operates on the first variable in `data_vars`. Pass a single wind-speed variable per call.

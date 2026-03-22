# Frequency-Domain Statistics

Spectral analysis operations that estimate statistics from time-series data using Welch's power spectral density method.

---

## `hmo`

Frequency-domain significant wave height (Hm0) computed by integrating the power spectral density over one or more frequency bands.

**Method**: Welch's method is used to estimate the PSD `S(f)`, then Hs is derived per band as:

```
Hs = 4 × √( ∫ S(f) df )
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `segsec` | float | `256` | Welch segment length in seconds. Longer segments give finer frequency resolution. |
| `bands` | dict | see below | Mapping of band label to `[fmin, fmax]` in Hz. `null` for a bound means use the data's min/max frequency. |

The sampling frequency is inferred automatically from the time coordinate spacing.

Output variables are renamed to `hs_{band_label}` and the result has a `band` coordinate.

---

### Default frequency bands

| Label | Range | Description |
|---|---|---|
| `0_25` | 0 – 0.25 Hz | Swell + wind sea |
| `8_25` | 0.08 – 0.25 Hz | Swell band |
| `25_120` | 0.25 – 1.20 Hz | Wind sea |
| `25_300` | 0.25 – 3.00 Hz | Wind sea (extended) |
| `tot` | full spectrum | Total Hs |

---

### Examples

Compute Hs over the default bands:

```yaml
- func: hmo
  dim: time
  data_vars: [eta]
  segsec: 256
```

Override with custom bands:

```yaml
- func: hmo
  dim: time
  data_vars: [eta]
  segsec: 512
  bands:
    swell: [0.04, 0.10]
    wind_sea: [0.10, 0.50]
    total: [null, null]
```

---

!!! note
    `hmo` expects a regularly sampled time series of surface elevation (or equivalent). Gaps or irregular sampling will produce inaccurate PSD estimates.

---

## API reference

::: onstats.ops.frequency_domain.hmo
    options:
      show_source: false
      heading_level: 3

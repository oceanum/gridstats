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
| `fs` | float | `null` | Sampling frequency in Hz. When omitted it is inferred from the time coordinate (see below). Set explicitly to override. |

### Sampling frequency

When `fs` is not given, it is inferred from the **mean** sampling interval over the whole record (rounded to 3 decimal places), not just the first interval. This is robust to small timing jitter — e.g. model output whose nominal 1 Hz cadence wanders by a few percent because the internal time step varies — and the rounding lets a nominal 1 s cadence collapse cleanly to `fs = 1.0`. Both numeric (seconds) and `datetime64` time coordinates are supported. Pass `fs` explicitly when the time coordinate is unreliable or absent.

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
    `hmo` expects an approximately regularly sampled time series of surface elevation (or equivalent). Small timing jitter is tolerated (the inferred `fs` uses the mean interval), but true gaps or strongly irregular sampling will produce inaccurate PSD estimates.

---

## API reference

::: gridstats.ops.frequency_domain.hmo
    options:
      show_source: false
      heading_level: 3

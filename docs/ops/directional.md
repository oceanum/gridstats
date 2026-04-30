# Directional Statistics

Apply one or more stat functions over equal-width directional sectors.

---

## `statdir`

Bins the dataset by direction, applies each listed function to each sector, and concatenates the results along a new `direction` dimension.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `funcs` | list[str] | — | Names of registered stat functions to apply per sector (e.g. `[mean, max]`). |
| `dir_var` | str | `"dpm"` | Variable name used for directional binning. Must be in the dataset. |
| `nsector` | int | `4` | Number of equally-spaced sectors. `8` → 45° sectors, `16` → 22.5°, etc. |

Additional keyword arguments (e.g. `group`) are forwarded to each stat function.

Output variables are named as `{variable}_{func}` and gain a `direction` coordinate with sector-centre values (0°, 45°, …).

```yaml
- func: statdir
  dim: time
  data_vars: [hs, tp]
  dir_var: dpm
  nsector: 8
  funcs: [mean, max]
```

---

## How sectors are defined

For `nsector: N` the sector width is `360 / N` degrees. Sectors are centred on:

```
0°, 360/N°, 2×360/N°, …
```

Each sector is a half-open interval `[centre − width/2, centre + width/2)`. The wrap-around sector (centred on 0°) correctly handles the 0°/360° boundary.

Data points outside a sector are set to NaN before the stat function is called, so all registered functions work transparently without modification.

---

## Difference from `nsector` on individual calls

Setting `nsector` on a regular call (e.g. `func: mean, nsector: 8`) is handled at the pipeline level and applies a single function. `statdir` lets you apply **multiple** functions in one pass while keeping the same directional binning, which is more efficient when you need several statistics per sector.

```yaml
# Equivalent to two separate calls with nsector, but done in one pass:
- func: statdir
  dim: time
  data_vars: [hs]
  nsector: 16
  funcs: [mean, max, quantile]
  q: [0.90, 0.99]
```

---

## API reference

::: gridstats.ops.directional.statdir
    options:
      show_source: false
      heading_level: 3

---

## `modal_direction`

Per-cell modal (most frequent) direction from a weighted circular histogram. Uses a histogram-based mode rather than a vector mean, making it robust to bimodal and anti-parallel direction distributions (e.g. monsoon reversals where the arithmetic mean of 90° and 270° is meaningless).

All direction variables in `data_vars` are processed; the optional `weight_var` is excluded from output but used to weight the histogram.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `weight_var` | str \| None | `None` | Variable name for histogram weights. `None` = unweighted frequency. |
| `bin_width_deg` | float | `10.0` | Histogram bin width (degrees). Must divide 360 evenly. |
| `smooth` | bool | `True` | Apply 3-bin circular moving average before taking the mode, stabilising against single-bin noise. |

```yaml
- func: modal_direction
  dim: time
  data_vars: [dpm, dp_sea, dp_sw1]
  weight_var: hs           # weight by wave energy
  bin_width_deg: 10.0
  smooth: true
```

With monthly grouping:

```yaml
- func: modal_direction
  dim: time
  group: month
  data_vars: [dpm]
  weight_var: hs
```

!!! note
    `apply_ufunc` with `allow_rechunk=True` handles the single-chunk requirement along `dim` automatically in the dask graph — no manual rechunking needed before calling this stat.

### API reference

::: gridstats.ops.directional.modal_direction
    options:
      show_source: false
      heading_level: 4

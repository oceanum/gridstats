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

::: onstats.ops.directional.statdir
    options:
      show_source: false
      heading_level: 3

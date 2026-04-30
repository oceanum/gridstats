# Sky Conditions

Derived sky condition variables computed from cloud area fraction.

---

## `clear_sky` — Clear sky

Boolean mask for clear-sky conditions. Returns `True` where cloud fraction is at or below `cover_threshold`.

| Parameter | Default | Description |
|---|---|---|
| `cloud_cover` | `"cloud_cover"` | Cloud area fraction variable name (0–1) |
| `cover_threshold` | `0.0` | Maximum cloud fraction considered clear sky |

```yaml
derived_vars:
  - name: clear_sky
    func: clear_sky
    cover_threshold: 0.1   # allow up to 10% cloud cover
```

---

## `covered_sky` — Covered sky

Boolean mask for fully overcast conditions. Returns `True` where cloud fraction is at or above `cover_threshold`.

| Parameter | Default | Description |
|---|---|---|
| `cloud_cover` | `"cloud_cover"` | Cloud area fraction variable name (0–1) |
| `cover_threshold` | `1.0` | Minimum cloud fraction considered covered sky |

```yaml
derived_vars:
  - name: covered_sky
    func: covered_sky
    cover_threshold: 0.9   # consider overcast above 90% cloud cover
```

---

## API reference

::: gridstats.derived.sky.clear_sky
    options:
      show_source: false
      heading_level: 3

::: gridstats.derived.sky.covered_sky
    options:
      show_source: false
      heading_level: 3

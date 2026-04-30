# Wind

Derived wind variables computed from the eastward (`u`) and northward (`v`) horizontal wind components.

---

## `wspd` — Wind speed

Returns wind speed (m/s) as √(u² + v²).

| Parameter | Default | Description |
|---|---|---|
| `uwnd` | `"uwnd"` | Eastward wind component variable name |
| `vwnd` | `"vwnd"` | Northward wind component variable name |

```yaml
derived_vars:
  - name: wspd
    func: wspd
    uwnd: u10   # override if your dataset uses a different name
    vwnd: v10
```

---

## `wdir` — Wind direction

Returns wind **coming-from** direction in degrees (meteorological convention: 0° = northerly, 90° = easterly).

| Parameter | Default | Description |
|---|---|---|
| `uwnd` | `"uwnd"` | Eastward wind component variable name |
| `vwnd` | `"vwnd"` | Northward wind component variable name |

```yaml
derived_vars:
  - name: wdir
    func: wdir
    uwnd: u10
    vwnd: v10
```

---

## API reference

::: gridstats.derived.wind.wspd
    options:
      show_source: false
      heading_level: 3

::: gridstats.derived.wind.wdir
    options:
      show_source: false
      heading_level: 3

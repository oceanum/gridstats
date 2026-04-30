# Ocean Current

Derived ocean current variables computed from the eastward (`u`) and northward (`v`) horizontal current components.

---

## `cspd` — Current speed

Returns sea water speed (m/s) as √(u² + v²).

| Parameter | Default | Description |
|---|---|---|
| `ucur` | `"ucur"` | Eastward current component variable name |
| `vcur` | `"vcur"` | Northward current component variable name |

```yaml
derived_vars:
  - name: cspd
    func: cspd
    ucur: uo   # override if your dataset uses different names
    vcur: vo
```

---

## `cdir` — Current direction

Returns sea water **going-to** direction in degrees (oceanographic convention: 0° = northward, 90° = eastward).

| Parameter | Default | Description |
|---|---|---|
| `ucur` | `"ucur"` | Eastward current component variable name |
| `vcur` | `"vcur"` | Northward current component variable name |

```yaml
derived_vars:
  - name: cdir
    func: cdir
    ucur: uo
    vcur: vo
```

---

## API reference

::: gridstats.derived.current.cspd
    options:
      show_source: false
      heading_level: 3

::: gridstats.derived.current.cdir
    options:
      show_source: false
      heading_level: 3

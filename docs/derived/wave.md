# Wave Parameters

Derived wave variables computed from integrated spectral wave parameters.

---

## `tp` — Peak period

Returns peak wave period (s) as 1/fp.

| Parameter | Default | Description |
|---|---|---|
| `fp` | `"fp"` | Peak wave frequency variable name (Hz) |

```yaml
derived_vars:
  - name: tp
    func: tp
    fp: peak_freq   # override if your dataset uses a different name
```

---

## `douglas_sea` — Douglas sea scale

Returns the Douglas sea scale (integer 0–9) from wind-sea significant wave height.

| Scale | Description | Hs range |
|---|---|---|
| 0 | Glassy | 0 m |
| 1 | Rippled | 0–0.1 m |
| 2 | Wavelets | 0.1–0.5 m |
| 3 | Slight | 0.5–1.25 m |
| 4 | Moderate | 1.25–2.5 m |
| 5 | Rough | 2.5–4 m |
| 6 | Very rough | 4–6 m |
| 7 | High | 6–9 m |
| 8 | Very high | 9–14 m |
| 9 | Phenomenal | > 14 m |

| Parameter | Default | Description |
|---|---|---|
| `hs_sea` | `"hs_sea"` | Wind-sea significant wave height variable name (m) |

```yaml
derived_vars:
  - name: douglas_sea
    func: douglas_sea
```

---

## `douglas_swell` — Douglas swell scale

Returns the Douglas swell scale (integer 0–9) from swell height and wavelength.

| Parameter | Default | Description |
|---|---|---|
| `hs_sw1` | `"hs_sw1"` | Primary swell significant wave height variable name (m) |
| `lp_sw1` | `"lp_sw1"` | Primary swell peak wavelength variable name (m) |

```yaml
derived_vars:
  - name: douglas_swell
    func: douglas_swell
```

---

## `crossing_seas` — Crossing-seas flag

Boolean mask indicating crossing-seas conditions. Crossing seas are detected when two wave systems are separated by more than an angle threshold *and* the weaker system carries at least a minimum energy fraction.

| Parameter | Default | Description |
|---|---|---|
| `hs` | `"hs"` | Total significant wave height variable (m) |
| `hs_sea` | `"hs_sea"` | Wind-sea Hs variable (m) |
| `hs_sw1` | `"hs_sw1"` | Primary swell Hs variable (m) |
| `dir_sea` | `"dir_sea"` | Wind-sea direction variable (degrees) |
| `dir_sw1` | `"dir_sw1"` | Primary swell direction variable (degrees) |
| `hs_sw2` | `None` | Secondary swell Hs variable (enables sw2 pair checks) |
| `dir_sw2` | `None` | Secondary swell direction variable |
| `hs_threshold` | `0.0` | Minimum total Hs (m) to report crossing seas |
| `angle_threshold` | `40.0` | Minimum relative angle between two systems (degrees) |
| `energy_fraction` | `0.2` | Minimum energy fraction of the weaker system |

```yaml
derived_vars:
  - name: crossing_seas
    func: crossing_seas
    angle_threshold: 45.0   # tighten the directional separation criterion
    energy_fraction: 0.25
```

---

## API reference

::: gridstats.ops.derived.tp
    options:
      show_source: false
      heading_level: 3

::: gridstats.ops.derived.douglas_sea
    options:
      show_source: false
      heading_level: 3

::: gridstats.ops.derived.douglas_swell
    options:
      show_source: false
      heading_level: 3

::: gridstats.ops.derived.crossing_seas
    options:
      show_source: false
      heading_level: 3

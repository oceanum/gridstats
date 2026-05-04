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
Applied to wind-sea Hs (WMO convention); use `hs_sea` to point at the correct variable
in your dataset.

| Scale | Description       | Wind-sea Hs (m) |
|------:|-------------------|-----------------|
| 0     | Calm (glassy)     | 0               |
| 1     | Calm (rippled)    | 0 – 0.1         |
| 2     | Smooth            | 0.1 – 0.5       |
| 3     | Slight            | 0.5 – 1.25      |
| 4     | Moderate          | 1.25 – 2.5      |
| 5     | Rough             | 2.5 – 4.0       |
| 6     | Very rough        | 4.0 – 6.0       |
| 7     | High              | 6.0 – 9.0       |
| 8     | Very high         | 9.0 – 14.0      |
| 9     | Phenomenal        | > 14.0          |

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

Returns the Douglas swell scale (integer 0–8) from primary swell height and wavelength.
Degree 9 ("confused") is a crossing-seas condition; apply `crossing_seas` separately and
overlay it on the output where needed.

| Scale | Description      | Swell Hs (m) | Wavelength (m) |
|------:|------------------|--------------|----------------|
| 0     | No swell         | ≤ 0          | —              |
| 1     | Very low         | 0 – 2        | 0 – 200        |
| 2     | Low              | 0 – 2        | > 200          |
| 3     | Light            | 2 – 4        | 0 – 100        |
| 4     | Moderate         | 2 – 4        | 100 – 200      |
| 5     | Moderate rough   | 2 – 4        | > 200          |
| 6     | Rough            | > 4          | 0 – 100        |
| 7     | High             | > 4          | 100 – 200      |
| 8     | Very high        | > 4          | > 200          |
| 9     | Confused         | crossing seas — see `crossing_seas` ||

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

Boolean mask indicating crossing-seas conditions, implementing the Li (2016) criterion.
A crossing sea is flagged when **both** of the following hold for at least one pair of
wave systems:

1. The shortest-arc angular separation between the two systems exceeds `angle_threshold`.
2. Each system independently carries at least `energy_fraction` of total wave energy,
   i.e. `Hs_partition > sqrt(energy_fraction) × Hs_total`.

All direction variables must use the **same convention** (all coming-from, or all
going-to). Check your hindcast metadata — mixing conventions silently produces wrong
angular separations.

| Parameter | Default | Description |
|---|---|---|
| `hs` | `"hs"` | Total significant wave height variable (m) |
| `hs_sea` | `"hs_sea"` | Wind-sea Hs variable (m) |
| `hs_sw1` | `"hs_sw1"` | Primary swell Hs variable (m) |
| `dir_sea` | `"dir_sea"` | Wind-sea mean direction variable (degrees) |
| `dir_sw1` | `"dir_sw1"` | Primary swell mean direction variable (degrees) |
| `hs_sw2` | `None` | Secondary swell Hs variable (enables sea/sw2 and sw1/sw2 pair checks) |
| `dir_sw2` | `None` | Secondary swell mean direction variable (degrees) |
| `hs_threshold` | `0.5` | Minimum total Hs (m) below which no crossing sea is reported. Raise to `1.0` for a strict alert product. |
| `angle_threshold` | `40.0` | Minimum angular separation (degrees). Li (2016) default. Use `30.0` for a looser climatology map or `45.0` for a stricter event flag. |
| `energy_fraction` | `0.2` | Minimum fraction of total energy each system must carry (Li 2016 default). |

```yaml
derived_vars:
  - name: crossing_seas
    func: crossing_seas
    hs_threshold: 1.0       # strict event-flagging product
    angle_threshold: 45.0   # tighter directional separation
```

**Reference:** Li, X.M. (2016). A new insight from space into swell propagation and
crossing in the global oceans. *Geophysical Research Letters*, 43(10), 5202–5209.
<https://doi.org/10.1002/2016GL068818>

---

## API reference

::: gridstats.derived.wave.tp
    options:
      show_source: false
      heading_level: 3

::: gridstats.derived.wave.douglas_sea
    options:
      show_source: false
      heading_level: 3

::: gridstats.derived.wave.douglas_swell
    options:
      show_source: false
      heading_level: 3

::: gridstats.derived.wave.crossing_seas
    options:
      show_source: false
      heading_level: 3

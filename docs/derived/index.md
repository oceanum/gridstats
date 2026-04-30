# Derived Variables

Derived variables are computed quantities added to the dataset *before* any stat function runs. They transform raw input variables (e.g. u/v wind components) into derived quantities (e.g. wind speed) that can then be passed to any stat via `data_vars`.

Each derived function is registered under a short name and returns a single `xr.DataArray`.

---

## Usage in config

### Full form

```yaml
calls:
  - func: mean
    data_vars: [wspd, hs]
    derived_vars:
      - name: wspd       # variable name added to the dataset
        func: wspd       # registered derived function
        uwnd: u10        # override default input variable name
        vwnd: v10
```

### Shorthand

When the output variable name, the function name, and all input defaults are the same, a bare string can be used:

```yaml
derived_vars:
  - wspd          # equivalent to {name: wspd, func: wspd}
```

### Stacking multiple derived variables

```yaml
calls:
  - func: quantile
    data_vars: [uorb_bed, uorb_5m]
    derived_vars:
      - name: uorb_bed
        func: uorb
        z: 0.0
      - name: uorb_5m
        func: uorb
        z: 5.0
    q: [0.50, 0.90, 0.99]
```

!!! note
    Derived variables listed in `derived_vars` are automatically included when
    computing the stat, even if they are not explicitly listed in `data_vars`.

---

## Available functions

| Function | Page | Description |
|---|---|---|
| [`wspd`](wind.md) | Wind | Wind speed from u/v components |
| [`wdir`](wind.md) | Wind | Wind coming-from direction |
| [`cspd`](current.md) | Ocean Current | Sea water speed from current components |
| [`cdir`](current.md) | Ocean Current | Sea water going-to direction |
| [`tp`](wave.md) | Wave Parameters | Peak wave period from peak frequency |
| [`douglas_sea`](wave.md) | Wave Parameters | Douglas sea scale (0–9) |
| [`douglas_swell`](wave.md) | Wave Parameters | Douglas swell scale (0–9) |
| [`crossing_seas`](wave.md) | Wave Parameters | Boolean crossing-seas flag |
| [`uorb`](uorb.md) | Wave Orbital Velocity | Significant horizontal wave orbital velocity |

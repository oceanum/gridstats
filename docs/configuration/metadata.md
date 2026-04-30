# Metadata

Extra CF-convention attributes merged into the output variables at finalisation. Supports the same keys as `attributes.yml`: `coords`, `data_vars`, `stats`.

```yaml
metadata:
  data_vars:
    hs:
      standard_name: sea_surface_wave_significant_height
      long_name: significant wave height
      units: m
  coords:
    latitude:
      standard_name: latitude
      long_name: latitude
      units: degrees_north
  stats:
    mean:
      long_name: mean
```

!!! note
    `metadata` controls **variable-level** CF attributes (standard_name, long_name, units).
    For **global** dataset attributes (title, institution, etc.) use
    [`output.global_attrs`](output.md#global-attributes) instead.

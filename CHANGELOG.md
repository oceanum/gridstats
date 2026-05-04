# Changelog

## 2.2.0 (unreleased)

### New Features

- **Output masking** — new `mask:` block on the output config applies a spatial mask to
  all output variables before writing. Two types are supported: `notnull` (keep points
  where a source variable is non-null) and `threshold` (keep points satisfying a
  numerical condition). Both accept an optional `isel:` dict to reduce the source
  variable to a 2-D slice before the mask is computed, and xarray broadcasting ensures
  the mask is applied correctly across any extra dimensions (time, quantile, direction).

### Documentation

- **Douglas scales** — corrected scale descriptions in the `douglas_sea` docs table
  (e.g. "Rippled" → "Calm (rippled)", "Wavelets" → "Smooth") and added a full
  degree/description/range table to the `douglas_swell` docs, including a note on
  degree 9.
- **Crossing seas** — rewrote the `crossing_seas` docs section: added a direction
  convention warning (all inputs must use the same coming-from or going-to convention),
  guidance on threshold choices (Li 2016 defaults and operational variants), and the
  full Li (2016) reference with DOI.

### Bug Fixes

- **Crossing seas** — changed `hs_threshold` default from `0.0` to `0.5` m. A zero
  threshold allowed the function to flag crossing seas in near-calm conditions where
  partition directions are undefined and noisy. The Li (2016) criterion is meaningful
  only when both systems are energetically significant; `0.5` m is the standard
  operational floor.

### Bug Fixes

- **Douglas sea scale** — fixed an off-by-one error where all degrees were shifted one
  lower than the correct Douglas classification (e.g. a 3 m sea was reported as degree 4
  "moderate" instead of degree 5 "rough"). Degree 9 (phenomenal, Hs > 14 m) was also
  never assigned.

### Improvements

- **Douglas sea scale** — replaced the 9-iteration `xr.where` loop with a single
  `xr.apply_ufunc(np.digitize, …)` call. Produces a flat dask graph (one task per chunk
  instead of nine) and runs ~1.5× faster on eager arrays; gains are larger under dask
  where the stacked task graph was a scheduling bottleneck.
- **Douglas swell scale** — `DOUGLAS_SWELL_BINS` replaced with `DOUGLAS_SWELL_INTERVALS`,
  a degree-keyed dict of `pd.Interval` pairs matching the original onstats definition.
  Degree 9 ("confused") is documented as a crossing-seas condition that cannot be assigned
  from height/wavelength thresholds alone; it was also unreachable in the original
  (`pd.Interval(inf, inf)` is never satisfied by any finite value).
- **Douglas swell scale** — replaced the 9-iteration `xr.where` loop with
  `xr.apply_ufunc(_classify_swell)` backed by two `np.digitize` calls and a 3×3 LUT.
  Eager compute cost is unchanged (2D classification has similar work to the loop), but
  the dask task graph shrinks ~7× (54 → 8 tasks per chunk), reducing graph construction
  and scheduler overhead on large multi-chunk hindcasts.

---

## 2.1.0 (2026-05-01)

### New Features

- **Derived variables** — new top-level `gridstats.derived` subpackage computes secondary
  variables (wind speed/direction, current speed/direction, wave parameters, wave orbital
  velocity, sky conditions) directly from source data before stats are applied.
- **Modal direction** — new `modal_direction` stat implemented as a vectorised ufunc.
- **Zarr append / consolidate** — new `append` and `consolidate` output options enable
  parallel Zarr writes across tiled or distributed runs.
- **Custom dataset metadata** — new `global_attrs` field on the output config block lets
  you inject arbitrary CF-compliant global attributes into the output dataset.

### Bug Fixes

- Derived variables were silently dropped when `data_vars` was set on a call; this is now
  fixed so derived variables survive variable selection.

### Improvements

- Info-level logging added for loader calls and derived variable definitions, making it
  easier to trace pipeline execution.
- Docs restructured: configuration, loaders, and derived variables each have their own
  section with per-topic pages; MathJax support added for equation rendering.

---

## 2.0.0 (2026-03-25)

A near-complete rewrite that renamed the library from **onstats** to **gridstats** and
replaced the legacy architecture with a modern, well-typed stack.

### Breaking Changes

- Library renamed from `onstats` to `gridstats`; all imports must be updated.
- Configuration schema rewritten using **Pydantic v2** — existing YAML configs require
  migration (see the [Configuration docs](configuration/index.md)).
- `slice_dict` replaced by explicit `sel` / `isel` fields on source config.
- Source config now uses a discriminated union: set `urlpath` for xarray or
  `catalog` + `dataset_id` for intake — the old `type:` field is gone.
- CLI commands changed: `gridstats run config.yml` and `gridstats list-stats`
  (powered by Typer instead of the old argparse CLI).

### New Features

- **Pipeline orchestrator** — new `Pipeline` class handles loading, stat dispatch,
  tiling, directional sectorisation, and output in a single object.
- **Registry with entry-point plugins** — `@register_stat` / `@register_loader`
  decorators; third-party packages can contribute stats or loaders via the
  `gridstats.stats` / `gridstats.loaders` entry-point groups.
- **Spatial tiling** — `tiles:` on any call keeps peak memory bounded for large grids.
- **flox integration** — `use_flox` per-call toggle (default `True`); automatically
  disabled for `quantile` to avoid ~2× memory overhead.
- **open_kwargs** on `XarraySourceConfig` — pass arbitrary kwargs to
  `xarray.open_dataset`.
- **CF attributes** — `gridstats/attributes.yml` maps variable names to CF
  standard\_name / long\_name / units; applied automatically on output.
- Packaging migrated to `pyproject.toml`; mkdocs documentation site published to
  GitHub Pages; GitHub Actions CI/CD added.

---

## 1.2.0 (2022-06-26)

### New Features

- **CF attribute support** — variable attributes (standard\_name, long\_name, units) can
  now be declared in config and are written to the output dataset.
- **Global attributes** — pipeline-level `global_attrs` for dataset-level metadata.

### Bug Fixes

- Fixed crash with non-uniform chunk sizes during output writing.

---

## 1.1.0 (2022-05-17)

### New Features

- **Probability stats** — new module with exceedance and non-exceedance probability
  functions (`pexc`, `pcount`).
- Allow transposing, fixing dtype, and setting chunks on output variables.
- Dask cluster can now be enabled or disabled individually per call.
- `chunks` can be specified when opening datasets from an intake catalog.

### Bug Fixes

- Fixed chunking when applying a land mask.
- Fixed bug in distribution function output.

### Internal Changes

- Removed deprecated KMZ stats code.

---

## 1.0.0 (2021-11-25)

### New Features

- **Return period values (RPV)** — new `rpv` stats module supporting groupby time
  aggregations.
- **Distribution stats** — histogram / empirical distribution functions.
- **Count function** — count non-NaN observations.
- **Directional sectorisation** — `nsector` / directional bin support for computing
  stats per direction sector.
- **Stepwise decorator** — compute stats in time chunks to bound peak memory on large
  datasets.
- **Dask cluster** — explicit cluster setup with configurable worker count (including
  `all` / `half` shortcuts).
- Logging throughout the pipeline.
- Refactored to a CLI entry point.

---

## 0.1.0 (2021-06-18)

First release on PyPI.

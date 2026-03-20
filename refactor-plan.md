# Onstats Refactoring Plan

## Phase 1 — Understand the existing codebase

Before making any changes, thoroughly review the existing code:

- Read every Python file in the package
- Read all config files and example configs to understand usage patterns. several configs are available in /config/hindcast/ontask/tasks/onstats and /config/hindcast/ontask/tasks/stats, but keep in mind some of these may be outdated, and that they also contain specification to run this in our argo cluster.
- Read any existing tests
- Read the README and any documentation

Once done, summarise back to me:

- What the library does
- How it is currently structured
- What the config format looks like
- What the main weaknesses and pain points are

**Ignore all the kmzstats functionality, those won't be addressed for now**

**Do not write any code until Phase 1 is complete and I have confirmed your understanding.**

---

## Phase 2 — Propose the new architecture

The refactored library must meet the following requirements:

### Data loading

- Users specify datasets to load via config
- Data is loaded into **dask-backed xarray Datasets** for out-of-memory computation
- The loading mechanism must be **extensible** with built-in support for `xarray.open_dataset` and `intake`
- Use a plugin/registry pattern so users can register custom loaders without modifying library code

### Statistics computation

- Users specify which statistics to compute on which variables, dimenstions, groups, etc, all via config
- Built-in support for standard xarray/dask operations (mean, std, quantile, min, max, etc)
- Users must be able to **define and register custom statistical functions**
- Custom functions receive an `xarray.DataArray` or `xarray.Dataset` and return a result
- Use a registry pattern so new stats can be added without modifying core code

### Output

- Results are saved as **NetCDF or Zarr** (configurable)
- Output destination should be extensible (local filesystem, cloud storage, etc)

### CLI

- Replace any existing CLI with **typer**
- Clean, minimal interface

### Configuration

- Use **YAML** as the config format
- The config should specify:
  - Data sources and how to load them
  - Which variables to process
  - Which statistics to compute per variable
  - Output format and destination

### Code quality

- Type hints throughout
- Docstrings on all public functions and classes
- Modular package structure with clear separation of concerns (loading, stats, output, config, CLI)
- Tests using **pytest** with fixtures and parametrisation
- Clear README with usage examples
- Get rid of any unecessary legacy dependencies, such as the oncore library.

**Present the proposed architecture (module structure, key classes/interfaces, config schema) before writing code. Wait for my approval.**

---

## Phase 3 — Implement

Once I approve the architecture, implement incrementally:

- One module at a time
- Tests for each module before moving to the next
- Commit after each module is complete and tested
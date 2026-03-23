# Custom Plugins

gridstats has two extension points: **stat functions** and **data loaders**. Both are registered in a central registry and discovered automatically via Python package entry points.

---

## Custom stat functions

A stat function has the signature:

```python
def my_stat(data: xr.Dataset, *, dim: str = "time", **kwargs) -> xr.Dataset:
    ...
```

Register it with the `@register_stat` decorator:

```python
# my_package/stats.py
import xarray as xr
from gridstats.registry import register_stat

@register_stat("my_stat")
def my_stat(data: xr.Dataset, *, dim: str = "time", threshold: float = 0.0, **kwargs) -> xr.Dataset:
    """Return the fraction of values above threshold."""
    return (data > threshold).mean(dim=dim)
```

Then declare the entry point in your package's `pyproject.toml`:

```toml
[project.entry-points."gridstats.stats"]
my_stat = "my_package.stats:my_stat"
```

After installing your package, `my_stat` is available in any pipeline config:

```yaml
- func: my_stat
  dim: time
  data_vars: [hs]
  threshold: 2.5
```

---

## Custom loaders

A loader consists of two parts: a **config model** (a Pydantic class defining the fields your source type accepts) and a **loader class** with a `load(config) -> xr.Dataset` method.

### 1. Define a config model

Subclass `_BaseSourceConfig` (which provides `mapping`, `sel`, `isel`, and `chunks`) and add any fields your source needs:

```python
# my_package/config.py
from typing import Literal
from gridstats.config import _BaseSourceConfig

class DatameshSourceConfig(_BaseSourceConfig):
    type: Literal["datamesh"]
    token: str
    dataset_id: str
    server: str = "https://datamesh.oceanum.io"
```

### 2. Implement the loader

```python
# my_package/loaders.py
import xarray as xr
from gridstats.loaders.xarray import XarrayLoader
from gridstats.registry import register_loader
from my_package.config import DatameshSourceConfig

@register_loader("datamesh")
class DatameshLoader:
    def load(self, config: DatameshSourceConfig) -> xr.Dataset:
        dset = my_datamesh_open(config.token, config.dataset_id, config.server)
        # Delegate renaming and sel/isel to the shared _preprocess helper
        return XarrayLoader()._preprocess(dset, config)
```

### 3. Register entry points

```toml
[project.entry-points."gridstats.loaders"]
datamesh = "my_package.loaders:DatameshLoader"
```

The new source type is then available in any pipeline config:

```yaml
source:
  type: datamesh
  token: my-api-token
  dataset_id: wave_nz_hindcast
  sel:
    time: {start: "2000-01-01", stop: "2020-12-31"}
```

---

## In-process registration

If you don't need a separate package, you can register functions directly before running a pipeline:

```python
import xarray as xr
from gridstats.registry import register_stat
from gridstats.pipeline import Pipeline

@register_stat("custom_mean")
def custom_mean(data: xr.Dataset, *, dim: str = "time", **kwargs) -> xr.Dataset:
    return data.mean(dim=dim) * 2

pipeline = Pipeline.from_yaml("config.yml")
result = pipeline.run()
```

---

## Listing registered functions

```bash
gridstats list-stats
```

Or from Python:

```python
import gridstats
from gridstats.registry import list_stats, list_loaders

print(list_stats())
print(list_loaders())
```

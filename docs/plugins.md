# Custom Plugins

onstats has two extension points: **stat functions** and **data loaders**. Both are registered in a central registry and discovered automatically via Python package entry points.

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
from onstats.registry import register_stat

@register_stat("my_stat")
def my_stat(data: xr.Dataset, *, dim: str = "time", threshold: float = 0.0, **kwargs) -> xr.Dataset:
    """Return the fraction of values above threshold."""
    return (data > threshold).mean(dim=dim)
```

Then declare the entry point in your package's `pyproject.toml`:

```toml
[project.entry-points."onstats.stats"]
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

A loader is a class with a `load(config: SourceConfig) -> xr.Dataset` method:

```python
# my_package/loaders.py
import xarray as xr
from onstats.config import SourceConfig
from onstats.registry import register_loader

@register_loader("my_loader")
class MyLoader:
    def load(self, config: SourceConfig) -> xr.Dataset:
        # config.model_extra contains any extra fields from the YAML
        conn_str = config.model_extra.get("connection_string")
        dset = my_open_function(conn_str)
        return dset
```

Register the entry point:

```toml
[project.entry-points."onstats.loaders"]
my_loader = "my_package.loaders:MyLoader"
```

Extra fields in the `source` block that are not part of the standard `SourceConfig` schema are accessible via `config.model_extra`.

---

## In-process registration

If you don't need a separate package, you can register functions directly before running a pipeline:

```python
import xarray as xr
from onstats.registry import register_stat
from onstats.pipeline import Pipeline

@register_stat("custom_mean")
def custom_mean(data: xr.Dataset, *, dim: str = "time", **kwargs) -> xr.Dataset:
    return data.mean(dim=dim) * 2

pipeline = Pipeline.from_yaml("config.yml")
result = pipeline.run()
```

---

## Listing registered functions

```bash
onstats list-stats
```

Or from Python:

```python
import onstats
from onstats.registry import list_stats, list_loaders

print(list_stats())
print(list_loaders())
```

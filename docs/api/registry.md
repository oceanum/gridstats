# registry

Functions for registering and retrieving stat functions and data loaders. Third-party packages extend gridstats by declaring entry points in the `gridstats.stats` or `gridstats.loaders` groups — see [Plugins](../plugins.md).

::: gridstats.registry
    options:
      members:
        - register_stat
        - register_loader
        - get_stat
        - get_loader
        - list_stats
        - list_loaders
      show_source: false

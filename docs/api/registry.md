# registry

Functions for registering and retrieving stat functions and data loaders. Third-party packages extend onstats by declaring entry points in the `onstats.stats` or `onstats.loaders` groups — see [Plugins](../plugins.md).

::: onstats.registry
    options:
      members:
        - register_stat
        - register_loader
        - get_stat
        - get_loader
        - list_stats
        - list_loaders
      show_source: false

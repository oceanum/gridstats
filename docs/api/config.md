# config

Pydantic models for pipeline configuration. Load a config with [`PipelineConfig.from_yaml`][gridstats.config.PipelineConfig.from_yaml].

The `source` field accepts either an `XarraySourceConfig` or an `IntakeSourceConfig`, selected by the `type` discriminator field.

::: gridstats.config.XarraySourceConfig
    options:
      show_source: false
      heading_level: 2

::: gridstats.config.IntakeSourceConfig
    options:
      show_source: false
      heading_level: 2

::: gridstats.config.OutputConfig
    options:
      show_source: false
      heading_level: 2

::: gridstats.config.ClusterConfig
    options:
      show_source: false
      heading_level: 2

::: gridstats.config.CallConfig
    options:
      show_source: false
      heading_level: 2

::: gridstats.config.PipelineConfig
    options:
      show_source: false
      heading_level: 2

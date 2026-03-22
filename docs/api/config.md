# config

Pydantic models for pipeline configuration. Load a config with [`PipelineConfig.from_yaml`][onstats.config.PipelineConfig.from_yaml].

The `source` field accepts either an `XarraySourceConfig` or an `IntakeSourceConfig`, selected by the `type` discriminator field.

::: onstats.config.XarraySourceConfig
    options:
      show_source: false
      heading_level: 2

::: onstats.config.IntakeSourceConfig
    options:
      show_source: false
      heading_level: 2

::: onstats.config.OutputConfig
    options:
      show_source: false
      heading_level: 2

::: onstats.config.ClusterConfig
    options:
      show_source: false
      heading_level: 2

::: onstats.config.CallConfig
    options:
      show_source: false
      heading_level: 2

::: onstats.config.PipelineConfig
    options:
      show_source: false
      heading_level: 2

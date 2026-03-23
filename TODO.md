# TODO

## Open questions

- **Dask cluster option** (`cluster:` in config): Benchmark whether a `LocalCluster` actually improves throughput vs the default threaded scheduler for typical single-machine workloads. If no measurable benefit, remove from config and pipeline to reduce complexity. The threaded scheduler + spatial `tiles:` already covers memory-bounded cases.

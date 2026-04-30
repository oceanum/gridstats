# Cluster

Optional Dask LocalCluster configuration. If `enabled: false` (the default), computations run in the main process without a cluster.

## Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Whether to start a local Dask cluster. |
| `n_workers` | int | `null` | Number of workers (defaults to number of CPUs). |
| `threads_per_worker` | int | `2` | Threads per worker. |
| `processes` | bool | `true` | Use separate processes (recommended for CPU-bound work). |

!!! note
    Defaults are tuned for cloud environments where CPUs are virtual (hyperthreaded).
    `threads_per_worker: 2` maps two vCPUs to one worker process, giving fewer workers
    with more memory each — important for memory-intensive operations like `quantile`.

```yaml
cluster:
  enabled: true
  n_workers: 4
  threads_per_worker: 2
  processes: true
```

Individual calls can opt out of the cluster by setting `use_dask_cluster: false` on the call, which is useful for operations that are better run in the main process.

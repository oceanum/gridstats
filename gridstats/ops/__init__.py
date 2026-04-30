"""Built-in statistical operations for gridstats.

All ops self-register via @register_stat / @register_derived on import.
"""
from gridstats.ops import (
    aggregations,
    derived,
    directional,
    distribution,
    exceedance,
    frequency_domain,
    probability,
    rpv,
    windpower,
)

__all__ = [
    "aggregations",
    "derived",
    "directional",
    "distribution",
    "exceedance",
    "frequency_domain",
    "probability",
    "rpv",
    "windpower",
]


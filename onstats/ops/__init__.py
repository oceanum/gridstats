"""Built-in statistical operations for onstats.

All ops self-register via @register_stat on import.
"""
from onstats.ops import (
    aggregations,
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
    "directional",
    "distribution",
    "exceedance",
    "frequency_domain",
    "probability",
    "rpv",
    "windpower",
]

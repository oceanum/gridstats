"""Built-in derived variable functions for gridstats.

All functions self-register via @register_derived on import.
"""
from gridstats.derived import current, sky, uorb, wave, wind

__all__ = ["wind", "current", "wave", "sky", "uorb"]

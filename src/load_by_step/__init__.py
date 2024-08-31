"""
Copyright (c) 2024 Alex Kerney. All rights reserved.

load_by_step: A xarray accessor to retrieve large quantities of data from a THREDDS server splitting the request in smaller requests.
"""

from __future__ import annotations

from ._load_by_step import DALoadByStep, DSLoadByStep
from ._options import OPTIONS, get_options, set_options
from ._version import version as __version__

__all__ = [
    "DALoadByStep",
    "DSLoadByStep",
    "OPTIONS",
    "set_options",
    "get_options",
    "__version__",
]

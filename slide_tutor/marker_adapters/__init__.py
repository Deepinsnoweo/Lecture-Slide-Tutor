"""Versioned Marker parser adapters.

Only the production v0.1.3 adapter is registered as the default.  Candidate
adapters are injected explicitly by callers until a production migration is
separately approved.
"""

from slide_tutor.marker_adapters.base import MarkerAdapter, ParserPageResult
from slide_tutor.marker_adapters.runtime import default_marker_adapter

__all__ = ["MarkerAdapter", "ParserPageResult", "default_marker_adapter"]

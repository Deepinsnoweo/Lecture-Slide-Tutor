"""Default adapter resolution without a user-facing route selector."""

from __future__ import annotations

from functools import lru_cache

from slide_tutor.marker_adapters.base import MarkerAdapter


@lru_cache(maxsize=1)
def default_marker_adapter() -> MarkerAdapter:
    """Return the fixed production adapter.

    This is deliberately not an environment-driven registry.  A future
    candidate adapter must be passed explicitly until migration is approved.
    """
    from slide_tutor.marker_adapters.v013 import V013MarkerAdapter

    return V013MarkerAdapter()

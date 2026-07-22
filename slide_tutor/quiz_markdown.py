"""One safe Markdown/LaTeX presentation boundary for Quiz text."""

from __future__ import annotations

import re
from typing import Any


_DISPLAY_MATH = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_INLINE_MATH = re.compile(r"\\\((.+?)\\\)", re.DOTALL)


def normalize_quiz_markdown(value: str | None) -> str:
    """Normalize common LaTeX delimiters without interpreting formula content."""
    text = str(value or "")
    text = _DISPLAY_MATH.sub(lambda match: f"$$\n{match.group(1).strip()}\n$$", text)
    return _INLINE_MATH.sub(lambda match: f"${match.group(1).strip()}$", text)


def render_quiz_markdown(container: Any, value: str | None, **kwargs: Any) -> Any:
    """Render Quiz text through one Streamlit-compatible Markdown entry point."""
    return container.markdown(normalize_quiz_markdown(value), **kwargs)


__all__ = ["normalize_quiz_markdown", "render_quiz_markdown"]

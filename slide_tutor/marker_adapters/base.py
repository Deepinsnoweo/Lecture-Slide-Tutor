"""Stable contracts for version-specific Marker adapters."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ParserPageResult(BaseModel):
    """One source-page candidate returned by a Marker adapter."""

    source_page_number: int = Field(ge=1)
    markdown: str = ""
    structured_blocks: list[dict[str, Any]] = Field(default_factory=list)
    parser_family: str = Field(min_length=1)
    parser_version: str | None = None
    mode: str = Field(min_length=1)
    ocr_used: bool = False
    warnings: list[str] = Field(default_factory=list)
    runtime_facts: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class MarkerAdapter(Protocol):
    """Programmatically injectable Marker candidate provider."""

    def is_available(self) -> bool:
        """Return whether the adapter can parse in the current environment."""

    def cache_identity(self) -> dict[str, str]:
        """Return content-affecting parser and adapter identity fields."""

    def parse_pages(
        self,
        pdf_path: Path,
        page_numbers: Sequence[int],
        *,
        cancel_check: Callable[[], None] | None = None,
    ) -> list[ParserPageResult]:
        """Parse source pages in order without persisting project state."""

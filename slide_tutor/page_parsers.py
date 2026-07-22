"""Page parsing orchestration and deterministic canonical-content routing.

Version-specific Marker behavior lives behind ``MarkerAdapter``.  This module
combines one adapter candidate with the always-available PyMuPDF text and maps
the result into the stable ``PageUnderstanding`` contract.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re

from slide_tutor import cache, config
from slide_tutor.marker_adapters import MarkerAdapter, default_marker_adapter
from slide_tutor.models import PageUnderstanding, SlidePage
from slide_tutor.page_understanding import (
    compute_quality_signals,
    route_canonical_content,
)
from slide_tutor.projects import utc_now


def _source_pdf_path(project_id: str, file_id: str) -> Path:
    return cache.get_project_file_dir(project_id, file_id) / "source.pdf"


def page_batch_cache_identity(
    marker_adapter: MarkerAdapter | None = None,
) -> dict[str, str]:
    """Return the complete parser identity used by page artifact cache keys."""
    adapter = marker_adapter or default_marker_adapter()
    try:
        pymupdf_version = version("PyMuPDF")
    except PackageNotFoundError:
        pymupdf_version = "unavailable"
    return {
        **adapter.cache_identity(),
        "fallback_implementation": "PyMuPDF",
        "fallback_version": pymupdf_version,
    }


def _clean_pymupdf_text(raw_text: str) -> str:
    """Remove common page noise from the PyMuPDF candidate."""
    lines = raw_text.replace("\r\n", "\n").splitlines()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        if re.match(
            r"^(©|Copyright|All Rights Reserved|Confidential)",
            stripped,
            re.IGNORECASE,
        ):
            continue
        cleaned.append(stripped)
    return "\n".join(cleaned).strip()


def _build_page_understanding(
    project_id: str,
    file_id: str,
    slide: SlidePage,
    *,
    marker_text: str = "",
    marker_version: str | None = None,
    warnings: Sequence[str] = (),
) -> PageUnderstanding:
    page_warnings = list(warnings)
    cleaned = _clean_pymupdf_text(slide.text)
    signals = compute_quality_signals(
        cleaned,
        marker_text,
        title=slide.title_guess or "",
        metadata=slide.metadata,
    )
    route_action, canonical, selected_provider, route_warnings = (
        route_canonical_content(
            signals,
            cleaned,
            marker_text,
            title=slide.title_guess or "",
        )
    )
    page_warnings.extend(route_warnings)
    if route_action == "empty":
        page_warnings.append("No extractable text on this page")

    if route_action in {"needs_vlm", "sparse_valid"}:
        quality = route_action
    elif route_action == "empty":
        quality = "empty"
    elif selected_provider == "marker":
        quality = "ok"
    else:
        quality = "fallback"

    return PageUnderstanding(
        project_id=project_id,
        file_id=file_id,
        slide_id=slide.slide_id,
        clean_markdown=canonical,
        raw_text=slide.text,
        parser_provider=selected_provider,
        parser_version=marker_version,
        quality=quality,  # type: ignore[arg-type]
        warnings=page_warnings,
        created_at=utc_now(),
        canonical_markdown=canonical,
        selected_provider=selected_provider,  # type: ignore[arg-type]
        pymupdf_text=cleaned,
        marker_markdown=marker_text,
        quality_signals=signals,
        route_action=route_action,
    )


def parse_page(
    project_id: str,
    file_id: str,
    slide: SlidePage,
    *,
    marker_adapter: MarkerAdapter | None = None,
) -> PageUnderstanding:
    """Parse one page with an explicitly injected or fixed default adapter."""
    return parse_pages(
        project_id,
        file_id,
        [slide],
        marker_adapter=marker_adapter,
    )[0]


def parse_pages(
    project_id: str,
    file_id: str,
    slides: Sequence[SlidePage],
    *,
    cancel_check: Callable[[], None] | None = None,
    marker_adapter: MarkerAdapter | None = None,
) -> list[PageUnderstanding]:
    """Parse one bounded batch while preserving source-page identity."""
    slide_batch = list(slides)
    if not slide_batch:
        return []
    adapter = marker_adapter or default_marker_adapter()
    marker_available = config.MARKER_ENABLED and adapter.is_available()
    candidates = adapter.parse_pages(
        _source_pdf_path(project_id, file_id),
        [slide.page_number for slide in slide_batch],
        cancel_check=cancel_check,
    ) if marker_available else []
    candidate_by_page = {
        candidate.source_page_number: candidate for candidate in candidates
    }
    expected_pages = {slide.page_number for slide in slide_batch}
    if marker_available and (
        len(candidates) != len(slide_batch)
        or len(candidate_by_page) != len(candidates)
        or set(candidate_by_page) != expected_pages
    ):
        raise RuntimeError(
            "Marker adapter returned incomplete or duplicate source pages"
        )

    pages: list[PageUnderstanding] = []
    for slide in slide_batch:
        if cancel_check is not None:
            cancel_check()
        candidate = candidate_by_page.get(slide.page_number)
        pages.append(_build_page_understanding(
            project_id,
            file_id,
            slide,
            marker_text=candidate.markdown if candidate is not None else "",
            marker_version=(
                candidate.parser_version if candidate is not None else None
            ),
            warnings=candidate.warnings if candidate is not None else (),
        ))
    return pages

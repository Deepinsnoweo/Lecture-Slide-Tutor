"""Page understanding orchestration — load, build, and cache per-page structured
text from Marker or PyMuPDF so DeepSeek gets clean markdown instead of raw PDF text."""

from __future__ import annotations

from slide_tutor import cache, config
import re
import hashlib
import json
from typing import Any

from slide_tutor.models import (
    EvidenceEligibility,
    PageQualitySignals,
    PageRouteAction,
    PageUnderstanding,
    PageTextResolution,
    PAGE_CONTENT_CONTRACT_VERSION,
    PAGE_ROUTER_CONTRACT_VERSION,
    SlidePage,
)

PAGE_GENERATION_META_FILENAME = "page_generation_meta.json"


def page_generation_meta_path(project_id: str):
    return cache.get_project_dir(project_id) / PAGE_GENERATION_META_FILENAME


def page_content_contract_signature() -> str:
    """Hash the parser/router contract independently from retrieval settings."""
    payload = {
        "page_content_contract_version": PAGE_CONTENT_CONTRACT_VERSION,
        "router_contract_version": PAGE_ROUTER_CONTRACT_VERSION,
        "signals": "quality-signals-v1",
        "route_thresholds": "marker-loss-035-sparse-080-visual-025",
        "marker_processing": "bounded-document-batch-v1",
        "marker_batch_pages": config.MARKER_BATCH_PAGES,
        "marker_enabled": config.MARKER_ENABLED,
        "marker_local_files_only": config.MARKER_LOCAL_FILES_ONLY,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def active_page_generation(project_id: str) -> dict[str, Any]:
    return cache.load_json(page_generation_meta_path(project_id), default={}) or {}


def page_generation_status(project_id: str, source_signature: str | None = None) -> str:
    """Return a small pure-state label for the active canonical-content layer."""
    meta = active_page_generation(project_id)
    if not meta.get("generation_id"):
        return "missing"
    # Local import avoids a module-import cycle while keeping every public
    # generation-state view aligned with artifact/generation invalidation.
    from slide_tutor.generation import page_batch_parser_identity

    if meta.get("page_content_contract_signature") != page_content_contract_signature():
        return "stale"
    if meta.get("parser_identity") != page_batch_parser_identity():
        return "stale"
    if not isinstance(meta.get("page_count"), int) or not meta.get("page_manifest_signature"):
        return "stale"
    if not meta.get("storage_root"):
        return "stale"
    if source_signature and meta.get("source_signature") != source_signature:
        return "stale"
    return "current"
from slide_tutor.projects import utc_now


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))


def _nonempty_lines(text: str) -> int:
    return sum(bool(line.strip()) for line in text.splitlines())


def _formula_count(text: str) -> int:
    return len(re.findall(r"\$\$?|\\\(|\\\[|\\begin\{", text))


def _table_line_count(text: str) -> int:
    return sum("|" in line and line.count("|") >= 2 for line in text.splitlines())


def _list_count(text: str) -> int:
    return sum(bool(re.match(r"^\s*(?:[-*+] |\d+[.)] )", line)) for line in text.splitlines())


def _repeated_noise_ratio(text: str) -> float:
    lines = [line.strip().casefold() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    repeated = len(lines) - len(set(lines))
    return min(1.0, repeated / len(lines))


def _orphan_fragment_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    fragments = sum(1 for line in lines if len(line) <= 3 and not re.search(r"[。.!?！？]$", line))
    return min(1.0, fragments / len(lines))


def compute_quality_signals(
    pymupdf_text: str,
    marker_markdown: str,
    *,
    title: str = "",
    metadata: dict[str, Any] | None = None,
) -> PageQualitySignals:
    """Compute stable, explainable signals without I/O or model calls."""
    pdf_chars = len(pymupdf_text.strip())
    marker_chars = len(marker_markdown.strip())
    ratio = marker_chars / pdf_chars if pdf_chars else 0.0
    visual_ratio = 0.0
    span_count = 0
    if metadata:
        visual_ratio = float(metadata.get("visual_ratio", metadata.get("non_text_ratio", 0.0)) or 0.0)
        span_count = int(metadata.get("pymupdf_span_count", metadata.get("span_count", 0)) or 0)
    return PageQualitySignals(
        pymupdf_char_count=pdf_chars,
        marker_char_count=marker_chars,
        pymupdf_word_count=_word_count(pymupdf_text),
        marker_word_count=_word_count(marker_markdown),
        pymupdf_nonempty_lines=_nonempty_lines(pymupdf_text),
        marker_nonempty_lines=_nonempty_lines(marker_markdown),
        marker_to_pymupdf_ratio=ratio,
        marker_loss_ratio=max(0.0, min(1.0, 1.0 - ratio)) if pdf_chars else 0.0,
        title_retained_by_pymupdf=bool(title.strip() and title.casefold() in pymupdf_text.casefold()),
        title_retained_by_marker=bool(title.strip() and title.casefold() in marker_markdown.casefold()),
        pymupdf_formula_count=_formula_count(pymupdf_text),
        marker_formula_count=_formula_count(marker_markdown),
        pymupdf_table_line_count=_table_line_count(pymupdf_text),
        marker_table_line_count=_table_line_count(marker_markdown),
        pymupdf_list_count=_list_count(pymupdf_text),
        marker_list_count=_list_count(marker_markdown),
        pymupdf_span_count=max(0, span_count),
        visual_ratio=max(0.0, min(1.0, visual_ratio)),
        repeated_noise_ratio=max(_repeated_noise_ratio(pymupdf_text), _repeated_noise_ratio(marker_markdown)),
        orphan_fragment_ratio=max(_orphan_fragment_ratio(pymupdf_text), _orphan_fragment_ratio(marker_markdown)),
    )


def route_canonical_content(
    signals: PageQualitySignals,
    pymupdf_text: str,
    marker_markdown: str,
    *,
    title: str = "",
) -> tuple[PageRouteAction, str, str, list[str]]:
    """Select a canonical candidate using generic thresholds only."""
    pdf = pymupdf_text.strip()
    marker = marker_markdown.strip()
    warnings: list[str] = []
    if not pdf and not marker:
        return "empty", "", "none", ["Both PyMuPDF and Marker candidates are empty"]

    if pdf and marker and signals.pymupdf_char_count >= 80 and signals.marker_to_pymupdf_ratio < 0.35:
        warnings.append("Marker candidate appears severely incomplete; PyMuPDF selected")
        return "select", pdf, "pymupdf", warnings

    marker_has_structure = (
        signals.marker_formula_count > signals.pymupdf_formula_count
        or signals.marker_table_line_count > signals.pymupdf_table_line_count
        or signals.marker_list_count > signals.pymupdf_list_count
    )
    if marker and (not pdf or marker_has_structure or signals.marker_to_pymupdf_ratio >= 0.65):
        if title and title.casefold() not in marker.casefold():
            warnings.append("Title supplemented from SlidePage metadata")
            return "supplement", f"{title}\n\n{marker}".strip(), "marker", warnings
        return "select", marker, "marker", warnings

    if pdf:
        if signals.pymupdf_char_count <= 80 and signals.visual_ratio >= 0.25:
            warnings.append("Text is sparse and page has a visual-content signal")
            return "needs_vlm", pdf, "pymupdf", warnings
        if signals.pymupdf_char_count <= 80:
            warnings.append("Page contains only sparse but non-empty text")
            return "sparse_valid", pdf, "pymupdf", warnings
        return "select", pdf, "pymupdf", warnings

    if marker:
        if signals.marker_char_count <= 80 and signals.visual_ratio >= 0.25:
            warnings.append("Marker text is sparse and page has a visual-content signal")
            return "needs_vlm", marker, "marker", warnings
        return "sparse_valid", marker, "marker", ["Only a sparse Marker candidate was available"]
    return "empty", "", "none", ["No usable canonical candidate"]


def evidence_eligibility(page: PageUnderstanding) -> EvidenceEligibility:
    """Map page quality into the evidence contract consumed by Stage C."""
    if page.quality == "empty" or page.route_action == "empty" or not page.canonical_markdown.strip():
        return "ineligible"
    if page.quality == "needs_vlm" or page.route_action == "needs_vlm":
        return "needs_vlm_only"
    if page.quality in {"degraded", "fallback", "sparse_valid"} or page.route_action == "sparse_valid":
        return "degraded"
    return "eligible"


def is_evidence_eligible(page: PageUnderstanding, *, allow_degraded: bool = True) -> bool:
    eligibility = evidence_eligibility(page)
    return eligibility == "eligible" or (allow_degraded and eligibility == "degraded")


def load_page_understanding(project_id: str, file_id: str, slide_id: str) -> PageUnderstanding | None:
    """Read a previously cached PageUnderstanding.  Returns None if missing."""
    active_meta = active_page_generation(project_id)
    generation_root = active_meta.get("storage_root")
    if generation_root:
        generation_path = cache.get_project_dir(project_id) / generation_root / file_id / f"{slide_id}.json"
        data = cache.load_json(generation_path, default=None)
    else:
        data = cache.load_json(cache.page_understanding_path(project_id, file_id, slide_id))
    if not data:
        return None
    try:
        return PageUnderstanding.model_validate(data)
    except Exception:
        return None


def save_page_understanding(ud: PageUnderstanding, *, generation_id: str | None = None) -> None:
    """Persist a PageUnderstanding to its cache file."""
    if generation_id is not None and ud.generation_id != generation_id:
        ud = ud.model_copy(update={"generation_id": generation_id})
    cache.save_json(
        cache.page_understanding_path(ud.project_id, ud.file_id, ud.slide_id),
        ud,
    )


def ensure_page_understanding(
    project_id: str,
    file_id: str,
    slide: SlidePage,
    *,
    generation_id: str | None = None,
) -> PageUnderstanding:
    """Return a cached PageUnderstanding, or build+save a new one via Marker/PyMuPDF.
    This is the single entry point used by explain, translate, and index building."""
    active_meta = active_page_generation(project_id)
    cached = load_page_understanding(project_id, file_id, slide.slide_id)
    if cached is not None and (
        generation_id is None or cached.generation_id == generation_id
    ):
        return cached

    # Lazy import — Marker may not be installed
    from slide_tutor.page_parsers import parse_page

    ud = parse_page(project_id, file_id, slide)
    if active_meta.get("generation_id"):
        # Never patch an active generation or write an ignored legacy cache.
        # Explicit pre-index will stage and activate a complete replacement.
        return ud
    save_page_understanding(ud, generation_id=generation_id)
    return ud


def get_page_text_for_prompt(
    project_id: str,
    file_id: str,
    slide: SlidePage,
) -> PageTextResolution:
    """Resolve prompt text while exposing whether it belongs to active state."""
    active_meta = active_page_generation(project_id)
    active_state = page_generation_status(project_id)
    try:
        ud = ensure_page_understanding(project_id, file_id, slide)
        canonical = ud.canonical_markdown or ud.clean_markdown
        if canonical.strip() and ud.quality != "empty":
            if active_meta.get("generation_id") and (
                active_state != "current"
                or ud.generation_id != active_meta.get("generation_id")
            ):
                return PageTextResolution(
                    text=canonical,
                    quality=ud.quality,
                    source_state="ephemeral_degraded",
                    warning=(
                        "Active page generation is stale or missing this page; using "
                        "degraded page content. Refresh sources to restore generation "
                        "consistency."
                    ),
                )
            return PageTextResolution(
                text=canonical,
                quality=ud.quality,
                source_state=(
                    "current_generation" if active_meta.get("generation_id") else "legacy_cache"
                ),
            )
    except Exception:
        pass
    fallback_text = slide.text or "(No extractable text)"
    return PageTextResolution(
        text=fallback_text,
        quality="empty" if not slide.text.strip() else "fallback",
        source_state="slide_fallback",
        warning="Canonical page content was unavailable; using extracted slide text.",
    )

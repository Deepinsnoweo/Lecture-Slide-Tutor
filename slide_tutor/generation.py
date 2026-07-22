"""Page/index generation contracts and coherent page staging helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from slide_tutor import cache, config
from slide_tutor.marker_adapters import MarkerAdapter, default_marker_adapter
from slide_tutor.models import (
    GenerationLifecycle,
    GenerationState,
    IndexProgress,
    PageGenerationIntegrity,
    PageGenerationMetadata,
    PageUnderstanding,
    SlidePage,
)
from slide_tutor.page_understanding import (
    active_page_generation,
    page_content_contract_signature,
    page_generation_meta_path,
)


PAGE_BATCH_ARTIFACT_SCHEMA_VERSION = "page-batch-artifact-v1"
PAGE_BATCH_OUTPUT_SCHEMA = "PageUnderstanding-v1"
PAGE_BATCH_ADAPTER_CONTRACT = "bounded-document-batch-v1"
PAGE_BATCH_POLICY_VERSION = "header-footer-equation-table-v1"
PAGE_BATCH_CONTEXT_POLICY = "fixed-file-batch-context-v1"


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def new_generation_id() -> str:
    return f"gen-{uuid4().hex}"


def staged_page_root(project_id: str) -> Path:
    project_dir = cache.get_project_dir(project_id)
    project_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=".page-build-", dir=project_dir))


def page_json_path(root: Path, file_id: str, slide_id: str) -> Path:
    return root / config.UNDERSTANDING_DIRNAME / file_id / f"{slide_id}.json"


def page_batch_artifact_dir(project_id: str) -> Path:
    """Return the project-scoped store for reusable page-batch artifacts."""
    return (
        cache.get_project_dir(project_id)
        / config.PAGE_ARTIFACTS_DIRNAME
        / config.PAGE_BATCH_ARTIFACTS_DIRNAME
    )


def _file_content_identity(project_id: str, file_id: str) -> str:
    """Resolve the immutable file hash used by the PIO-8 content key.

    Tests and compatibility callers may build generations without a persisted
    project manifest.  In that case the file id is a deterministic fallback;
    production project-first ingestion always supplies ``ProjectFile.sha256``.
    """
    try:
        from slide_tutor.projects import load_project

        project = load_project(project_id)
        for file in project.files:
            if file.file_id == file_id:
                return file.sha256
    except Exception:
        pass
    return file_id


def page_batch_parser_identity(
    marker_adapter: MarkerAdapter | None = None,
) -> dict[str, str]:
    """Return the installed parser and active batch-adapter identity."""
    from slide_tutor.page_parsers import page_batch_cache_identity

    return page_batch_cache_identity(marker_adapter)


def page_batch_artifact_identity(
    project_id: str,
    file_id: str,
    slides: list[SlidePage],
    *,
    parser_identity: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the stable content identity for one parser batch.

    Deliberately excluded from this identity: device, worker count, process
    count, timing instrumentation and project file order.  Those are runtime
    facts, not content semantics.
    """
    page_numbers = [slide.page_number for slide in slides]
    return {
        "artifact_schema_version": PAGE_BATCH_ARTIFACT_SCHEMA_VERSION,
        "file_sha256": _file_content_identity(project_id, file_id),
        "page_numbers": page_numbers,
        "parser_family": "marker-pymupdf-quality-router",
        "parser_identity": dict(parser_identity or page_batch_parser_identity()),
        "parser_contract": page_content_contract_signature(),
        "adapter_contract": PAGE_BATCH_ADAPTER_CONTRACT,
        "output_schema": PAGE_BATCH_OUTPUT_SCHEMA,
        "mode_ocr_content_policy": {
            "marker_enabled": config.MARKER_ENABLED,
            "marker_local_files_only": config.MARKER_LOCAL_FILES_ONLY,
        },
        "header_footer_equation_table_policy": PAGE_BATCH_POLICY_VERSION,
        "batch_context_policy": PAGE_BATCH_CONTEXT_POLICY,
    }


def page_batch_artifact_key(identity: dict[str, Any]) -> str:
    """Hash a canonical artifact identity into a filesystem-safe key."""
    payload = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def page_batch_artifact_path(
    project_id: str,
    file_id: str,
    slides: list[SlidePage],
    *,
    parser_identity: dict[str, str] | None = None,
) -> Path:
    identity = page_batch_artifact_identity(
        project_id,
        file_id,
        slides,
        parser_identity=parser_identity,
    )
    return page_batch_artifact_dir(project_id) / f"{page_batch_artifact_key(identity)}.json"


def _save_json_atomic(path: Path, payload: Any) -> None:
    """Write a complete cache artifact before exposing its final filename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        cache.save_json(temporary, payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def generation_lifecycle_path(project_id: str) -> Path:
    """Return the project-scoped baseline/enhanced lifecycle metadata path."""
    return cache.get_project_dir(project_id) / config.GENERATION_LIFECYCLE_FILENAME


def _active_generation_facts(project_id: str) -> dict[str, Any]:
    """Read active page/index identifiers without changing project state."""
    from slide_tutor.index_store import _index_meta_path

    page_meta = active_page_generation(project_id)
    index_meta = cache.load_json(_index_meta_path(project_id), default={}) or {}
    page_id = page_meta.get("generation_id")
    index_id = index_meta.get("generation_id")
    page_source_signature = page_meta.get("source_signature", "")
    index_source_signature = index_meta.get("source_signature", "")
    source_signature = index_source_signature or page_source_signature
    current_parser_identity = page_batch_parser_identity()
    page_current = bool(
        page_id
        and page_meta.get("storage_root")
        and page_meta.get("page_manifest_signature")
        and page_meta.get("page_content_contract_signature") == page_content_contract_signature()
        and page_meta.get("parser_identity") == current_parser_identity
    )
    index_current = bool(
        index_id
        and index_meta.get("source_signature")
        and (not page_id or index_meta.get("page_generation_id") == page_id)
    )
    return {
        "page_generation_id": page_id,
        "index_generation_id": index_id,
        "source_signature": source_signature,
        "page_source_signature": page_source_signature,
        "index_source_signature": index_source_signature,
        "page_current": page_current,
        "index_current": index_current,
        "coherent": page_current and index_current and (
            not page_id or index_meta.get("page_generation_id") == page_id
        ) and (
            not page_source_signature
            or not index_source_signature
            or page_source_signature == index_source_signature
        ),
    }


def _derived_lifecycle(project_id: str, source_signature: str = "") -> GenerationLifecycle:
    """Derive a lifecycle for projects written before PIO-10 metadata existed."""
    facts = _active_generation_facts(project_id)
    source = source_signature or facts["source_signature"] or ""
    if facts["coherent"] and (
        not source_signature or facts["source_signature"] == source_signature
    ):
        return GenerationLifecycle(
            project_id=project_id,
            source_signature=source,
            lifecycle_status="baseline_current",
            active_layer="baseline",
            baseline_status="current",
            enhanced_status="stale",
            baseline_page_generation_id=facts["page_generation_id"],
            baseline_index_generation_id=facts["index_generation_id"],
            baseline_source_signature=source,
            enhanced_source_signature=source,
            updated_at=utc_now(),
        )
    return GenerationLifecycle(
        project_id=project_id,
        source_signature=source,
        lifecycle_status="stale",
        active_layer="baseline",
        baseline_status=("stale" if facts["page_generation_id"] or facts["index_generation_id"] else "missing"),
        enhanced_status="stale" if facts["page_generation_id"] or facts["index_generation_id"] else "missing",
        baseline_page_generation_id=facts["page_generation_id"],
        baseline_index_generation_id=facts["index_generation_id"],
        baseline_source_signature=facts["source_signature"],
        enhanced_source_signature=facts["source_signature"],
        updated_at=utc_now(),
        error="active page/index generation is missing, stale or incoherent",
    )


def load_generation_lifecycle(
    project_id: str,
    source_signature: str = "",
) -> GenerationLifecycle:
    """Load validated lifecycle metadata, deriving a safe legacy view if absent."""
    raw = cache.load_json(generation_lifecycle_path(project_id), default=None)
    lifecycle: GenerationLifecycle | None = None
    if isinstance(raw, dict):
        try:
            lifecycle = GenerationLifecycle.model_validate(raw)
        except Exception:
            lifecycle = None
    if lifecycle is None or lifecycle.project_id != project_id:
        return _derived_lifecycle(project_id, source_signature)

    requested_source = source_signature or lifecycle.source_signature
    if requested_source and lifecycle.source_signature and requested_source != lifecycle.source_signature:
        return lifecycle.model_copy(update={
            "source_signature": requested_source,
            "lifecycle_status": "stale",
            "active_layer": "baseline",
            "baseline_status": "stale" if lifecycle.baseline_page_generation_id or lifecycle.baseline_index_generation_id else "missing",
            "enhanced_status": "stale" if lifecycle.enhanced_page_generation_id or lifecycle.enhanced_index_generation_id else "missing",
            "error": "source signature changed; rebuild required",
        })

    facts = _active_generation_facts(project_id)
    expected_page = (
        lifecycle.enhanced_page_generation_id
        if lifecycle.active_layer == "enhanced"
        else lifecycle.baseline_page_generation_id
    )
    expected_index = (
        lifecycle.enhanced_index_generation_id
        if lifecycle.active_layer == "enhanced"
        else lifecycle.baseline_index_generation_id
    )
    if (
        lifecycle.lifecycle_status in {"baseline_current", "enhanced_current"}
        and (facts["page_generation_id"] != expected_page or facts["index_generation_id"] != expected_index)
    ):
        return lifecycle.model_copy(update={
            "lifecycle_status": "stale",
            "active_layer": "baseline",
            "baseline_status": "stale" if lifecycle.baseline_page_generation_id or lifecycle.baseline_index_generation_id else "missing",
            "enhanced_status": "stale" if lifecycle.enhanced_page_generation_id or lifecycle.enhanced_index_generation_id else "missing",
            "error": "active generation does not match lifecycle metadata",
        })
    return lifecycle


def _save_generation_lifecycle(lifecycle: GenerationLifecycle) -> GenerationLifecycle:
    lifecycle = lifecycle.model_copy(update={"updated_at": utc_now()})
    _save_json_atomic(generation_lifecycle_path(lifecycle.project_id), lifecycle)
    return lifecycle


def begin_baseline_build(project_id: str, source_signature: str = "") -> GenerationLifecycle:
    """Mark a baseline rebuild while keeping any previous active snapshot queryable."""
    current = load_generation_lifecycle(project_id, source_signature)
    source = source_signature or current.source_signature
    source_changed = bool(
        source and current.baseline_source_signature and current.baseline_source_signature != source
    )
    return _save_generation_lifecycle(current.model_copy(update={
        "source_signature": source,
        "lifecycle_status": "baseline_building",
        "baseline_status": "building",
        "enhanced_status": "stale" if source_changed else current.enhanced_status,
        "enhanced_page_generation_id": None if source_changed else current.enhanced_page_generation_id,
        "enhanced_index_generation_id": None if source_changed else current.enhanced_index_generation_id,
        "enhanced_source_signature": source if source_changed else current.enhanced_source_signature,
        "error": None,
        "started_at": utc_now(),
    }))


def mark_baseline_current(
    project_id: str,
    *,
    page_generation_id: str | None,
    index_generation_id: str | None,
    source_signature: str = "",
) -> GenerationLifecycle:
    """Record a successful baseline activation as one coherent snapshot."""
    current = load_generation_lifecycle(project_id, source_signature)
    source = source_signature or current.source_signature
    return _save_generation_lifecycle(current.model_copy(update={
        "source_signature": source,
        "lifecycle_status": "baseline_current",
        "active_layer": "baseline",
        "baseline_status": "current",
        "baseline_page_generation_id": page_generation_id,
        "baseline_index_generation_id": index_generation_id,
        "baseline_source_signature": source,
        "enhanced_status": "stale",
        "enhanced_page_generation_id": None,
        "enhanced_index_generation_id": None,
        "enhanced_source_signature": source,
        "error": None,
        "started_at": current.started_at,
    }))


def begin_enhancement(project_id: str, source_signature: str = "") -> GenerationLifecycle:
    """Start an enhanced build without changing the active baseline pointers."""
    current = load_generation_lifecycle(project_id, source_signature)
    if current.baseline_status != "current":
        raise RuntimeError("Cannot enhance before a current baseline generation exists")
    source = source_signature or current.source_signature
    return _save_generation_lifecycle(current.model_copy(update={
        "source_signature": source,
        "lifecycle_status": "enhancing",
        "active_layer": "baseline",
        "enhanced_status": "building",
        "error": None,
        "started_at": utc_now(),
    }))


def mark_enhancement_current(
    project_id: str,
    *,
    page_generation_id: str,
    index_generation_id: str,
    source_signature: str = "",
) -> GenerationLifecycle:
    """Atomically record that a complete enhanced snapshot is now active."""
    current = load_generation_lifecycle(project_id, source_signature)
    source = source_signature or current.source_signature
    return _save_generation_lifecycle(current.model_copy(update={
        "source_signature": source,
        "lifecycle_status": "enhanced_current",
        "active_layer": "enhanced",
        "baseline_status": "current",
        "enhanced_status": "current",
        "enhanced_page_generation_id": page_generation_id,
        "enhanced_index_generation_id": index_generation_id,
        "enhanced_source_signature": source,
        "error": None,
    }))


def mark_enhancement_failed(
    project_id: str,
    error: str,
    *,
    cancelled: bool = False,
) -> GenerationLifecycle:
    """Record enhancement failure/cancellation while retaining baseline pointers."""
    current = load_generation_lifecycle(project_id)
    baseline_ok = current.baseline_status == "current"
    return _save_generation_lifecycle(current.model_copy(update={
        "lifecycle_status": "enhancement_failed",
        "active_layer": "baseline",
        "baseline_status": "current" if baseline_ok else current.baseline_status,
        "enhanced_status": "failed",
        "error": ("cancelled: " if cancelled else "") + error,
    }))


def mark_baseline_recovered(project_id: str, error: str, *, cancelled: bool = False) -> GenerationLifecycle:
    """Restore a truthful baseline/stale state after an unactivated rebuild fails."""
    current = load_generation_lifecycle(project_id)
    facts = _active_generation_facts(project_id)
    baseline_ok = (
        current.baseline_status == "building"
        and facts["coherent"]
        and facts["page_generation_id"] == current.baseline_page_generation_id
        and facts["index_generation_id"] == current.baseline_index_generation_id
    )
    return _save_generation_lifecycle(current.model_copy(update={
        "lifecycle_status": "baseline_current" if baseline_ok else "stale",
        "active_layer": "baseline",
        "baseline_status": "current" if baseline_ok else "stale",
        "enhanced_status": "stale" if current.enhanced_status == "building" else current.enhanced_status,
        "error": ("cancelled: " if cancelled else "") + error,
    }))


def load_page_batch_artifact(
    project_id: str,
    file_id: str,
    slides: list[SlidePage],
    *,
    parser_identity: dict[str, str] | None = None,
) -> dict[str, PageUnderstanding] | None:
    """Load one complete, validated content artifact or return ``None``.

    A malformed or partial artifact is treated as a miss.  The caller then
    rebuilds only this batch; other artifact files remain untouched.
    """
    if not slides:
        return {}
    identity = page_batch_artifact_identity(
        project_id,
        file_id,
        slides,
        parser_identity=parser_identity,
    )
    path = page_batch_artifact_dir(project_id) / f"{page_batch_artifact_key(identity)}.json"
    try:
        payload = cache.load_json(path, default=None)
        if not isinstance(payload, dict):
            return None
        if payload.get("artifact_schema_version") != PAGE_BATCH_ARTIFACT_SCHEMA_VERSION:
            return None
        if payload.get("artifact_key") != page_batch_artifact_key(identity):
            return None
        if payload.get("identity") != identity:
            return None
        if payload.get("file_id") != file_id:
            return None
        if payload.get("slide_ids") != [slide.slide_id for slide in slides]:
            return None
        raw_pages = payload.get("pages")
        if not isinstance(raw_pages, list) or len(raw_pages) != len(slides):
            return None
        pages = [PageUnderstanding.model_validate(raw) for raw in raw_pages]
        expected_ids = {slide.slide_id for slide in slides}
        if {page.slide_id for page in pages} != expected_ids:
            return None
        if any(page.file_id != file_id for page in pages):
            return None
        return {page.slide_id: page.model_copy(update={"generation_id": None}) for page in pages}
    except Exception:
        return None


def save_page_batch_artifact(
    project_id: str,
    file_id: str,
    slides: list[SlidePage],
    pages: list[PageUnderstanding],
    *,
    parser_identity: dict[str, str] | None = None,
) -> Path:
    """Persist a complete parser batch as a generation-neutral artifact."""
    if len(slides) != len(pages):
        raise ValueError(
            f"Page artifact batch mismatch for {file_id}: "
            f"slides={len(slides)} pages={len(pages)}"
        )
    identity = page_batch_artifact_identity(
        project_id,
        file_id,
        slides,
        parser_identity=parser_identity,
    )
    artifact_key = page_batch_artifact_key(identity)
    page_by_id = {page.slide_id: page for page in pages}
    expected_ids = [slide.slide_id for slide in slides]
    if list(page_by_id) != expected_ids:
        raise ValueError(f"Page artifact order/identity mismatch for {file_id}")
    if any(page.file_id != file_id for page in pages):
        raise ValueError(f"Page artifact file mismatch for {file_id}")
    path = page_batch_artifact_dir(project_id) / f"{artifact_key}.json"
    payload = {
        "artifact_schema_version": PAGE_BATCH_ARTIFACT_SCHEMA_VERSION,
        "artifact_key": artifact_key,
        "identity": identity,
        "file_id": file_id,
        "slide_ids": expected_ids,
        "page_count": len(pages),
        "created_at": utc_now(),
        "pages": [
            page.model_copy(update={"generation_id": None}).model_dump(mode="json")
            for page in pages
        ],
    }
    _save_json_atomic(path, payload)
    return path


def load_page_from_root(root: Path, file_id: str, slide_id: str) -> PageUnderstanding | None:
    data = cache.load_json(page_json_path(root, file_id, slide_id), default=None)
    if not data:
        return None
    try:
        return PageUnderstanding.model_validate(data)
    except Exception:
        return None


def save_page_to_root(root: Path, page: PageUnderstanding) -> None:
    cache.save_json(page_json_path(root, page.file_id, page.slide_id), page)


def write_generation_metadata(root: Path, metadata: PageGenerationMetadata) -> None:
    cache.save_json(root / "page_generation_meta.json", metadata)


def page_manifest_signature(
    files_and_slides: list[tuple[str, list[SlidePage]]],
) -> str:
    identities = [
        {
            "file_id": file_id,
            "slide_id": slide.slide_id,
            "page_number": slide.page_number,
        }
        for file_id, slides in files_and_slides
        for slide in slides
    ]
    return hashlib.sha256(
        json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_active_page_generation(
    project_id: str,
    files_and_slides: list[tuple[str, list[SlidePage]]],
    *,
    source_signature: str,
) -> PageGenerationIntegrity:
    """Validate metadata, manifest and every page without mutating caches."""
    raw_meta = active_page_generation(project_id)
    expected_count = sum(len(slides) for _file_id, slides in files_and_slides)
    if not raw_meta.get("generation_id"):
        return PageGenerationIntegrity(
            status="missing", reason="no active page generation",
            expected_count=expected_count, valid_count=0,
        )
    try:
        meta = PageGenerationMetadata.model_validate(raw_meta)
    except Exception as exc:
        return PageGenerationIntegrity(
            status="stale", reason=f"legacy or invalid generation metadata: {exc}",
            expected_count=expected_count, valid_count=0,
        )
    expected_manifest = page_manifest_signature(files_and_slides)
    current_parser_identity = page_batch_parser_identity()
    if (
        meta.source_signature != source_signature
        or meta.page_content_contract_signature != page_content_contract_signature()
        or meta.parser_identity != current_parser_identity
        or meta.page_count != expected_count
        or meta.page_manifest_signature != expected_manifest
    ):
        return PageGenerationIntegrity(
            status="stale",
            reason="source, parser, contract or page manifest changed",
            expected_count=expected_count, valid_count=0,
        )
    if not meta.storage_root:
        return PageGenerationIntegrity(
            status="corrupt", reason="active generation has no storage_root",
            expected_count=expected_count, valid_count=0,
        )
    root = cache.get_project_dir(project_id) / meta.storage_root
    valid_count = 0
    for file_id, slides in files_and_slides:
        for slide in slides:
            data = cache.load_json(root / file_id / f"{slide.slide_id}.json", default=None)
            try:
                page = PageUnderstanding.model_validate(data)
            except Exception:
                return PageGenerationIntegrity(
                    status="corrupt",
                    reason=f"missing or invalid page: {file_id}/{slide.slide_id}",
                    expected_count=expected_count,
                    valid_count=valid_count,
                )
            if page.generation_id != meta.generation_id:
                return PageGenerationIntegrity(
                    status="corrupt",
                    reason=f"generation mismatch: {file_id}/{slide.slide_id}",
                    expected_count=expected_count,
                    valid_count=valid_count,
                )
            valid_count += 1
    return PageGenerationIntegrity(
        status="current", reason="active page generation is complete",
        expected_count=expected_count, valid_count=valid_count,
    )


def build_page_generation(
    project_id: str,
    files_and_slides: list[tuple[str, list[SlidePage]]],
    *,
    source_signature: str,
    cancel_signal: Any = None,
    force_reprocess: bool = False,
    progress_callback: Callable[[IndexProgress], None] | None = None,
    marker_adapter: MarkerAdapter | None = None,
) -> tuple[Path, PageGenerationMetadata]:
    """Build a complete page generation without mutating the active cache."""
    from slide_tutor.index_store import _check_cancelled
    from slide_tutor.page_parsers import parse_pages
    from slide_tutor.page_understanding import load_page_understanding

    generation_id = new_generation_id()
    adapter = marker_adapter or default_marker_adapter()
    parser_identity = page_batch_parser_identity(adapter)
    root = staged_page_root(project_id)
    total_pages = sum(len(slides) for _file_id, slides in files_and_slides)
    operation_started = time.perf_counter()
    completed_pages = 0
    reused_pages = 0
    if progress_callback is not None:
        progress_callback(IndexProgress(
            stage="build page content", status="started", total=total_pages,
        ))
    try:
        active_meta = active_page_generation(project_id)
        can_reuse = (
            not force_reprocess
            and active_meta.get("source_signature") == source_signature
            and active_meta.get("page_content_contract_signature") == page_content_contract_signature()
            and active_meta.get("parser_identity") == parser_identity
        )
        batch_size = max(1, config.MARKER_BATCH_PAGES)
        for file_id, slides in files_and_slides:
            for batch_start in range(0, len(slides), batch_size):
                slide_batch = slides[batch_start:batch_start + batch_size]
                _check_cancelled(cancel_signal)
                cached_pages: dict[str, PageUnderstanding] | None = None
                reuse_detail = ""
                if not force_reprocess:
                    cached_pages = load_page_batch_artifact(
                        project_id,
                        file_id,
                        list(slide_batch),
                        parser_identity=parser_identity,
                    )
                    if cached_pages is not None:
                        reuse_detail = "content-addressed page-batch artifact"

                if cached_pages is None and can_reuse:
                    active_cached_pages = {
                        slide.slide_id: load_page_understanding(
                            project_id, file_id, slide.slide_id,
                        )
                        for slide in slide_batch
                    }
                    if all(
                        active_cached_pages.get(slide.slide_id) is not None
                        for slide in slide_batch
                    ):
                        cached_pages = active_cached_pages
                        # Promote a complete legacy/current generation batch into
                        # the content-addressed store without changing its text.
                        save_page_batch_artifact(
                            project_id,
                            file_id,
                            list(slide_batch),
                            [active_cached_pages[slide.slide_id] for slide in slide_batch],
                            parser_identity=parser_identity,
                        )
                        reuse_detail = "active page generation promoted to artifact"

                # Marker header/title filtering is batch-sensitive. Rebuild the
                # whole bounded batch if any member is absent so one generation
                # never mixes outputs produced under different page context.
                missing_slides = [] if cached_pages is not None else list(slide_batch)
                parsed_pages: dict[str, PageUnderstanding] = {}
                if missing_slides:
                    if progress_callback is not None:
                        progress_callback(IndexProgress(
                            stage="build page content",
                            status="progress",
                            completed=completed_pages,
                            total=total_pages,
                            reused=reused_pages,
                            elapsed_seconds=time.perf_counter() - operation_started,
                            detail=(
                                f"Parsing {file_id} pages "
                                f"{missing_slides[0].page_number}-"
                                f"{missing_slides[-1].page_number}"
                            ),
                        ))
                    parsed_pages = {
                        page.slide_id: page
                        for page in parse_pages(
                            project_id,
                            file_id,
                            missing_slides,
                            cancel_check=lambda: _check_cancelled(cancel_signal),
                            marker_adapter=adapter,
                        )
                    }
                    missing_ids = {slide.slide_id for slide in missing_slides}
                    if set(parsed_pages) != missing_ids:
                        raise RuntimeError(
                            f"Parser batch returned incomplete pages for {file_id}"
                        )
                    save_page_batch_artifact(
                        project_id,
                        file_id,
                        list(slide_batch),
                        [parsed_pages[slide.slide_id] for slide in slide_batch],
                        parser_identity=parser_identity,
                    )
                    cached_pages = {}

                for slide in slide_batch:
                    _check_cancelled(cancel_signal)
                    page = cached_pages.get(slide.slide_id) if cached_pages is not None else None
                    was_reused = page is not None
                    if was_reused:
                        reused_pages += 1
                    else:
                        page = parsed_pages[slide.slide_id]
                    page = page.model_copy(update={"generation_id": generation_id})
                    save_page_to_root(root, page)
                    completed_pages += 1
                    if progress_callback is not None:
                        progress_callback(IndexProgress(
                            stage="build page content",
                            status="progress",
                            completed=completed_pages,
                            total=total_pages,
                            reused=reused_pages,
                            elapsed_seconds=time.perf_counter() - operation_started,
                            detail=reuse_detail if was_reused else "",
                        ))
        metadata = PageGenerationMetadata(
            generation_id=generation_id,
            project_id=project_id,
            source_signature=source_signature,
            page_content_contract_signature=page_content_contract_signature(),
            parser_identity=parser_identity,
            page_count=total_pages,
            page_manifest_signature=page_manifest_signature(files_and_slides),
            created_at=utc_now(),
            status="current",
        )
        write_generation_metadata(root, metadata)
        if progress_callback is not None:
            progress_callback(IndexProgress(
                stage="build page content",
                status="completed",
                completed=completed_pages,
                total=total_pages,
                reused=reused_pages,
                elapsed_seconds=time.perf_counter() - operation_started,
            ))
        return root, metadata
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def generation_state(project_id: str, source_signature: str = "") -> GenerationState:
    """Compute page/index state without mutating caches."""
    from slide_tutor.index_store import _index_meta_path

    page_meta = active_page_generation(project_id)
    index_meta = cache.load_json(_index_meta_path(project_id), default={}) or {}
    current_parser_identity = page_batch_parser_identity()
    page_status = "missing"
    if page_meta.get("generation_id"):
        page_status = (
            "current"
            if page_meta.get("page_content_contract_signature") == page_content_contract_signature()
            and page_meta.get("parser_identity") == current_parser_identity
            and isinstance(page_meta.get("page_count"), int)
            and bool(page_meta.get("page_manifest_signature"))
            and bool(page_meta.get("storage_root"))
            and (not source_signature or page_meta.get("source_signature") == source_signature)
            else "stale"
        )
    index_status = "missing"
    if index_meta.get("generation_id"):
        index_status = (
            "current"
            if (not source_signature or index_meta.get("source_signature") == source_signature)
            and index_meta.get("page_generation_id") == page_meta.get("generation_id")
            else "stale"
        )
    lifecycle = load_generation_lifecycle(project_id, source_signature)
    return GenerationState(
        page_status=page_status,
        index_status=index_status,
        active_generation_id=page_meta.get("generation_id"),
        active_index_generation_id=index_meta.get("generation_id"),
        source_signature=source_signature,
        page_content_contract_signature=page_meta.get("page_content_contract_signature", ""),
        retrieval_contract_signature=index_meta.get("retrieval_contract_signature", ""),
        detail="page and index generations must match before current is reported",
        lifecycle_status=lifecycle.lifecycle_status,
        active_layer=lifecycle.active_layer,
        baseline_status=lifecycle.baseline_status,
        enhanced_status=lifecycle.enhanced_status,
        baseline_page_generation_id=lifecycle.baseline_page_generation_id,
        baseline_index_generation_id=lifecycle.baseline_index_generation_id,
        enhanced_page_generation_id=lifecycle.enhanced_page_generation_id,
        enhanced_index_generation_id=lifecycle.enhanced_index_generation_id,
        enhancement_error=lifecycle.error,
    )

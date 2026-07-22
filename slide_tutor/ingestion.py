from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any

import fitz

from slide_tutor import cache, config
from slide_tutor.models import FileSlidesPayload, ModuleManifest, ProjectFile, SlidePage, SlidesPayload, SourceFile
from slide_tutor.projects import ProjectUploadResult, load_project, upsert_project_file, utc_now
from slide_tutor.thumbnails import render_pdf_pages


# ══════════════════════════════════════════════════════════════════════════════
# Legacy Phase 1.0 module-based ingestion (deprecated).
# These are retained only for old-cache compatibility under .slide_tutor_data/modules/.
# New code MUST use add_uploads_to_project() + projects.py for project-first ingestion.
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SavedUpload:
    raw_name: str
    safe_name: str
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ModuleLoadResult:
    module_id: str
    module_dir: Path
    manifest: ModuleManifest
    slides: list[SlidePage]
    used_cache: bool


class IngestionError(RuntimeError):
    pass


def safe_upload_name(raw_name: str, fallback_index: int, used_names: set[str]) -> str:
    base_name = PurePath(PureWindowsPath(str(raw_name)).name).name
    sanitized = re.sub(r"[^A-Za-z0-9._ -]+", "_", base_name).strip(" .")
    if not sanitized:
        sanitized = f"upload-{fallback_index}.pdf"

    candidate_path = Path(sanitized)
    suffix = candidate_path.suffix.lower()
    if suffix not in config.SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(config.SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type for {raw_name!r}. Supported types: {supported}")

    stem = candidate_path.stem.strip(" .") or f"upload-{fallback_index}"
    candidate = f"{stem}{suffix}"
    counter = 2
    while candidate.lower() in used_names:
        candidate = f"{stem}-{counter}{suffix}"
        counter += 1

    used_names.add(candidate.lower())
    return candidate


def save_uploaded_files(uploaded_files: list[Any], module_dir: Path) -> list[SavedUpload]:
    uploads_dir = module_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    saved: list[SavedUpload] = []
    used_names: set[str] = set()
    for idx, uploaded in enumerate(uploaded_files, start=1):
        raw_name = str(getattr(uploaded, "name", f"upload-{idx}.pdf"))
        safe_name = safe_upload_name(raw_name, idx, used_names)
        data = cache.uploaded_bytes(uploaded)
        digest = hashlib.sha256(data).hexdigest()
        path = uploads_dir / safe_name
        path.write_bytes(data)
        saved.append(
            SavedUpload(
                raw_name=raw_name,
                safe_name=safe_name,
                path=path,
                sha256=digest,
                size_bytes=len(data),
            )
        )
    return saved


def _normalize_text(raw_text: str) -> str:
    lines = [line.rstrip() for line in raw_text.replace("\r\n", "\n").splitlines()]
    text = "\n".join(lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def guess_title(text: str) -> str | None:
    for line in text.splitlines():
        candidate = line.strip()
        if len(candidate) < 3:
            continue
        if re.fullmatch(r"\d+", candidate):
            continue
        return candidate[:120]
    return None


def classify_text_quality(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "empty"
    useful_chars = sum(ch.isalnum() for ch in stripped)
    if useful_chars < config.TEXT_LOW_CHAR_THRESHOLD:
        return "low"
    return "ok"


def _slide_id_for(path: Path, page_number: int, prefix: str | None = None) -> str:
    stem = prefix or re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")
    if not stem:
        stem = "slide"
    return f"{stem}-p{page_number}"


def extract_pdf_slides(
    path: Path,
    module_id: str,
    global_start_index: int,
    *,
    project_id: str | None = None,
    file_id: str | None = None,
    display_name: str | None = None,
) -> list[SlidePage]:
    slides: list[SlidePage] = []

    with fitz.open(path) as document:
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            page_number = page_index + 1
            text = _normalize_text(page.get_text("text") or "")
            slides.append(
                SlidePage(
                    slide_id=_slide_id_for(path, page_number, prefix=file_id),
                    module_id=module_id,
                    project_id=project_id,
                    file_id=file_id,
                    source_file=display_name or path.name,
                    source_type="pdf",
                    page_number=page_number,
                    global_index=global_start_index + page_index,
                    title_guess=guess_title(text),
                    text=text,
                    text_quality=classify_text_quality(text),
                    metadata={"page_count": document.page_count},
                )
            )

    return slides


# ── End of legacy module-based ingestion ──
# ── Project-first ingestion (current) ──


def _existing_safe_names(project_files: list[ProjectFile]) -> set[str]:
    return {file.safe_name.lower() for file in project_files}


def add_uploads_to_project(project_id: str, uploaded_files: list[Any]) -> ProjectUploadResult:
    if not uploaded_files:
        raise ValueError("Please upload at least one PDF file.")

    project = load_project(project_id)
    added: list[ProjectFile] = []
    skipped: list[str] = []
    failed: list[str] = []
    used_names = _existing_safe_names(project.files)

    for idx, uploaded in enumerate(uploaded_files, start=1):
        raw_name = str(getattr(uploaded, "name", f"upload-{idx}.pdf"))
        file_record: ProjectFile | None = None
        try:
            data = cache.uploaded_bytes(uploaded)
            file_id = cache.compute_file_id(data)
            digest = cache.compute_file_digest(data)

            if any(file.file_id == file_id for file in project.files):
                skipped.append(raw_name)
                continue

            safe_name = safe_upload_name(raw_name, idx, used_names)
            paths = cache.ensure_project_file_dirs(project_id, file_id)
            source_path = paths["file"] / "source.pdf"
            source_path.write_bytes(data)

            file_record = ProjectFile(
                file_id=file_id,
                original_name=raw_name,
                safe_name=safe_name,
                sha256=digest,
                size_bytes=len(data),
                page_count=0,
                created_at=utc_now(),
                status="processing",
            )
            project = upsert_project_file(project_id, file_record)

            slides = extract_pdf_slides(
                source_path,
                project_id,
                1,
                project_id=project_id,
                file_id=file_id,
                display_name=raw_name,
            )
            slides = render_pdf_pages(
                path=source_path,
                slides=slides,
                pages_dir=paths["pages"],
                thumbnails_dir=paths["thumbnails"],
            )
            payload = FileSlidesPayload(project_id=project_id, file_id=file_id, slides=slides)
            cache.save_json(cache.project_file_slides_path(project_id, file_id), payload)

            file_record = file_record.model_copy(update={"page_count": len(slides), "status": "ready"})
            project = upsert_project_file(project_id, file_record)
            added.append(file_record)
        except Exception as exc:
            failed.append(f"{raw_name}: {exc}")
            try:
                if file_record is not None:
                    failed_record = file_record.model_copy(update={"status": "failed", "error_message": str(exc)})
                    project = upsert_project_file(project_id, failed_record)
            except Exception:
                pass

    return ProjectUploadResult(
        project=load_project(project_id),
        added_files=added,
        skipped_files=skipped,
        failed_files=failed,
    )



def _load_cached_module(module_id: str) -> ModuleLoadResult | None:
    module_dir = cache.get_module_dir(module_id)
    manifest_data = cache.load_json(cache.manifest_path(module_id))
    slides_data = cache.load_json(cache.slides_path(module_id))
    if not manifest_data or not slides_data:
        return None

    manifest = ModuleManifest.model_validate(manifest_data)
    if manifest.status != "ready":
        return None

    payload = SlidesPayload.model_validate(slides_data)
    return ModuleLoadResult(
        module_id=module_id,
        module_dir=module_dir,
        manifest=manifest,
        slides=payload.slides,
        used_cache=True,
    )


def _save_failed_manifest(
    module_id: str,
    source_files: list[SourceFile],
    slide_count: int,
    error: Exception,
) -> None:
    manifest = ModuleManifest(
        module_id=module_id,
        created_at=utc_now(),
        source_files=source_files,
        slide_count=slide_count,
        status="failed",
        error_message=str(error),
    )
    cache.save_json(cache.manifest_path(module_id), manifest)


def load_or_process_uploads(uploaded_files: list[Any]) -> ModuleLoadResult:
    if not uploaded_files:
        raise ValueError("Please upload at least one PDF file.")

    module_id = cache.compute_module_id(uploaded_files)
    paths = cache.ensure_module_dirs(module_id)
    module_dir = paths["module"]

    cached = _load_cached_module(module_id)
    if cached is not None:
        return cached

    source_files: list[SourceFile] = []
    slides: list[SlidePage] = []

    try:
        saved_files = save_uploaded_files(uploaded_files, module_dir)
        global_index = 1

        for saved in saved_files:
            file_slides = extract_pdf_slides(saved.path, module_id, global_index)
            file_slides = render_pdf_pages(
                path=saved.path,
                slides=file_slides,
                pages_dir=paths["pages"],
                thumbnails_dir=paths["thumbnails"],
            )
            slides.extend(file_slides)
            source_files.append(
                SourceFile(
                    file_name=saved.raw_name,
                    safe_name=saved.safe_name,
                    sha256=saved.sha256,
                    size_bytes=saved.size_bytes,
                    page_count=len(file_slides),
                )
            )
            global_index += len(file_slides)

        manifest = ModuleManifest(
            module_id=module_id,
            created_at=utc_now(),
            source_files=source_files,
            slide_count=len(slides),
            status="ready",
        )
        payload = SlidesPayload(module_id=module_id, slides=slides)
        cache.save_json(cache.manifest_path(module_id), manifest)
        cache.save_json(cache.slides_path(module_id), payload)

        return ModuleLoadResult(
            module_id=module_id,
            module_dir=module_dir,
            manifest=manifest,
            slides=slides,
            used_cache=False,
        )
    except Exception as exc:
        _save_failed_manifest(module_id, source_files, len(slides), exc)
        raise IngestionError(f"Failed to process uploaded slides: {exc}") from exc

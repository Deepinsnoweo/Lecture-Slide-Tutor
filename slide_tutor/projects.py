from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime

from slide_tutor import cache, config
from slide_tutor.models import FileSlidesPayload, ProjectFile, ProjectManifest, SlidePage


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return slug[:48] or "project"


def create_project(name: str) -> ProjectManifest:
    now = utc_now()
    slug = _slugify(name)
    project_id = f"{slug}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
    project = ProjectManifest(
        project_id=project_id,
        name=name,
        created_at=now,
        updated_at=now,
    )
    cache.ensure_project_dirs(project_id)
    save_project(project)
    return project


def list_projects() -> list[ProjectManifest]:
    config.ensure_data_root()
    projects: list[ProjectManifest] = []
    for project_file in sorted(config.PROJECTS_DIR.glob(f"*/{config.PROJECT_FILENAME}")):
        data = cache.load_json(project_file)
        if not data:
            continue
        try:
            projects.append(ProjectManifest.model_validate(data))
        except Exception:
            continue
    return sorted(projects, key=lambda project: project.updated_at, reverse=True)


def load_project(project_id: str) -> ProjectManifest:
    data = cache.load_json(cache.project_path(project_id))
    if not data:
        raise FileNotFoundError(f"Project not found: {project_id}")
    return ProjectManifest.model_validate(data)


def save_project(project: ProjectManifest) -> None:
    cache.ensure_project_dirs(project.project_id)
    cache.save_json(cache.project_path(project.project_id), project)


def ordered_files(project: ProjectManifest) -> list[ProjectFile]:
    by_id = {file.file_id: file for file in project.files}
    ordered: list[ProjectFile] = []
    seen: set[str] = set()
    for file_id in project.file_order:
        file = by_id.get(file_id)
        if file is None:
            continue
        ordered.append(file)
        seen.add(file_id)
    for file in project.files:
        if file.file_id not in seen:
            ordered.append(file)
    return ordered


def update_file_order(project_id: str, ordered_file_ids: list[str]) -> ProjectManifest:
    project = load_project(project_id)
    known = {file.file_id for file in project.files}
    clean_order = [file_id for file_id in ordered_file_ids if file_id in known]
    for file in project.files:
        if file.file_id not in clean_order:
            clean_order.append(file.file_id)
    project = project.model_copy(update={"file_order": clean_order, "updated_at": utc_now()})
    save_project(project)
    return project


def upsert_project_file(project_id: str, file: ProjectFile) -> ProjectManifest:
    project = load_project(project_id)
    files = [existing for existing in project.files if existing.file_id != file.file_id]
    files.append(file)
    file_order = list(project.file_order)
    if file.file_id not in file_order:
        file_order.append(file.file_id)
    project = project.model_copy(update={"files": files, "file_order": file_order, "updated_at": utc_now()})
    save_project(project)
    return project


def load_file_slides(project_id: str, file_id: str) -> list[SlidePage]:
    data = cache.load_json(cache.project_file_slides_path(project_id, file_id))
    if not data:
        return []
    return FileSlidesPayload.model_validate(data).slides


@dataclass(frozen=True)
class ProjectUploadResult:
    project: ProjectManifest
    added_files: list[ProjectFile]
    skipped_files: list[str]
    failed_files: list[str]


# ── Delete operations ──


def delete_project_file(project_id: str, file_id: str) -> ProjectManifest | None:
    """Remove a single PDF from the project and delete its cached data from disk."""
    project = load_project(project_id)
    # Remove the file record and its position in file_order
    project = project.model_copy(update={
        "files": [f for f in project.files if f.file_id != file_id],
        "file_order": [fid for fid in project.file_order if fid != file_id],
        "updated_at": utc_now(),
    })
    save_project(project)
    # Delete the per-file directory (pages, thumbnails, slides.json, source.pdf)
    file_dir = cache.get_project_file_dir(project_id, file_id)
    if file_dir.exists():
        shutil.rmtree(file_dir, ignore_errors=True)
    return project


def delete_project(project_id: str) -> None:
    """Delete an entire project and all its data from disk."""
    project_dir = cache.get_project_dir(project_id)
    if project_dir.exists():
        shutil.rmtree(project_dir, ignore_errors=True)

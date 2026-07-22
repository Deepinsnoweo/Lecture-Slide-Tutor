"""Read-only Stage C storage inventory and conservative scale decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from slide_tutor import config


StorageCategory = Literal[
    "source", "page_generation", "images", "index", "quiz", "history", "other",
]


class StorageCategoryStat(BaseModel):
    category: StorageCategory
    file_count: int = Field(ge=0)
    size_bytes: int = Field(ge=0)


class ProjectStorageStat(BaseModel):
    project_id: str
    total_bytes: int = Field(ge=0)
    categories: list[StorageCategoryStat] = Field(default_factory=list)


class StorageAuditReport(BaseModel):
    data_root: str
    total_bytes: int = Field(ge=0)
    file_count: int = Field(ge=0)
    categories: list[StorageCategoryStat] = Field(default_factory=list)
    projects: list[ProjectStorageStat] = Field(default_factory=list)
    decisions: dict[str, str | bool | int | float] = Field(default_factory=dict)


def classify_storage_file(path: Path) -> StorageCategory:
    """Classify by stable project-first path/name contracts, not file content."""
    parts = {part.casefold() for part in path.parts}
    name = path.name.casefold()
    if "uploads" in parts or name == "source.pdf" or path.suffix.casefold() == ".pdf":
        return "source"
    if "pages" in parts or "thumbnails" in parts or path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".webp"}:
        return "images"
    if "index" in parts or name in {"chunks.json", "embeddings.npy"}:
        return "index"
    if "understanding" in parts or name in {
        "slides.json", "page_generation_meta.json", "generation_lifecycle.json",
    }:
        return "page_generation"
    if name == "qa_answer_history_v1.json":
        return "history"
    if "quiz_attempts" in parts or name.startswith((
        "quiz_cache_", "reviewed_exam_bundle_", "reviewed_exam_metadata_",
    )) or name == "topic_registry.json":
        return "quiz"
    return "other"


def _stats(paths: list[Path], root: Path) -> list[StorageCategoryStat]:
    totals = {
        category: [0, 0]
        for category in ("source", "page_generation", "images", "index", "quiz", "history", "other")
    }
    for path in paths:
        category = classify_storage_file(path.relative_to(root))
        totals[category][0] += 1
        totals[category][1] += path.stat().st_size
    return [
        StorageCategoryStat(category=category, file_count=values[0], size_bytes=values[1])
        for category, values in totals.items()
    ]


def audit_storage(data_root: Path | None = None) -> StorageAuditReport:
    """Inventory current data without deleting, touching, or rewriting artifacts."""
    root = (data_root or config.DATA_ROOT).resolve()
    files = [path for path in root.rglob("*") if path.is_file()] if root.exists() else []
    project_root = root / "projects"
    projects: list[ProjectStorageStat] = []
    if project_root.exists():
        for project_dir in sorted(path for path in project_root.iterdir() if path.is_dir()):
            project_files = [path for path in project_dir.rglob("*") if path.is_file()]
            projects.append(ProjectStorageStat(
                project_id=project_dir.name,
                total_bytes=sum(path.stat().st_size for path in project_files),
                categories=_stats(project_files, root),
            ))
    total_bytes = sum(path.stat().st_size for path in files)
    largest_project = max((project.total_bytes for project in projects), default=0)
    needs_lru = total_bytes >= 5 * 1024**3 or largest_project >= 1024**3
    needs_heavy_vector = len(projects) >= 100 or largest_project >= 5 * 1024**3
    return StorageAuditReport(
        data_root=str(root),
        total_bytes=total_bytes,
        file_count=len(files),
        categories=_stats(files, root),
        projects=projects,
        decisions={
            "implement_lru_now": needs_lru,
            "implement_heavy_vector_now": needs_heavy_vector,
            "delete_legacy_data": False,
            "lru_global_threshold_bytes": 5 * 1024**3,
            "lru_project_threshold_bytes": 1024**3,
            "basis": "measured_capacity_thresholds",
        },
    )


def main() -> None:
    print(json.dumps(audit_storage().model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

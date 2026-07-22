from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from slide_tutor import config


def _uploaded_name(uploaded: Any) -> str:
    return str(getattr(uploaded, "name", "upload.pdf"))


def uploaded_bytes(uploaded: Any) -> bytes:
    if hasattr(uploaded, "getvalue"):
        return uploaded.getvalue()
    if isinstance(uploaded, (bytes, bytearray)):
        return bytes(uploaded)
    if hasattr(uploaded, "read"):
        position = uploaded.tell() if hasattr(uploaded, "tell") else None
        data = uploaded.read()
        if position is not None and hasattr(uploaded, "seek"):
            uploaded.seek(position)
        return data
    raise TypeError(f"Unsupported uploaded file object: {type(uploaded)!r}")


def compute_module_id(uploaded_files: list[Any]) -> str:
    file_parts: list[tuple[str, str, int]] = []
    for uploaded in uploaded_files:
        data = uploaded_bytes(uploaded)
        digest = hashlib.sha256(data).hexdigest()
        file_parts.append((_uploaded_name(uploaded).lower(), digest, len(data)))

    root = hashlib.sha256()
    for name, digest, size in sorted(file_parts):
        root.update(name.encode("utf-8"))
        root.update(b"\0")
        root.update(digest.encode("ascii"))
        root.update(b"\0")
        root.update(str(size).encode("ascii"))
        root.update(b"\n")

    return root.hexdigest()[:16]


def compute_file_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_file_id(data: bytes) -> str:
    return compute_file_digest(data)[:16]


def get_module_dir(module_id: str) -> Path:
    return config.MODULES_DIR / module_id


def get_project_dir(project_id: str) -> Path:
    return config.PROJECTS_DIR / project_id


def project_path(project_id: str) -> Path:
    return get_project_dir(project_id) / config.PROJECT_FILENAME


def get_project_file_dir(project_id: str, file_id: str) -> Path:
    return get_project_dir(project_id) / "files" / file_id


def project_file_slides_path(project_id: str, file_id: str) -> Path:
    return get_project_file_dir(project_id, file_id) / config.SLIDES_FILENAME


def manifest_path(module_id: str) -> Path:
    return get_module_dir(module_id) / config.MANIFEST_FILENAME


def slides_path(module_id: str) -> Path:
    return get_module_dir(module_id) / config.SLIDES_FILENAME


def ensure_module_dirs(module_id: str) -> dict[str, Path]:
    config.ensure_data_root()
    module_dir = get_module_dir(module_id)
    paths = {
        "module": module_dir,
        "uploads": module_dir / "uploads",
        "pages": module_dir / "pages",
        "thumbnails": module_dir / "thumbnails",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def ensure_project_dirs(project_id: str) -> dict[str, Path]:
    config.ensure_data_root()
    project_dir = get_project_dir(project_id)
    paths = {
        "project": project_dir,
        "files": project_dir / "files",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def ensure_project_file_dirs(project_id: str, file_id: str) -> dict[str, Path]:
    ensure_project_dirs(project_id)
    file_dir = get_project_file_dir(project_id, file_id)
    paths = {
        "file": file_dir,
        "pages": file_dir / "pages",
        "thumbnails": file_dir / "thumbnails",
        "understanding": file_dir / config.UNDERSTANDING_DIRNAME,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


# ── Phase 2 Advance: page understanding cache paths ──


def page_understanding_path(project_id: str, file_id: str, slide_id: str) -> Path:
    """Cache path for a single page's PageUnderstanding JSON."""
    return get_project_file_dir(project_id, file_id) / config.UNDERSTANDING_DIRNAME / f"{slide_id}.json"


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))

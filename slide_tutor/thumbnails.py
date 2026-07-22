from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import fitz

from slide_tutor import config
from slide_tutor.models import SlidePage

logger = logging.getLogger(__name__)


def _render_page(page: fitz.Page, out_path: Path, target_width: int) -> None:
    width = max(float(page.rect.width), 1.0)
    scale = max(target_width / width, 0.1)
    matrix = fitz.Matrix(scale, scale)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pixmap.save(str(out_path))


def render_pdf_pages(
    path: Path,
    slides: list[SlidePage],
    pages_dir: Path,
    thumbnails_dir: Path,
) -> list[SlidePage]:
    file_dir = pages_dir.parent  # files/{file_id}/
    rendered: list[SlidePage] = []

    with fitz.open(path) as document:
        for slide in slides:
            page_index = slide.page_number - 1
            if page_index < 0 or page_index >= document.page_count:
                raise ValueError(f"Page {slide.page_number} is out of range for {path.name}")

            page = document.load_page(page_index)
            image_path = pages_dir / f"{slide.slide_id}.png"
            thumbnail_path = thumbnails_dir / f"{slide.slide_id}.png"

            _render_page(page, image_path, config.PAGE_IMAGE_WIDTH)
            _render_page(page, thumbnail_path, config.THUMBNAIL_WIDTH)

            rendered.append(
                slide.model_copy(
                    update={
                        "image_path": str(image_path.relative_to(file_dir)),
                        "thumbnail_path": str(thumbnail_path.relative_to(file_dir)),
                    }
                )
            )

    return rendered


def ensure_page_image(project_id: str, file_id: str, slide: SlidePage) -> Path | None:
    """Return a cached high-resolution page image, rendering it atomically if missing."""
    from slide_tutor.cache import get_project_file_dir

    file_dir = get_project_file_dir(project_id, file_id)
    relative_path = Path(slide.image_path) if slide.image_path else Path("pages") / f"{slide.slide_id}.png"
    output_path = file_dir / relative_path
    try:
        # Cached paths come from slides.json, but keep recovery inside the file directory.
        output_path.resolve().relative_to(file_dir.resolve())
    except ValueError:
        logger.warning("Refusing page image path outside project file directory: %s", output_path)
        return None
    if output_path.exists():
        return output_path

    source_path = file_dir / "source.pdf"
    if not source_path.exists() or slide.page_number < 1:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with fitz.open(source_path) as document:
            page_index = slide.page_number - 1
            if page_index >= document.page_count:
                return None
            page = document.load_page(page_index)
            with tempfile.NamedTemporaryFile(
                dir=output_path.parent,
                prefix=f".{slide.slide_id}-",
                suffix=".png",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
            _render_page(page, temp_path, config.PAGE_IMAGE_WIDTH)
            os.replace(temp_path, output_path)
            return output_path
    except Exception as exc:
        logger.warning("Could not restore page image for %s page %s: %s", source_path, slide.page_number, exc)
        return None
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)

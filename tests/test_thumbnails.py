from pathlib import Path

import fitz

from slide_tutor import cache
from slide_tutor.models import SlidePage
from slide_tutor.thumbnails import ensure_page_image


def _slide(page_number: int = 1) -> SlidePage:
    return SlidePage(
        slide_id=f"slide-{page_number}",
        module_id="project-1",
        project_id="project-1",
        file_id="file-1",
        source_file="lecture.pdf",
        source_type="pdf",
        page_number=page_number,
        global_index=page_number,
        text="page text",
        image_path=f"pages/slide-{page_number}.png",
        text_quality="ok",
    )


def _write_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page(width=300, height=200)
    page.insert_text((30, 60), "Lazy page image")
    document.save(path)
    document.close()


def test_missing_page_image_is_restored_from_source_pdf(isolated_data_root):
    file_dir = cache.get_project_file_dir("project-1", "file-1")
    _write_pdf(file_dir / "source.pdf")

    image_path = ensure_page_image("project-1", "file-1", _slide())

    assert image_path == file_dir / "pages" / "slide-1.png"
    assert image_path.exists()
    with fitz.open(image_path) as image:
        assert image.page_count == 1


def test_existing_page_image_is_not_rendered_again(isolated_data_root, monkeypatch):
    file_dir = cache.get_project_file_dir("project-1", "file-1")
    image_path = file_dir / "pages" / "slide-1.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"already cached")

    monkeypatch.setattr("slide_tutor.thumbnails._render_page", lambda *args: (_ for _ in ()).throw(AssertionError()))

    assert ensure_page_image("project-1", "file-1", _slide()) == image_path
    assert image_path.read_bytes() == b"already cached"


def test_invalid_page_or_missing_source_returns_none(isolated_data_root):
    assert ensure_page_image("project-1", "file-1", _slide(page_number=2)) is None

    file_dir = cache.get_project_file_dir("project-1", "file-1")
    _write_pdf(file_dir / "source.pdf")
    assert ensure_page_image("project-1", "file-1", _slide(page_number=2)) is None

"""Deterministic local project used by the retrieval gold evaluation.

The fixture is intentionally made from the normal project manifest/slides
artifacts, not from a test-only in-memory index.  It therefore exercises the
same scope, chunk, source-signature, and positional-page paths as a real
project while remaining independent of PDFs, Marker, LLMs, and the network.
"""

from __future__ import annotations

import hashlib
from slide_tutor import cache
from slide_tutor.models import FileSlidesPayload, ProjectFile, ProjectManifest, SlidePage
from slide_tutor.projects import save_project

FIXTURE_PROJECT_ID = "retrieval-gold-fixture"
FIXTURE_CREATED_AT = "2026-01-01T00:00:00Z"


def _file_sha(file_id: str) -> str:
    return hashlib.sha256(f"retrieval-gold:{file_id}".encode()).hexdigest()


def _slide(
    project_id: str,
    file_id: str,
    source_file: str,
    page_number: int,
    title: str,
    body: str,
) -> SlidePage:
    slide_id = f"slide-{file_id.removeprefix('file-')}-{page_number:03d}"
    return SlidePage(
        slide_id=slide_id,
        module_id=project_id,
        project_id=project_id,
        file_id=file_id,
        source_file=source_file,
        source_type="pdf",
        page_number=page_number,
        global_index=page_number,
        title_guess=title,
        text=f"{title}\n\n{body}",
        text_quality="ok",
    )


def _neuro_slides(project_id: str) -> list[SlidePage]:
    special: dict[int, tuple[str, str]] = {
        1: ("Introduction", "This neuroscience lecture introduces neurons, learning, and synaptic mechanisms."),
        4: ("BMI", "Body mass index (BMI) summarizes body size using weight and height."),
        11: ("Action potential", "An action potential carries an electrical signal along a neuron."),
        17: ("Synaptic signaling", "Neural signaling depends on synaptic transmission and receptor response."),
        29: ("Long-Term Memory", "Long-term memory stores information over extended periods."),
        30: (
            "Memory consolidation",
            "New memories become long-term and stable through memory consolidation.",
        ),
        48: (
            "Synaptic plasticity",
            "Synaptic plasticity is the capacity of connections to change with activity.",
        ),
        50: (
            "Long-Term Depression",
            "LTD (long-term depression) is a lasting weakening of synaptic strength caused by low-frequency stimulation.",
        ),
        51: (
            "Long-Term Depression",
            "LTD mechanisms reduce synaptic efficacy and complement long-term potentiation (LTP).",
        ),
        52: (
            "Long-Term Potentiation",
            "LTP (long-term potentiation) is a lasting strengthening of synaptic connections after high-frequency activity.",
        ),
    }
    slides: list[SlidePage] = []
    for page in range(1, 53):
        title, body = special.get(
            page,
            (
                f"Neuroscience topic {page}",
                "Neural circuits, learning, and synaptic mechanisms provide background context.",
            ),
        )
        slides.append(_slide(project_id, "file-neuro", "neuro.pdf", page, title, body))
    return slides


def _ml_slides(project_id: str) -> list[SlidePage]:
    special: dict[int, tuple[str, str]] = {
        1: ("Introduction", "This machine learning course introduces models, data, and evaluation."),
        3: ("Python", "Python uses dynamic typing and values receive types at runtime."),
        4: ("Java", "Java commonly uses static typing and compile-time type checks."),
        14: (
            "MPG",
            "MPG means miles per gallon, a fuel-efficiency measure for a vehicle.",
        ),
        15: (
            "Fuel efficiency applications",
            "MPG is useful for comparing fuel consumption, travel cost, and vehicle efficiency.",
        ),
        17: ("Data pipeline", "A data pipeline prepares examples before model training."),
        22: (
            "Gradient Descent",
            "Gradient descent updates parameters in the direction that reduces the loss function.",
        ),
        31: (
            "Attention mechanism",
            "Attention assigns weights to relevant tokens so a model can focus on important context.",
        ),
        40: (
            "Overfitting",
            "Overfitting occurs when a model performs well on training data but generalizes poorly.",
        ),
        41: (
            "Regularization",
            "Regularization constrains model complexity and limits over-reliance on training data.",
        ),
    }
    slides: list[SlidePage] = []
    for page in range(1, 42):
        title, body = special.get(
            page,
            (
                f"Machine learning topic {page}",
                "Models learn patterns from data and are evaluated on held-out examples.",
            ),
        )
        slides.append(_slide(project_id, "file-ml", "ml.pdf", page, title, body))
    return slides


def _write_file(
    project_id: str,
    file_id: str,
    name: str,
    slides: list[SlidePage],
) -> ProjectFile:
    file = ProjectFile(
        file_id=file_id,
        original_name=name,
        safe_name=name,
        sha256=_file_sha(file_id),
        size_bytes=0,
        page_count=len(slides),
        created_at=FIXTURE_CREATED_AT,
        status="ready",
    )
    file_dir = cache.get_project_file_dir(project_id, file_id)
    file_dir.mkdir(parents=True, exist_ok=True)
    cache.save_json(
        cache.project_file_slides_path(project_id, file_id),
        FileSlidesPayload(
            project_id=project_id,
            file_id=file_id,
            slides=slides,
        ),
    )
    return file


def build_retrieval_gold_project(project_id: str = FIXTURE_PROJECT_ID) -> ProjectManifest:
    """Build or refresh the deterministic gold project under the active data root."""
    cache.ensure_project_dirs(project_id)
    neuro = _neuro_slides(project_id)
    ml = _ml_slides(project_id)
    files = [
        _write_file(project_id, "file-neuro", "neuro.pdf", neuro),
        _write_file(project_id, "file-ml", "ml.pdf", ml),
    ]
    project = ProjectManifest(
        project_id=project_id,
        name="Retrieval Gold Fixture",
        created_at=FIXTURE_CREATED_AT,
        updated_at=FIXTURE_CREATED_AT,
        file_order=[file.file_id for file in files],
        files=files,
    )
    save_project(project)
    return project


__all__ = ["FIXTURE_PROJECT_ID", "build_retrieval_gold_project"]

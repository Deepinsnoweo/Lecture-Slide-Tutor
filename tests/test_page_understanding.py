from __future__ import annotations

import os
from pathlib import Path

import pytest

from slide_tutor import cache
from slide_tutor.marker_adapters.base import ParserPageResult
from slide_tutor.marker_adapters.v013 import V013MarkerAdapter
from slide_tutor.models import PageUnderstanding, SlidePage
from slide_tutor.page_understanding import (
    compute_quality_signals,
    evidence_eligibility,
    get_page_text_for_prompt,
    page_content_contract_signature,
    route_canonical_content,
)


def _slide(text: str, *, metadata=None) -> SlidePage:
    return SlidePage(
        slide_id="slide-1",
        module_id="p",
        project_id="p",
        file_id="f",
        source_file="lecture.pdf",
        source_type="pdf",
        page_number=1,
        global_index=1,
        title_guess="A title",
        text=text,
        text_quality="ok" if text else "empty",
        metadata=metadata or {},
    )


class _FakeMarkerAdapter:
    def __init__(self, results=None, *, available=True):
        self.results = list(results or [])
        self.available = available
        self.calls: list[list[int]] = []

    def is_available(self):
        return self.available

    def cache_identity(self):
        return {
            "parser_implementation": "fake-marker",
            "parser_version": "test",
            "adapter_implementation": "tests.fake-marker",
            "adapter_version": "v1",
        }

    def parse_pages(self, _pdf_path, page_numbers, *, cancel_check=None):
        if cancel_check is not None:
            cancel_check()
        self.calls.append(list(page_numbers))
        return list(self.results)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "project_id": "p", "file_id": "f", "slide_id": "s",
            "clean_markdown": "legacy body", "raw_text": "raw body",
            "parser_provider": "pymupdf", "quality": "fallback",
        },
        {
            "project_id": "p", "file_id": "f", "slide_id": "s",
            "canonical_markdown": "canonical body", "pymupdf_text": "raw body",
            "selected_provider": "pymupdf", "route_action": "select",
            "quality": "degraded",
        },
    ],
)
def test_new_and_legacy_page_understanding_round_trip(payload):
    page = PageUnderstanding.model_validate(payload)
    restored = PageUnderstanding.model_validate(page.model_dump(mode="json"))
    assert restored.canonical_markdown
    assert restored.pymupdf_text
    assert restored.quality_signals.model_dump()
    assert restored.route_action == "select"


def test_invalid_quality_and_route_are_rejected():
    with pytest.raises(ValueError):
        PageUnderstanding(
            project_id="p", file_id="f", slide_id="s", quality="invented",
        )
    with pytest.raises(ValueError):
        PageUnderstanding(
            project_id="p", file_id="f", slide_id="s", route_action="invented",
        )


def test_quality_signals_are_deterministic_and_measure_structure():
    pdf = "Title\nA complete sentence with a formula $x_i$ and a table.\n| a | b |"
    marker = "Title\n$$x_i$$\n| a | b |\n- item"
    first = compute_quality_signals(pdf, marker, title="Title")
    second = compute_quality_signals(pdf, marker, title="Title")
    assert first == second
    assert first.title_retained_by_marker is True
    assert first.marker_formula_count >= first.pymupdf_formula_count
    assert first.marker_table_line_count == 1
    assert first.marker_list_count == 1


def test_route_rejects_severely_truncated_marker_candidate():
    pdf = "This is a complete sentence. " * 8
    marker = "what"
    signals = compute_quality_signals(pdf, marker, title="A title")
    route, canonical, provider, warnings = route_canonical_content(
        signals, pdf, marker, title="A title",
    )
    assert route == "select"
    assert provider == "pymupdf"
    assert canonical == pdf.strip()
    assert warnings


@pytest.mark.parametrize(
    ("metadata", "expected_route"),
    [({}, "sparse_valid"), ({"visual_ratio": 0.8}, "needs_vlm")],
)
def test_sparse_pages_distinguish_valid_title_from_visual_page(metadata, expected_route):
    signals = compute_quality_signals("A title", "", title="A title", metadata=metadata)
    route, _canonical, _provider, _warnings = route_canonical_content(
        signals, "A title", "", title="A title",
    )
    assert route == expected_route


def test_formula_and_table_marker_candidate_is_preserved():
    pdf = "A formula is described in prose."
    marker = "The formula is $x_i = y_i$\n|x|y|\n|-|-|"
    signals = compute_quality_signals(pdf, marker)
    route, canonical, provider, _warnings = route_canonical_content(signals, pdf, marker)
    assert route == "select"
    assert provider == "marker"
    assert "$x_i" in canonical


def test_evidence_eligibility_boundaries():
    base = dict(project_id="p", file_id="f", slide_id="s", canonical_markdown="body")
    assert evidence_eligibility(PageUnderstanding(**base, quality="ok")) == "eligible"
    assert evidence_eligibility(PageUnderstanding(**base, quality="fallback")) == "degraded"
    assert evidence_eligibility(PageUnderstanding(**base, quality="needs_vlm")) == "needs_vlm_only"
    assert evidence_eligibility(PageUnderstanding(**base, quality="empty")) == "ineligible"


def test_parser_routes_truncated_marker_to_pymupdf_without_second_marker_call(monkeypatch):
    from slide_tutor import page_parsers

    adapter = _FakeMarkerAdapter([ParserPageResult(
        source_page_number=1,
        markdown="what",
        parser_family="marker",
        parser_version="marker-v",
        mode="test",
    )])
    page = page_parsers.parse_page(
        "p", "f", _slide("A complete sentence. " * 8),
        marker_adapter=adapter,
    )
    assert adapter.calls == [[1]]
    assert page.selected_provider == "pymupdf"
    assert page.canonical_markdown.startswith("A complete sentence")
    assert page.route_action == "select"
    assert page.quality in {"fallback", "degraded"}


def test_parser_handles_marker_missing_and_empty_text(monkeypatch):
    from slide_tutor import page_parsers

    adapter = _FakeMarkerAdapter(available=False)
    page = page_parsers.parse_page("p", "f", _slide(""), marker_adapter=adapter)
    assert page.route_action == "empty"
    assert page.quality == "empty"
    assert "No extractable text" in " ".join(page.warnings)


def test_parse_pages_uses_one_marker_batch_and_preserves_page_identity(monkeypatch):
    from slide_tutor import page_parsers

    slides = [
        _slide(f"PyMuPDF page {page_number}").model_copy(update={
            "slide_id": f"slide-{page_number}",
            "page_number": page_number,
            "global_index": page_number,
        })
        for page_number in range(1, 4)
    ]
    adapter = _FakeMarkerAdapter([
        ParserPageResult(
            source_page_number=page_number,
            markdown=f"Marker page {page_number} with complete structured content.",
            parser_family="marker",
            parser_version="0.1.3",
            mode="test",
        )
        for page_number in range(1, 4)
    ])
    pages = page_parsers.parse_pages("p", "f", slides, marker_adapter=adapter)

    assert adapter.calls == [[1, 2, 3]]
    assert [page.slide_id for page in pages] == ["slide-1", "slide-2", "slide-3"]
    assert [page.marker_markdown for page in pages] == [
        "Marker page 1 with complete structured content.",
        "Marker page 2 with complete structured content.",
        "Marker page 3 with complete structured content.",
    ]


def test_marker_batch_failure_bisects_and_keeps_successful_batches(monkeypatch):
    batch_calls = []
    single_calls = []
    adapter = V013MarkerAdapter()

    def extract_batch(_pdf_path, page_numbers):
        batch_calls.append(list(page_numbers))
        if 2 in page_numbers:
            raise RuntimeError("bad page")
        return [
            ParserPageResult(
                source_page_number=page_number,
                markdown=f"batch {page_number}",
                parser_family="marker",
                parser_version="0.1.3",
                mode="test",
            )
            for page_number in page_numbers
        ]

    def extract_single(_pdf_path, page_number):
        single_calls.append(page_number)
        return ParserPageResult(
            source_page_number=page_number,
            markdown=f"single {page_number}",
            parser_family="marker",
            parser_version="0.1.3",
            mode="test",
        )

    monkeypatch.setattr(adapter, "_extract_batch", extract_batch)
    monkeypatch.setattr(adapter, "_extract_single", extract_single)
    results = adapter._parse_resilient(
        Path("unused.pdf"), [1, 2, 3, 4], None,
    )
    results_by_page = {result.source_page_number: result for result in results}

    assert batch_calls == [[1, 2, 3, 4], [1, 2], [3, 4]]
    assert single_calls == [1, 2]
    assert results_by_page[1].markdown == "single 1"
    assert results_by_page[2].markdown == "single 2"
    assert results_by_page[3].markdown == "batch 3"
    assert results_by_page[4].markdown == "batch 4"


def test_marker_local_runtime_disables_network_retries(monkeypatch):
    from slide_tutor import config

    monkeypatch.setattr(config, "MARKER_LOCAL_FILES_ONLY", True)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    with V013MarkerAdapter().local_runtime():
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"

    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ


def test_page_content_contract_changes_with_marker_batch_size(monkeypatch):
    from slide_tutor import config

    original = page_content_contract_signature()
    monkeypatch.setattr(config, "MARKER_BATCH_PAGES", config.MARKER_BATCH_PAGES + 1)

    assert page_content_contract_signature() != original


def test_active_generation_missing_page_uses_ephemeral_parse_without_legacy_write(
    isolated_data_root, monkeypatch,
):
    slide = _slide("fallback slide text")
    cache.save_json(cache.get_project_dir("p") / "page_generation_meta.json", {
        "generation_id": "gen-active",
        "project_id": "p",
        "source_signature": "source",
        "page_content_contract_signature": page_content_contract_signature(),
        "page_count": 1,
        "page_manifest_signature": "manifest",
        "created_at": "2026-07-13T00:00:00Z",
        "storage_root": "generations/gen-active",
        "status": "current",
    })
    parsed = PageUnderstanding(
        project_id="p", file_id="f", slide_id=slide.slide_id,
        canonical_markdown="ephemeral canonical text",
        pymupdf_text="ephemeral canonical text",
        selected_provider="pymupdf", route_action="select", quality="ok",
    )
    from slide_tutor import page_parsers
    monkeypatch.setattr(page_parsers, "parse_page", lambda *_args: parsed)
    resolution = get_page_text_for_prompt("p", "f", slide)
    assert resolution.source_state == "ephemeral_degraded"
    assert "Refresh sources" in resolution.warning
    assert not cache.page_understanding_path("p", "f", slide.slide_id).exists()


def test_legacy_generation_metadata_never_reports_current_content(isolated_data_root):
    slide = _slide("fallback slide text")
    generation_root = cache.get_project_dir("p") / "generations" / "gen-old"
    cache.save_json(cache.get_project_dir("p") / "page_generation_meta.json", {
        "generation_id": "gen-old",
        "project_id": "p",
        "source_signature": "source",
        "page_content_contract_signature": page_content_contract_signature(),
        # Legacy metadata intentionally lacks page_count and manifest signature.
        "created_at": "2026-07-13T00:00:00Z",
        "storage_root": "generations/gen-old",
        "status": "current",
    })
    cached = PageUnderstanding(
        project_id="p", file_id="f", slide_id=slide.slide_id,
        generation_id="gen-old", canonical_markdown="legacy generation text",
        pymupdf_text="legacy generation text", selected_provider="pymupdf",
        route_action="select", quality="ok",
    )
    cache.save_json(generation_root / "f" / f"{slide.slide_id}.json", cached)

    resolution = get_page_text_for_prompt("p", "f", slide)

    assert resolution.source_state == "ephemeral_degraded"
    assert "stale" in resolution.warning

from __future__ import annotations

import json
from threading import Event

import pytest

from slide_tutor import cache, config, generation, index_store, page_parsers
from slide_tutor.marker_adapters.base import ParserPageResult
from slide_tutor.generation import (
    begin_enhancement,
    build_page_generation,
    generation_state,
    load_page_from_root,
    load_generation_lifecycle,
    mark_baseline_current,
    mark_enhancement_failed,
    page_batch_artifact_identity,
    page_batch_artifact_key,
    page_batch_artifact_path,
    page_batch_parser_identity,
    page_manifest_signature,
    validate_active_page_generation,
)
from slide_tutor.models import PageUnderstanding, RetrievalChunk, SlidePage
from slide_tutor.page_understanding import (
    active_page_generation,
    load_page_understanding,
    page_content_contract_signature,
)
from slide_tutor.projects import load_file_slides
from slide_tutor.retrieval_fixture import build_retrieval_gold_project


def test_generation_state_is_pure_and_detects_signature_mismatch(isolated_data_root):
    before = generation_state("missing-project")
    assert before.page_status == "missing"
    assert before.index_status == "missing"
    cache.save_json(cache.get_project_dir("p") / "page_generation_meta.json", {
        "generation_id": "gen-1",
        "source_signature": "source-1",
        "page_content_contract_signature": page_content_contract_signature(),
    })
    state = generation_state("p", "source-1")
    assert state.page_status == "stale"
    assert state.index_status == "missing"
    stale = generation_state("p", "source-2")
    assert stale.page_status == "stale"


def test_enhancement_failure_preserves_current_baseline_lifecycle(isolated_data_root):
    project_id = "lifecycle-project"
    cache.save_json(cache.get_project_dir(project_id) / "page_generation_meta.json", {
        "generation_id": "page-baseline",
        "source_signature": "source-1",
        "page_content_contract_signature": page_content_contract_signature(),
        "parser_identity": page_batch_parser_identity(),
        "page_count": 1,
        "page_manifest_signature": "manifest-1",
        "storage_root": "generations/page-baseline",
    })
    cache.save_json(cache.get_project_dir(project_id) / config.INDEX_DIRNAME / "index_meta.json", {
        "generation_id": "index-baseline",
        "source_signature": "source-1",
        "page_generation_id": "page-baseline",
    })

    baseline = mark_baseline_current(
        project_id,
        page_generation_id="page-baseline",
        index_generation_id="index-baseline",
        source_signature="source-1",
    )
    assert baseline.lifecycle_status == "baseline_current"
    assert baseline.baseline_status == "current"

    enhancing = begin_enhancement(project_id, "source-1")
    assert enhancing.lifecycle_status == "enhancing"
    assert enhancing.baseline_status == "current"
    assert enhancing.enhanced_status == "building"

    failed = mark_enhancement_failed(
        project_id, "test cancellation", cancelled=True,
    )
    assert failed.lifecycle_status == "enhancement_failed"
    assert failed.active_layer == "baseline"
    assert failed.baseline_status == "current"
    assert failed.enhanced_status == "failed"
    assert failed.error == "cancelled: test cancellation"

    reloaded = load_generation_lifecycle(project_id, "source-1")
    assert reloaded.lifecycle_status == "enhancement_failed"
    assert reloaded.baseline_page_generation_id == "page-baseline"
    assert reloaded.baseline_index_generation_id == "index-baseline"


def test_rebuild_activates_coherent_page_and_index_generation(
    isolated_data_root, monkeypatch,
):
    build_retrieval_gold_project()
    monkeypatch.setattr(index_store, "_load_embedding_model_with_timeout", lambda: None)
    monkeypatch.setattr(index_store, "_load_embedding_model", lambda: None)
    monkeypatch.setattr(config, "MARKER_ENABLED", False)

    events = []
    index_store.rebuild_index("retrieval-gold-fixture", progress_callback=events.append)

    page_meta = active_page_generation("retrieval-gold-fixture")
    index_meta = cache.load_json(
        index_store._index_meta_path("retrieval-gold-fixture"), default={}
    )
    assert page_meta["generation_id"]
    assert page_meta["storage_root"]
    assert index_meta["generation_id"] == page_meta["generation_id"]
    assert index_meta["page_generation_id"] == page_meta["generation_id"]
    assert generation_state(
        "retrieval-gold-fixture", index_meta["source_signature"]
    ).index_status == "current"
    lifecycle_state = generation_state(
        "retrieval-gold-fixture", index_meta["source_signature"],
    )
    assert lifecycle_state.lifecycle_status == "baseline_current"
    assert lifecycle_state.baseline_status == "current"
    assert lifecycle_state.active_layer == "baseline"
    slide = load_file_slides("retrieval-gold-fixture", "file-neuro")[49]
    page = load_page_understanding("retrieval-gold-fixture", "file-neuro", slide.slide_id)
    assert page is not None
    assert page.generation_id == page_meta["generation_id"]
    assert index_store.load_index("retrieval-gold-fixture") is not None
    page_events = [event for event in events if event.stage == "build page content"]
    assert page_events[0].status == "started"
    assert page_events[-1].status == "completed"
    assert page_events[-1].completed == page_events[-1].total == page_meta["page_count"]


def test_rebuild_without_callback_preserves_page_index_coherence(
    isolated_data_root, monkeypatch,
):
    """The compatibility rebuild path must retain the active page generation link."""
    build_retrieval_gold_project()
    monkeypatch.setattr(index_store, "_load_embedding_model_with_timeout", lambda: None)
    monkeypatch.setattr(index_store, "_load_embedding_model", lambda: None)
    monkeypatch.setattr(config, "MARKER_ENABLED", False)

    index_store.rebuild_index("retrieval-gold-fixture")

    page_meta = active_page_generation("retrieval-gold-fixture")
    index_meta = cache.load_json(
        index_store._index_meta_path("retrieval-gold-fixture"), default={}
    )
    assert index_meta["page_generation_id"] == page_meta["generation_id"]
    assert generation_state(
        "retrieval-gold-fixture", index_meta["source_signature"]
    ).index_status == "current"


def test_rebuild_builds_index_from_staged_page_root_canonical_content(
    isolated_data_root, monkeypatch,
):
    """The staged page root must resolve the same canonical text that is activated."""
    project_id = "staged-root-contract"
    stage_root = cache.get_project_dir(project_id) / ".page-stage-test"
    page_dir = stage_root / config.UNDERSTANDING_DIRNAME / "file-1"
    page_dir.mkdir(parents=True, exist_ok=True)
    cache.save_json(stage_root / "page_generation_meta.json", {
        "generation_id": "page-staged",
        "project_id": project_id,
        "source_signature": "source-v1",
        "page_content_contract_signature": page_content_contract_signature(),
        "parser_identity": page_batch_parser_identity(),
        "page_count": 1,
        "page_manifest_signature": "manifest-v1",
        "storage_root": "",
        "status": "current",
    })
    cache.save_json(page_dir / "slide-1.json", PageUnderstanding(
        project_id=project_id,
        file_id="file-1",
        slide_id="slide-1",
        canonical_markdown="canonical-body",
        clean_markdown="canonical-body",
        pymupdf_text="raw-body",
        raw_text="raw-body",
        parser_provider="marker",
        selected_provider="marker",
        quality="ok",
        generation_id="page-staged",
    ))

    captured_roots = []

    def fake_build_chunks(_project_id, *, understanding_root=None, **_kwargs):
        captured_roots.append(understanding_root)
        page = load_page_from_root(understanding_root, "file-1", "slide-1")
        assert page is not None
        return [RetrievalChunk(
            chunk_id="slide-1",
            project_id=project_id,
            file_id="file-1",
            slide_id="slide-1",
            source_file="lecture.pdf",
            page_number=1,
            text=page.canonical_markdown,
        )]

    monkeypatch.setattr(index_store, "compute_source_signature", lambda _pid: "source-v1")
    monkeypatch.setattr(
        index_store,
        "_maybe_stage_page_generation",
        lambda _pid, **_kwargs: (stage_root, "page-staged"),
    )
    monkeypatch.setattr(index_store, "build_chunks", fake_build_chunks)
    monkeypatch.setattr(index_store, "_load_embedding_model_with_timeout", lambda: None)
    monkeypatch.setattr(index_store, "_load_embedding_model", lambda: None)

    index_store.rebuild_index(project_id)

    assert captured_roots == [stage_root]
    chunks = index_store.load_index(project_id)[0]
    assert [chunk.text for chunk in chunks] == ["canonical-body"]


def test_generation_integrity_detects_missing_and_mismatched_pages(
    isolated_data_root, monkeypatch,
):
    build_retrieval_gold_project()
    monkeypatch.setattr(index_store, "_load_embedding_model_with_timeout", lambda: None)
    monkeypatch.setattr(index_store, "_load_embedding_model", lambda: None)
    monkeypatch.setattr(config, "MARKER_ENABLED", False)
    index_store.rebuild_index("retrieval-gold-fixture")
    project = __import__("slide_tutor.projects", fromlist=["load_project"]).load_project(
        "retrieval-gold-fixture"
    )
    files_and_slides = [
        (file.file_id, load_file_slides(project.project_id, file.file_id))
        for file in project.files if file.status == "ready"
    ]
    meta = active_page_generation(project.project_id)
    assert meta["page_manifest_signature"] == page_manifest_signature(files_and_slides)
    integrity = validate_active_page_generation(
        project.project_id, files_and_slides, source_signature=meta["source_signature"],
    )
    assert integrity.status == "current"
    first_file, first_slides = files_and_slides[0]
    page_path = (
        cache.get_project_dir(project.project_id) / meta["storage_root"] /
        first_file / f"{first_slides[0].slide_id}.json"
    )
    page_path.unlink()
    broken = validate_active_page_generation(
        project.project_id, files_and_slides, source_signature=meta["source_signature"],
    )
    assert broken.status == "corrupt"


def test_cancelled_page_generation_does_not_create_active_generation(
    isolated_data_root, monkeypatch,
):
    build_retrieval_gold_project()
    cancel = Event()
    cancel.set()
    with pytest.raises(index_store.IndexBuildCancelled):
        index_store.rebuild_index("retrieval-gold-fixture", cancel_signal=cancel)
    assert active_page_generation("retrieval-gold-fixture") == {}


def test_page_generation_uses_bounded_parser_batches(
    isolated_data_root, monkeypatch,
):
    slides = [
        SlidePage(
            slide_id=f"slide-{page_number}",
            module_id="p",
            project_id="p",
            file_id="f",
            source_file="lecture.pdf",
            source_type="pdf",
            page_number=page_number,
            global_index=page_number,
            title_guess=f"Page {page_number}",
            text=f"Body {page_number}",
            text_quality="ok",
        )
        for page_number in range(1, 6)
    ]
    calls = []

    def fake_parse_pages(
        project_id, file_id, slide_batch, *, cancel_check=None,
        marker_adapter=None,
    ):
        if cancel_check is not None:
            cancel_check()
        calls.append([slide.page_number for slide in slide_batch])
        return [
            PageUnderstanding(
                project_id=project_id,
                file_id=file_id,
                slide_id=slide.slide_id,
                canonical_markdown=slide.text,
                clean_markdown=slide.text,
                pymupdf_text=slide.text,
                raw_text=slide.text,
                selected_provider="pymupdf",
                parser_provider="pymupdf",
                quality="fallback",
            )
            for slide in slide_batch
        ]

    monkeypatch.setattr(config, "MARKER_BATCH_PAGES", 2)
    monkeypatch.setattr(page_parsers, "parse_pages", fake_parse_pages)
    events = []
    root, metadata = build_page_generation(
        "p", [("f", slides)], source_signature="source", progress_callback=events.append,
    )

    assert calls == [[1, 2], [3, 4], [5]]
    assert metadata.page_count == 5
    assert events[-1].status == "completed"
    assert events[-1].completed == events[-1].total == 5
    for slide in slides:
        page = cache.load_json(
            root / config.UNDERSTANDING_DIRNAME / "f" / f"{slide.slide_id}.json",
            default={},
        )
        assert page["generation_id"] == metadata.generation_id


def test_incomplete_reusable_batch_is_reparsed_as_one_contract_unit(
    isolated_data_root, monkeypatch,
):
    slides = [
        SlidePage(
            slide_id=f"slide-{page_number}",
            module_id="p",
            project_id="p",
            file_id="f",
            source_file="lecture.pdf",
            source_type="pdf",
            page_number=page_number,
            global_index=page_number,
            title_guess=f"Page {page_number}",
            text=f"Body {page_number}",
            text_quality="ok",
        )
        for page_number in (1, 2)
    ]
    reusable = PageUnderstanding(
        project_id="p",
        file_id="f",
        slide_id="slide-1",
        canonical_markdown="old page",
        clean_markdown="old page",
        pymupdf_text="old page",
        raw_text="old page",
        quality="fallback",
    )
    parsed_batches = []

    def fake_parse_pages(
        project_id, file_id, slide_batch, *, cancel_check=None,
        marker_adapter=None,
    ):
        parsed_batches.append([slide.slide_id for slide in slide_batch])
        return [
            PageUnderstanding(
                project_id=project_id,
                file_id=file_id,
                slide_id=slide.slide_id,
                canonical_markdown=f"new {slide.slide_id}",
                clean_markdown=f"new {slide.slide_id}",
                pymupdf_text=slide.text,
                raw_text=slide.text,
                quality="fallback",
            )
            for slide in slide_batch
        ]

    monkeypatch.setattr(config, "MARKER_BATCH_PAGES", 2)
    monkeypatch.setattr(
        "slide_tutor.generation.active_page_generation",
        lambda _project_id: {
            "source_signature": "source",
            "page_content_contract_signature": page_content_contract_signature(),
            "parser_identity": page_batch_parser_identity(),
        },
    )
    monkeypatch.setattr(
        "slide_tutor.page_understanding.load_page_understanding",
        lambda _project_id, _file_id, slide_id: reusable if slide_id == "slide-1" else None,
    )
    monkeypatch.setattr(page_parsers, "parse_pages", fake_parse_pages)

    root, _metadata = build_page_generation(
        "p", [("f", slides)], source_signature="source",
    )

    assert parsed_batches == [["slide-1", "slide-2"]]
    rebuilt = cache.load_json(
        root / config.UNDERSTANDING_DIRNAME / "f" / "slide-1.json", default={},
    )
    assert rebuilt["canonical_markdown"] == "new slide-1"


def _artifact_test_slides(file_id: str, count: int) -> list[SlidePage]:
    return [
        SlidePage(
            slide_id=f"{file_id}-slide-{page_number}",
            module_id="p",
            project_id="p",
            file_id=file_id,
            source_file=f"{file_id}.pdf",
            source_type="pdf",
            page_number=page_number,
            global_index=page_number,
            title_guess=f"Page {page_number}",
            text=f"Body {file_id} {page_number}",
            text_quality="ok",
        )
        for page_number in range(1, count + 1)
    ]


def _artifact_test_parser(monkeypatch):
    calls: list[list[str]] = []

    def fake_parse_pages(
        project_id, file_id, slide_batch, *, cancel_check=None,
        marker_adapter=None,
    ):
        if cancel_check is not None:
            cancel_check()
        calls.append([slide.slide_id for slide in slide_batch])
        return [
            PageUnderstanding(
                project_id=project_id,
                file_id=file_id,
                slide_id=slide.slide_id,
                canonical_markdown=slide.text,
                clean_markdown=slide.text,
                pymupdf_text=slide.text,
                raw_text=slide.text,
                selected_provider="pymupdf",
                parser_provider="pymupdf",
                quality="fallback",
            )
            for slide in slide_batch
        ]

    monkeypatch.setattr(config, "MARKER_BATCH_PAGES", 2)
    monkeypatch.setattr(page_parsers, "parse_pages", fake_parse_pages)
    return calls


class _InjectedMarkerAdapter:
    def __init__(self, *, parser_version: str, adapter_version: str):
        self.parser_version = parser_version
        self.adapter_version = adapter_version
        self.identity_calls = 0
        self.parse_calls: list[list[int]] = []

    def is_available(self):
        return True

    def cache_identity(self):
        self.identity_calls += 1
        return {
            "parser_implementation": "fake-marker",
            "parser_version": self.parser_version,
            "adapter_implementation": "tests.injected-marker-adapter",
            "adapter_version": self.adapter_version,
        }

    def parse_pages(self, _pdf_path, page_numbers, *, cancel_check=None):
        if cancel_check is not None:
            cancel_check()
        self.parse_calls.append(list(page_numbers))
        return [
            ParserPageResult(
                source_page_number=page_number,
                markdown=f"Structured Marker content for page {page_number}.",
                parser_family="marker",
                parser_version=self.parser_version,
                mode="test",
            )
            for page_number in page_numbers
        ]


def test_file_batch_artifacts_reuse_unchanged_files_when_a_file_is_added(
    isolated_data_root, monkeypatch,
):
    calls = _artifact_test_parser(monkeypatch)
    file_a = _artifact_test_slides("file-a", 2)
    file_b = _artifact_test_slides("file-b", 2)

    build_page_generation("p", [("file-a", file_a)], source_signature="source-a")
    _root, _metadata = build_page_generation(
        "p",
        [("file-a", file_a), ("file-b", file_b)],
        source_signature="source-a-plus-b",
    )

    assert calls == [
        [slide.slide_id for slide in file_a],
        [slide.slide_id for slide in file_b],
    ]
    artifacts = list(
        (cache.get_project_dir("p") / config.PAGE_ARTIFACTS_DIRNAME / config.PAGE_BATCH_ARTIFACTS_DIRNAME).glob("*.json")
    )
    assert len(artifacts) == 2


def test_replacing_one_file_rebuilds_only_replacement_batches(
    isolated_data_root, monkeypatch,
):
    calls = _artifact_test_parser(monkeypatch)
    unchanged = _artifact_test_slides("file-unchanged", 2)
    original = _artifact_test_slides("file-original", 2)
    replacement = _artifact_test_slides("file-replacement", 2)

    build_page_generation(
        "p",
        [("file-unchanged", unchanged), ("file-original", original)],
        source_signature="before-replacement",
    )
    calls.clear()
    build_page_generation(
        "p",
        [("file-unchanged", unchanged), ("file-replacement", replacement)],
        source_signature="after-replacement",
    )

    assert calls == [[slide.slide_id for slide in replacement]]


def test_corrupt_page_batch_artifact_rebuilds_only_that_batch(
    isolated_data_root, monkeypatch,
):
    calls = _artifact_test_parser(monkeypatch)
    slides = _artifact_test_slides("file-a", 4)
    build_page_generation("p", [("file-a", slides)], source_signature="source-a")
    first_batch = slides[:2]
    page_batch_artifact_path("p", "file-a", first_batch).write_text("{", encoding="utf-8")

    calls.clear()
    build_page_generation("p", [("file-a", slides)], source_signature="source-a")

    assert calls == [[slide.slide_id for slide in first_batch]]


def test_file_reorder_reuses_batches_and_content_key_ignores_runtime_facts(
    isolated_data_root, monkeypatch,
):
    calls = _artifact_test_parser(monkeypatch)
    file_a = _artifact_test_slides("file-a", 2)
    file_b = _artifact_test_slides("file-b", 2)
    build_page_generation(
        "p", [("file-a", file_a), ("file-b", file_b)], source_signature="same-files",
    )
    calls.clear()
    build_page_generation(
        "p", [("file-b", file_b), ("file-a", file_a)], source_signature="same-files",
    )
    assert calls == []

    identity = page_batch_artifact_identity("p", "file-a", file_a)
    key = page_batch_artifact_key(identity)
    assert "device" not in json.dumps(identity)
    assert "worker" not in json.dumps(identity)
    assert page_batch_artifact_key(identity) == key

    monkeypatch.setattr(config, "MARKER_ENABLED", not config.MARKER_ENABLED)
    assert page_batch_artifact_key(
        page_batch_artifact_identity("p", "file-a", file_a),
    ) != key


def test_page_batch_key_misses_when_parser_or_adapter_identity_changes(
    isolated_data_root,
):
    slides = _artifact_test_slides("file-a", 2)
    adapter_a = _InjectedMarkerAdapter(
        parser_version="0.1.3", adapter_version="adapter-v1",
    )
    identity = page_batch_artifact_identity(
        "p", "file-a", slides,
        parser_identity=page_batch_parser_identity(adapter_a),
    )
    key = page_batch_artifact_key(identity)

    parser_changed = _InjectedMarkerAdapter(
        parser_version="2.0.0", adapter_version="adapter-v1",
    )
    parser_changed_key = page_batch_artifact_key(
        page_batch_artifact_identity(
            "p", "file-a", slides,
            parser_identity=page_batch_parser_identity(parser_changed),
        ),
    )
    assert parser_changed_key != key

    adapter_changed = _InjectedMarkerAdapter(
        parser_version="0.1.3", adapter_version="adapter-v2",
    )
    adapter_changed_key = page_batch_artifact_key(
        page_batch_artifact_identity(
            "p", "file-a", slides,
            parser_identity=page_batch_parser_identity(adapter_changed),
        ),
    )
    assert adapter_changed_key != key


def _activate_artifact_test_generation(project_id, root, metadata) -> None:
    project_dir = cache.get_project_dir(project_id)
    active_root = project_dir / "generations" / metadata.generation_id
    active_root.parent.mkdir(parents=True, exist_ok=True)
    (root / config.UNDERSTANDING_DIRNAME).replace(active_root)
    active_metadata = metadata.model_copy(update={
        "storage_root": str(active_root.relative_to(project_dir)),
    })
    cache.save_json(project_dir / "page_generation_meta.json", active_metadata)


def test_parser_identity_change_reparses_instead_of_promoting_active_generation(
    isolated_data_root,
):
    slides = _artifact_test_slides("file-a", 2)
    adapter_a = _InjectedMarkerAdapter(
        parser_version="0.1.3", adapter_version="adapter-v1",
    )

    root, metadata = build_page_generation(
        "p", [("file-a", slides)], source_signature="same-source",
        marker_adapter=adapter_a,
    )
    _activate_artifact_test_generation("p", root, metadata)
    assert metadata.parser_identity["parser_version"] == "0.1.3"

    adapter_b = _InjectedMarkerAdapter(
        parser_version="2.0.0", adapter_version="adapter-v2",
    )

    _root, changed_metadata = build_page_generation(
        "p", [("file-a", slides)], source_signature="same-source",
        marker_adapter=adapter_b,
    )

    assert adapter_b.parse_calls == [[1, 2]]
    assert changed_metadata.parser_identity["parser_version"] == "2.0.0"
    assert changed_metadata.parser_identity["adapter_version"] == "adapter-v2"


def test_injected_adapter_identity_and_parser_are_bound_once_per_generation(
    isolated_data_root,
):
    slides = _artifact_test_slides("file-a", 2)
    adapter = _InjectedMarkerAdapter(
        parser_version="candidate", adapter_version="candidate-v1",
    )

    _root, metadata = build_page_generation(
        "p", [("file-a", slides)], source_signature="candidate-source",
        marker_adapter=adapter,
    )

    assert adapter.identity_calls == 1
    assert adapter.parse_calls == [[1, 2]]
    assert metadata.parser_identity["parser_version"] == "candidate"

    build_page_generation(
        "p", [("file-a", slides)], source_signature="candidate-source",
        marker_adapter=adapter,
    )
    assert adapter.identity_calls == 2
    assert adapter.parse_calls == [[1, 2]]


def test_injected_adapter_failure_cleans_staged_generation(
    isolated_data_root,
):
    slides = _artifact_test_slides("file-a", 2)
    adapter = _InjectedMarkerAdapter(
        parser_version="candidate", adapter_version="candidate-v1",
    )

    def fail_parse(*_args, **_kwargs):
        raise RuntimeError("candidate failed")

    adapter.parse_pages = fail_parse
    with pytest.raises(RuntimeError, match="candidate failed"):
        build_page_generation(
            "p", [("file-a", slides)], source_signature="candidate-source",
            marker_adapter=adapter,
        )

    assert active_page_generation("p") == {}
    assert list(cache.get_project_dir("p").glob(".page-build-*")) == []


def test_legacy_generation_without_parser_identity_is_stale(isolated_data_root):
    project_dir = cache.get_project_dir("legacy-parser-identity")
    cache.save_json(project_dir / "page_generation_meta.json", {
        "generation_id": "legacy-page-generation",
        "project_id": "legacy-parser-identity",
        "source_signature": "same-source",
        "page_content_contract_signature": page_content_contract_signature(),
        "page_count": 1,
        "page_manifest_signature": "legacy-manifest",
        "created_at": "2026-07-22T00:00:00Z",
        "storage_root": "generations/legacy-page-generation",
        "status": "current",
    })

    state = generation_state("legacy-parser-identity", "same-source")

    assert state.page_status == "stale"
    assert state.lifecycle_status == "stale"

from pathlib import Path
import tempfile
import numpy as np
import pytest
from types import SimpleNamespace
from threading import Event
import time

from slide_tutor import cache, config, index_store
from slide_tutor.models import RetrievalChunk, SlidePage


def _seed_generation_pair(project_id: str, monkeypatch) -> None:
    """Create a minimal coherent active page/index pair for lifecycle tests."""
    from slide_tutor.generation import mark_baseline_current, page_batch_parser_identity
    from slide_tutor.page_understanding import page_content_contract_signature

    project_dir = cache.get_project_dir(project_id)
    baseline_root = project_dir / "generations" / "page-baseline"
    (baseline_root / "file-1").mkdir(parents=True, exist_ok=True)
    cache.save_json(project_dir / "page_generation_meta.json", {
        "generation_id": "page-baseline",
        "source_signature": "source-v1",
        "page_content_contract_signature": page_content_contract_signature(),
        "parser_identity": page_batch_parser_identity(),
        "page_count": 1,
        "page_manifest_signature": "manifest-v1",
        "storage_root": "generations/page-baseline",
    })
    monkeypatch.setattr(index_store, "compute_source_signature", lambda _pid: "source-v1")
    index_store.save_index(
        project_id,
        ([_chunk(project_id)], object(), np.asarray([[0.1, 0.2]], dtype=np.float32)),
    )
    meta = cache.load_json(index_store._index_meta_path(project_id), default={})
    meta.update({
        "generation_id": "index-baseline",
        "page_generation_id": "page-baseline",
        "source_signature": "source-v1",
    })
    cache.save_json(index_store._index_meta_path(project_id), meta)
    mark_baseline_current(
        project_id,
        page_generation_id="page-baseline",
        index_generation_id="index-baseline",
        source_signature="source-v1",
    )


def _enhanced_page_stage(project_id: str) -> Path:
    from slide_tutor.generation import page_batch_parser_identity
    from slide_tutor.page_understanding import page_content_contract_signature

    project_dir = cache.get_project_dir(project_id)
    stage = Path(tempfile.mkdtemp(prefix=".enhanced-page-", dir=project_dir))
    understanding = stage / config.UNDERSTANDING_DIRNAME / "file-1"
    understanding.mkdir(parents=True, exist_ok=True)
    cache.save_json(stage / "page_generation_meta.json", {
        "generation_id": "page-enhanced",
        "project_id": project_id,
        "source_signature": "source-v1",
        "page_content_contract_signature": page_content_contract_signature(),
        "parser_identity": page_batch_parser_identity(),
        "page_count": 1,
        "page_manifest_signature": "manifest-enhanced",
        "storage_root": "",
        "status": "current",
    })
    cache.save_json(understanding / "slide-1.json", {
        "project_id": project_id,
        "file_id": "file-1",
        "slide_id": "slide-1",
        "canonical_markdown": "enhanced",
        "clean_markdown": "enhanced",
        "pymupdf_text": "enhanced",
        "raw_text": "enhanced",
        "quality": "fallback",
        "generation_id": "page-enhanced",
    })
    return stage


def _chunk(project_id: str = "project-1") -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id="slide-1",
        project_id=project_id,
        file_id="file-1",
        slide_id="slide-1",
        source_file="lecture.pdf",
        page_number=1,
        text="Long-Term Depression",
    )


def test_rebuild_index_preserves_saved_metadata(isolated_data_root, monkeypatch):
    """Pre-index should add indexed_at without erasing load_index metadata."""
    project_id = "project-1"

    monkeypatch.setattr(index_store, "build_chunks", lambda pid: ["chunk-a", "chunk-b"])
    monkeypatch.setattr(index_store, "build_index", lambda chunks: ("chunks", "backend", "matrix"))

    def fake_save_index(pid, index):
        cache.save_json(index_store._index_meta_path(pid), {
            "schema_version": "v1",
            "backend": "embedding",
            "embedding_model": "test-model",
            "chunk_count": 2,
            "created_at": "2026-07-09T00:00:00Z",
        })

    monkeypatch.setattr(index_store, "save_index", fake_save_index)

    index_store.rebuild_index(project_id)

    meta = cache.load_json(index_store._index_meta_path(project_id))
    assert meta["backend"] == "embedding"
    assert meta["embedding_model"] == "test-model"
    assert meta["chunk_count"] == 2
    assert meta["created_at"] == "2026-07-09T00:00:00Z"
    assert meta["indexed_at"]


def test_embedding_index_uses_shared_model_cache(isolated_data_root, monkeypatch):
    """Saving an index must not copy the SentenceTransformer into the project."""
    monkeypatch.setattr(index_store, "compute_source_signature", lambda _pid: "source-v1")

    index_store.save_index(
        "project-1",
        ([_chunk()], object(), np.asarray([[0.1, 0.2]], dtype=np.float32)),
    )

    index_dir = index_store._index_dir("project-1")
    meta = cache.load_json(index_store._index_meta_path("project-1"))
    assert not (index_dir / "embedding_model").exists()
    assert meta["embedding_model"] == config.EMBEDDING_MODEL
    assert meta["embedding_dimension"] == 2
    assert meta["index_config_signature"]


def test_embedding_reuse_encodes_only_new_chunk_keys(isolated_data_root, monkeypatch):
    monkeypatch.setattr(index_store, "compute_source_signature", lambda _pid: "source-v1")

    old_chunk = index_store._chunk_with_content_keys(_chunk())
    index_store.save_index(
        "project-1",
        ([old_chunk], object(), np.asarray([[0.1, 0.2]], dtype=np.float32)),
    )
    new_chunk = RetrievalChunk(
        chunk_id="slide-2",
        project_id="project-1",
        file_id="file-2",
        slide_id="slide-2",
        source_file="lecture-2.pdf",
        page_number=1,
        text="New content",
    )

    class FakeModel:
        def __init__(self):
            self.calls: list[list[str]] = []

        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, texts, **_kwargs):
            self.calls.append(list(texts))
            return np.asarray([[0.9, 0.8] for _text in texts], dtype=np.float32)

    model = FakeModel()
    chunks = [old_chunk, index_store._chunk_with_content_keys(new_chunk)]
    _backend, matrix = index_store._build_embedding_index_reusing(
        "project-1", chunks, model,
    )

    assert len(model.calls) == 1
    assert model.calls[0] == [f"{config.EMBEDDING_PASSAGE_PREFIX}New content"]
    assert np.allclose(matrix[0], [0.1, 0.2])
    assert np.allclose(matrix[1], [0.9, 0.8])
    assert index_store._last_embedding_reuse_stats == {
        "reused": 1,
        "encoded": 1,
        "total": 2,
        "source": "previous_index",
    }


def test_embedding_model_contract_change_invalidates_previous_rows(
    isolated_data_root, monkeypatch,
):
    monkeypatch.setattr(index_store, "compute_source_signature", lambda _pid: "source-v1")
    old_chunk = index_store._chunk_with_content_keys(_chunk())
    index_store.save_index(
        "project-1",
        ([old_chunk], object(), np.asarray([[0.1, 0.2]], dtype=np.float32)),
    )

    class FakeModel:
        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, texts, **_kwargs):
            return np.asarray([[0.7, 0.6] for _text in texts], dtype=np.float32)

    monkeypatch.setattr(config, "EMBEDDING_MODEL", "different-model")
    _backend, matrix = index_store._build_embedding_index_reusing(
        "project-1", [old_chunk], FakeModel(),
    )

    assert np.allclose(matrix[0], [0.7, 0.6])
    assert index_store._last_embedding_reuse_stats["reused"] == 0
    assert index_store._last_embedding_reuse_stats["encoded"] == 1


def test_enhanced_snapshot_switch_is_atomic_and_keeps_layer_coherence(
    isolated_data_root, monkeypatch,
):
    _seed_generation_pair("project-enhanced", monkeypatch)
    from slide_tutor.generation import load_generation_lifecycle

    stage = _enhanced_page_stage("project-enhanced")
    try:
        index_store.activate_enhanced_index(
            "project-enhanced",
            ([_chunk("project-enhanced")], object(), np.asarray([[0.9, 0.8]], dtype=np.float32)),
            page_stage_root=stage,
            page_generation_id="page-enhanced",
            source_signature="source-v1",
        )
    finally:
        if stage.exists():
            import shutil
            shutil.rmtree(stage, ignore_errors=True)

    lifecycle = load_generation_lifecycle("project-enhanced", "source-v1")
    assert lifecycle.lifecycle_status == "enhanced_current"
    assert lifecycle.active_layer == "enhanced"
    assert lifecycle.baseline_status == "current"
    assert lifecycle.enhanced_status == "current"
    page_meta = cache.load_json(cache.get_project_dir("project-enhanced") / "page_generation_meta.json")
    index_meta = cache.load_json(index_store._index_meta_path("project-enhanced"))
    assert page_meta["generation_id"] == "page-enhanced"
    assert index_meta["page_generation_id"] == "page-enhanced"


def test_direct_enhanced_activation_derives_omitted_source_signature(
    isolated_data_root, monkeypatch,
):
    """The public activation seam keeps its backwards-compatible source default."""
    project_id = "project-enhanced-derived-source"
    _seed_generation_pair(project_id, monkeypatch)
    from slide_tutor.generation import load_generation_lifecycle

    stage = _enhanced_page_stage(project_id)
    try:
        index_store.activate_enhanced_index(
            project_id,
            ([_chunk(project_id)], object(), np.asarray([[0.9, 0.8]], dtype=np.float32)),
            page_stage_root=stage,
            page_generation_id="page-enhanced",
        )
    finally:
        if stage.exists():
            import shutil
            shutil.rmtree(stage, ignore_errors=True)

    lifecycle = load_generation_lifecycle(project_id, "source-v1")
    assert lifecycle.lifecycle_status == "enhanced_current"
    assert lifecycle.active_layer == "enhanced"


def test_cancelled_enhancement_preserves_baseline_snapshot(
    isolated_data_root, monkeypatch,
):
    _seed_generation_pair("project-enhanced-cancel", monkeypatch)
    from slide_tutor.generation import load_generation_lifecycle

    old_chunks = cache.load_json(index_store._chunks_path("project-enhanced-cancel"))
    old_page_meta = cache.load_json(
        cache.get_project_dir("project-enhanced-cancel") / "page_generation_meta.json",
    )
    stage = _enhanced_page_stage("project-enhanced-cancel")
    cancel = Event()
    cancel.set()
    try:
        with pytest.raises(index_store.IndexBuildCancelled):
            index_store.activate_enhanced_index(
                "project-enhanced-cancel",
                ([_chunk("project-enhanced-cancel")], object(), np.asarray([[0.9, 0.8]], dtype=np.float32)),
                page_stage_root=stage,
                page_generation_id="page-enhanced",
                source_signature="source-v1",
                cancel_signal=cancel,
            )
    finally:
        if stage.exists():
            import shutil
            shutil.rmtree(stage, ignore_errors=True)

    assert cache.load_json(index_store._chunks_path("project-enhanced-cancel")) == old_chunks
    assert cache.load_json(
        cache.get_project_dir("project-enhanced-cancel") / "page_generation_meta.json",
    ) == old_page_meta
    lifecycle = load_generation_lifecycle("project-enhanced-cancel", "source-v1")
    assert lifecycle.lifecycle_status == "enhancement_failed"
    assert lifecycle.active_layer == "baseline"
    assert lifecycle.baseline_status == "current"
    assert lifecycle.enhanced_status == "failed"


def test_direct_enhanced_activation_rejects_cross_project_index(
    isolated_data_root, monkeypatch,
):
    """The public atomic activation seam enforces the same scope contract."""
    _seed_generation_pair("project-enhanced-direct-scope", monkeypatch)
    from slide_tutor.generation import load_generation_lifecycle

    old_chunks = cache.load_json(
        index_store._chunks_path("project-enhanced-direct-scope"),
    )
    stage = _enhanced_page_stage("project-enhanced-direct-scope")
    with pytest.raises(ValueError, match="another project"):
        index_store.activate_enhanced_index(
            "project-enhanced-direct-scope",
            ([_chunk("different-project")], object(), np.asarray([[0.9, 0.8]], dtype=np.float32)),
            page_stage_root=stage,
            page_generation_id="page-enhanced",
            source_signature="source-v1",
        )

    assert not stage.exists()
    assert cache.load_json(
        index_store._chunks_path("project-enhanced-direct-scope"),
    ) == old_chunks
    lifecycle = load_generation_lifecycle(
        "project-enhanced-direct-scope", "source-v1",
    )
    assert lifecycle.lifecycle_status == "baseline_current"
    assert lifecycle.active_layer == "baseline"


def test_direct_enhanced_activation_rejects_cross_project_page_stage(
    isolated_data_root, monkeypatch,
):
    """A page stage from another project cannot be moved into active storage."""
    _seed_generation_pair("project-enhanced-direct-page-scope", monkeypatch)
    from slide_tutor.generation import load_generation_lifecycle

    stage = _enhanced_page_stage("project-enhanced-direct-page-scope")
    page_meta = cache.load_json(stage / "page_generation_meta.json")
    page_meta["project_id"] = "different-project"
    cache.save_json(stage / "page_generation_meta.json", page_meta)
    with pytest.raises(ValueError, match="project/generation"):
        index_store.activate_enhanced_index(
            "project-enhanced-direct-page-scope",
            ([_chunk("project-enhanced-direct-page-scope")], object(), np.asarray([[0.9, 0.8]], dtype=np.float32)),
            page_stage_root=stage,
            page_generation_id="page-enhanced",
            source_signature="source-v1",
        )

    assert not stage.exists()
    lifecycle = load_generation_lifecycle(
        "project-enhanced-direct-page-scope", "source-v1",
    )
    assert lifecycle.lifecycle_status == "baseline_current"
    assert lifecycle.active_layer == "baseline"


def test_background_enhanced_job_activates_complete_snapshot(
    isolated_data_root, monkeypatch,
):
    """The opt-in worker owns the enhanced lifecycle and activates one pair."""
    _seed_generation_pair("project-enhanced-job", monkeypatch)
    from slide_tutor.generation import load_generation_lifecycle

    def build_snapshot(project_id, _progress_callback, _cancel_signal):
        return index_store.EnhancedSnapshot(
            index=(
                [_chunk(project_id)],
                object(),
                np.asarray([[0.9, 0.8]], dtype=np.float32),
            ),
            page_stage_root=_enhanced_page_stage(project_id),
            page_generation_id="page-enhanced",
            source_signature="source-v1",
        )

    job = index_store.start_enhanced_index_job(
        "project-enhanced-job", build_snapshot,
    )
    job.thread.join(timeout=2)
    assert not job.thread.is_alive()
    assert job.status == "completed"
    snapshot = index_store.get_index_job_snapshot("project-enhanced-job")
    assert snapshot is not None
    assert snapshot.operation == "enhanced_preindex"

    lifecycle = load_generation_lifecycle("project-enhanced-job", "source-v1")
    assert lifecycle.lifecycle_status == "enhanced_current"
    assert lifecycle.active_layer == "enhanced"
    assert lifecycle.baseline_status == "current"
    assert lifecycle.enhanced_status == "current"
    page_meta = cache.load_json(
        cache.get_project_dir("project-enhanced-job") / "page_generation_meta.json",
    )
    index_meta = cache.load_json(index_store._index_meta_path("project-enhanced-job"))
    assert page_meta["generation_id"] == "page-enhanced"
    assert index_meta["page_generation_id"] == "page-enhanced"
    index_store.mark_index_job_terminal_delivered("project-enhanced-job")
    index_store.clear_finished_index_job("project-enhanced-job")


def test_background_enhanced_job_failure_keeps_baseline_active(
    isolated_data_root, monkeypatch,
):
    """A builder error is visible and never replaces the baseline pair."""
    _seed_generation_pair("project-enhanced-job-failure", monkeypatch)
    from slide_tutor.generation import load_generation_lifecycle

    old_chunks = cache.load_json(
        index_store._chunks_path("project-enhanced-job-failure"),
    )
    old_page_meta = cache.load_json(
        cache.get_project_dir("project-enhanced-job-failure") / "page_generation_meta.json",
    )

    def fail_builder(_project_id, _progress_callback, _cancel_signal):
        raise RuntimeError("enhanced parser failed")

    job = index_store.start_enhanced_index_job(
        "project-enhanced-job-failure", fail_builder,
    )
    job.thread.join(timeout=2)
    assert not job.thread.is_alive()
    assert job.status == "failed"
    assert "enhanced parser failed" in (job.error or "")

    lifecycle = load_generation_lifecycle(
        "project-enhanced-job-failure", "source-v1",
    )
    assert lifecycle.lifecycle_status == "enhancement_failed"
    assert lifecycle.active_layer == "baseline"
    assert lifecycle.baseline_status == "current"
    assert lifecycle.enhanced_status == "failed"
    assert cache.load_json(
        index_store._chunks_path("project-enhanced-job-failure"),
    ) == old_chunks
    assert cache.load_json(
        cache.get_project_dir("project-enhanced-job-failure") / "page_generation_meta.json",
    ) == old_page_meta
    index_store.mark_index_job_terminal_delivered("project-enhanced-job-failure")
    index_store.clear_finished_index_job("project-enhanced-job-failure")


def test_background_enhanced_job_rejects_cross_project_snapshot(
    isolated_data_root, monkeypatch,
):
    """A complete-looking stage cannot activate chunks owned by another project."""
    _seed_generation_pair("project-enhanced-job-scope", monkeypatch)
    from slide_tutor.generation import load_generation_lifecycle

    old_chunks = cache.load_json(
        index_store._chunks_path("project-enhanced-job-scope"),
    )

    def cross_project_builder(project_id, _progress_callback, _cancel_signal):
        return index_store.EnhancedSnapshot(
            index=(
                [_chunk("different-project")],
                object(),
                np.asarray([[0.9, 0.8]], dtype=np.float32),
            ),
            page_stage_root=_enhanced_page_stage(project_id),
            page_generation_id="page-enhanced",
            source_signature="source-v1",
        )

    job = index_store.start_enhanced_index_job(
        "project-enhanced-job-scope", cross_project_builder,
    )
    job.thread.join(timeout=2)
    assert not job.thread.is_alive()
    assert job.status == "failed"
    assert "another project" in (job.error or "")
    lifecycle = load_generation_lifecycle(
        "project-enhanced-job-scope", "source-v1",
    )
    assert lifecycle.lifecycle_status == "enhancement_failed"
    assert lifecycle.active_layer == "baseline"
    assert lifecycle.baseline_status == "current"
    assert cache.load_json(
        index_store._chunks_path("project-enhanced-job-scope"),
    ) == old_chunks
    assert not list(
        cache.get_project_dir("project-enhanced-job-scope").glob(".enhanced-page-*")
    )
    index_store.mark_index_job_terminal_delivered("project-enhanced-job-scope")
    index_store.clear_finished_index_job("project-enhanced-job-scope")


def test_background_enhanced_job_cancel_keeps_baseline_active(
    isolated_data_root, monkeypatch,
):
    """Cancellation after a staged builder returns cannot activate its snapshot."""
    _seed_generation_pair("project-enhanced-job-cancel", monkeypatch)
    from slide_tutor.generation import load_generation_lifecycle

    old_chunks = cache.load_json(
        index_store._chunks_path("project-enhanced-job-cancel"),
    )
    old_page_meta = cache.load_json(
        cache.get_project_dir("project-enhanced-job-cancel") / "page_generation_meta.json",
    )
    builder_started = Event()
    release_builder = Event()

    def wait_builder(project_id, _progress_callback, _cancel_signal):
        builder_started.set()
        assert release_builder.wait(timeout=1)
        return index_store.EnhancedSnapshot(
            index=(
                [_chunk(project_id)],
                object(),
                np.asarray([[0.9, 0.8]], dtype=np.float32),
            ),
            page_stage_root=_enhanced_page_stage(project_id),
            page_generation_id="page-enhanced",
            source_signature="source-v1",
        )

    job = index_store.start_enhanced_index_job(
        "project-enhanced-job-cancel", wait_builder,
    )
    assert builder_started.wait(timeout=1)
    assert index_store.cancel_index_job("project-enhanced-job-cancel") is True
    release_builder.set()
    job.thread.join(timeout=2)
    assert not job.thread.is_alive()
    assert job.status == "cancelled"

    lifecycle = load_generation_lifecycle(
        "project-enhanced-job-cancel", "source-v1",
    )
    assert lifecycle.lifecycle_status == "enhancement_failed"
    assert lifecycle.active_layer == "baseline"
    assert lifecycle.baseline_status == "current"
    assert lifecycle.enhanced_status == "failed"
    assert lifecycle.error.startswith("cancelled:")
    assert cache.load_json(
        index_store._chunks_path("project-enhanced-job-cancel"),
    ) == old_chunks
    assert cache.load_json(
        cache.get_project_dir("project-enhanced-job-cancel") / "page_generation_meta.json",
    ) == old_page_meta
    assert not list(
        cache.get_project_dir("project-enhanced-job-cancel").glob(".enhanced-page-*")
    )
    index_store.mark_index_job_terminal_delivered("project-enhanced-job-cancel")
    index_store.clear_finished_index_job("project-enhanced-job-cancel")


def test_save_index_preserves_indexed_at_for_same_sources(isolated_data_root, monkeypatch):
    """An algorithm-only rebuild keeps the user's completed pre-index timestamp."""
    monkeypatch.setattr(index_store, "compute_source_signature", lambda _pid: "source-v1")
    cache.save_json(index_store._index_meta_path("project-1"), {
        "source_signature": "source-v1",
        "indexed_at": "2026-07-10T00:00:00Z",
    })

    index_store.save_index(
        "project-1",
        ([_chunk()], object(), np.asarray([[0.1, 0.2]], dtype=np.float32)),
    )

    meta = cache.load_json(index_store._index_meta_path("project-1"))
    assert meta["indexed_at"] == "2026-07-10T00:00:00Z"


def test_load_index_rejects_changed_retrieval_contract(isolated_data_root, monkeypatch):
    monkeypatch.setattr(index_store, "compute_source_signature", lambda _pid: "source-v1")
    monkeypatch.setattr(index_store, "_load_embedding_model", lambda: object())
    index_store.save_index(
        "project-1",
        ([_chunk()], object(), np.asarray([[0.1, 0.2]], dtype=np.float32)),
    )
    meta = cache.load_json(index_store._index_meta_path("project-1"))
    meta["chunk_text_version"] = "different"
    meta["index_config_signature"] = "stale"
    cache.save_json(index_store._index_meta_path("project-1"), meta)

    assert index_store.load_index("project-1") is None


def test_build_chunks_keeps_title_when_clean_markdown_drops_it(
    isolated_data_root, monkeypatch,
):
    """Marker-cleaned body must not erase a title present in source page metadata."""
    from slide_tutor import page_understanding, projects

    slide = SlidePage(
        slide_id="slide-50",
        module_id="project-1",
        project_id="project-1",
        file_id="file-1",
        source_file="neuro.pdf",
        source_type="pdf",
        page_number=50,
        global_index=50,
        title_guess="Long-Term Depression",
        text="Long-Term Depression raw PDF text",
        text_quality="ok",
    )
    payload = {"project_id": "project-1", "file_id": "file-1", "slides": [slide.model_dump()]}
    project = SimpleNamespace(files=[SimpleNamespace(file_id="file-1", status="ready")])
    understanding = SimpleNamespace(clean_markdown="This leads to LTD at input 1 synapses.")

    monkeypatch.setattr(projects, "load_project", lambda _pid: project)
    monkeypatch.setattr(index_store, "load_json", lambda _path: payload)
    monkeypatch.setattr(page_understanding, "load_page_understanding", lambda *args: understanding)

    chunks = index_store.build_chunks("project-1")

    assert chunks[0].title == "Long-Term Depression"
    assert chunks[0].normalized_title == "long-term depression"
    assert chunks[0].text.startswith("Long-Term Depression\n\nThis leads to LTD")


def test_rebuild_emits_observable_stages_and_uses_atomic_metadata(
    isolated_data_root, monkeypatch,
):
    events = []
    monkeypatch.setattr(index_store, "build_chunks", lambda _pid: [_chunk()])
    monkeypatch.setattr(
        index_store,
        "build_index",
        lambda chunks: (chunks, object(), np.asarray([[0.1, 0.2]], dtype=np.float32)),
    )
    monkeypatch.setattr(index_store, "compute_source_signature", lambda _pid: "source-v1")

    index_store.rebuild_index("project-1", progress_callback=events.append)

    assert [event.stage for event in events if event.status == "started"] == [
        "load shared model",
        "build chunks",
        "encode passages",
        "persist index",
        "activate index",
    ]
    assert events[-1].stage == "activate index"
    assert events[-1].status == "completed"
    assert events[-1].backend == "embedding"
    assert events[-1].dimension == 2
    meta = cache.load_json(index_store._index_meta_path("project-1"))
    assert meta["source_signature"] == "source-v1"
    assert meta["indexed_at"]
    assert meta["model_preflight"]["backend"] == "embedding"


def test_cancel_before_activation_preserves_previous_index(isolated_data_root, monkeypatch):
    monkeypatch.setattr(index_store, "compute_source_signature", lambda _pid: "source-v1")
    old = ([_chunk()], object(), np.asarray([[0.1, 0.2]], dtype=np.float32))
    index_store.save_index("project-1", old)
    old_chunks = cache.load_json(index_store._chunks_path("project-1"))

    monkeypatch.setattr(index_store, "build_chunks", lambda _pid: [_chunk()])
    monkeypatch.setattr(
        index_store,
        "build_index",
        lambda chunks: (chunks, object(), np.asarray([[0.9, 0.8]], dtype=np.float32)),
    )
    cancel = Event()

    def request_cancel(event):
        if event.stage == "persist index" and event.status == "completed":
            cancel.set()

    with pytest.raises(index_store.IndexBuildCancelled):
        index_store.rebuild_index(
            "project-1",
            progress_callback=request_cancel,
            cancel_signal=cancel,
        )

    assert cache.load_json(index_store._chunks_path("project-1")) == old_chunks
    assert not list(index_store._index_dir("project-1").parent.glob(".index-build-*"))


def test_activation_failure_rolls_back_complete_previous_generation(
    isolated_data_root, monkeypatch,
):
    monkeypatch.setattr(index_store, "compute_source_signature", lambda _pid: "source-v1")
    old = ([_chunk()], object(), np.asarray([[0.1, 0.2]], dtype=np.float32))
    index_store.save_index("project-1", old)
    old_chunks = cache.load_json(index_store._chunks_path("project-1"))
    original_replace = index_store.os.replace
    failed = False

    def fail_new_index(source, destination):
        nonlocal failed
        if (
            not failed
            and ".index-build-" in str(source)
            and Path(source).name == "index"
            and Path(destination) == index_store._index_dir("project-1")
        ):
            failed = True
            raise OSError("injected activation failure")
        return original_replace(source, destination)

    monkeypatch.setattr(index_store.os, "replace", fail_new_index)
    with pytest.raises(OSError, match="injected activation failure"):
        index_store.save_index(
            "project-1",
            ([_chunk()], object(), np.asarray([[0.9, 0.8]], dtype=np.float32)),
        )

    assert cache.load_json(index_store._chunks_path("project-1")) == old_chunks
    assert cache.load_json(index_store._index_meta_path("project-1"))["chunk_count"] == 1


def test_progress_terminal_events_are_stage_local(isolated_data_root, monkeypatch):
    events = []
    monkeypatch.setattr(index_store, "build_chunks", lambda _pid: [_chunk()])
    monkeypatch.setattr(
        index_store,
        "build_index",
        lambda chunks, **_kwargs: (chunks, object(), np.asarray([[0.1, 0.2]], dtype=np.float32)),
    )
    # Keep the injected builder out of the real model preflight branch.
    monkeypatch.setattr(index_store, "_DEFAULT_BUILD_INDEX", object())
    index_store._build_index_with_progress("project-1", progress_callback=events.append)

    for stage in {event.stage for event in events}:
        terminal = [
            event.status for event in events
            if event.stage == stage and event.status in {"completed", "failed", "cancelled"}
        ]
        assert len(terminal) == 1
        assert terminal[0] == "completed"


def test_cancel_in_encode_stage_does_not_cancel_activate(isolated_data_root, monkeypatch):
    events = []
    cancel = Event()
    monkeypatch.setattr(index_store, "build_chunks", lambda _pid: [_chunk()])

    def fake_build(chunks, progress_callback=None, cancel_signal=None):
        if progress_callback:
            progress_callback(index_store.IndexProgress(
                stage="encode passages", status="progress", completed=1, total=1,
            ))
        cancel.set()
        index_store._check_cancelled(cancel_signal)
        return chunks, object(), np.asarray([[0.1, 0.2]], dtype=np.float32)

    monkeypatch.setattr(index_store, "build_index", fake_build)
    with pytest.raises(index_store.IndexBuildCancelled):
        index_store._build_index_with_progress(
            "project-1", progress_callback=events.append, cancel_signal=cancel,
        )
    terminal = [event for event in events if event.status in {"cancelled", "failed"}]
    assert [(event.stage, event.status) for event in terminal] == [
        ("encode passages", "cancelled"),
    ]


def test_cancel_in_build_chunks_stage_is_not_reported_as_activate(
    isolated_data_root, monkeypatch,
):
    events = []
    cancel = Event()

    def fake_chunks(_project_id, progress_callback=None, cancel_signal=None):
        if progress_callback:
            progress_callback(index_store.IndexProgress(
                stage="build chunks", status="progress", completed=1, total=2,
            ))
        cancel.set()
        index_store._check_cancelled(cancel_signal)
        return [_chunk()]

    monkeypatch.setattr(index_store, "build_chunks", fake_chunks)
    with pytest.raises(index_store.IndexBuildCancelled):
        index_store._build_index_with_progress(
            "project-1", progress_callback=events.append, cancel_signal=cancel,
        )
    terminal = [event for event in events if event.status in {"cancelled", "failed"}]
    assert [(event.stage, event.status) for event in terminal] == [
        ("build chunks", "cancelled"),
    ]


def test_cancel_in_persist_stage_discards_staging_and_preserves_old_index(
    isolated_data_root, monkeypatch,
):
    monkeypatch.setattr(index_store, "compute_source_signature", lambda _pid: "source-v1")
    old = ([_chunk()], object(), np.asarray([[0.1, 0.2]], dtype=np.float32))
    index_store.save_index("project-1", old)
    old_chunks = cache.load_json(index_store._chunks_path("project-1"))
    events = []
    cancel = Event()

    def request_cancel(event):
        if event.stage == "persist index" and event.status == "started":
            cancel.set()

    with pytest.raises(index_store.IndexBuildCancelled):
        index_store._persist_index_atomic(
            "project-1",
            ([_chunk()], object(), np.asarray([[0.9, 0.8]], dtype=np.float32)),
            progress_callback=lambda event: (events.append(event), request_cancel(event)),
            cancel_signal=cancel,
        )
    assert [(event.stage, event.status) for event in events if event.status in {"cancelled", "failed"}] == [
        ("persist index", "cancelled"),
    ]
    assert cache.load_json(index_store._chunks_path("project-1")) == old_chunks
    assert not list(index_store._index_dir("project-1").parent.glob(".index-build-*"))


def test_background_worker_failure_is_visible_to_ui_state(isolated_data_root, monkeypatch):
    def fail_rebuild(_project_id, **_kwargs):
        raise ValueError("injected worker failure")

    monkeypatch.setattr(index_store, "rebuild_index", fail_rebuild)
    job = index_store.start_index_job("project-failure")
    job.thread.join(timeout=1)
    assert job.status == "failed"
    assert "injected worker failure" in job.error
    index_store.clear_finished_index_job("project-failure")


def test_embedding_dimension_mismatch_uses_production_build_path(
    isolated_data_root, monkeypatch,
):
    events = []
    cache.save_json(index_store._index_meta_path("project-1"), {
        "backend": "embedding",
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_model_revision": config.EMBEDDING_MODEL_REVISION,
        "embedding_dimension": 3,
    })
    monkeypatch.setattr(index_store, "build_chunks", lambda _pid: [_chunk()])
    fake_build = lambda chunks, **_kwargs: (
        chunks, object(), np.asarray([[0.1, 0.2]], dtype=np.float32),
    )
    monkeypatch.setattr(index_store, "build_index", fake_build)
    monkeypatch.setattr(index_store, "_DEFAULT_BUILD_INDEX", fake_build)

    def preflight(expected_dimension=None):
        assert expected_dimension == 3
        return index_store.ModelPreflight(
            available=True,
            model_id=config.EMBEDDING_MODEL,
            revision=config.EMBEDDING_MODEL_REVISION,
            dimension=3,
            reason="ready",
            backend="embedding",
        )

    monkeypatch.setattr(index_store, "preflight_shared_model", preflight)
    with pytest.raises(index_store.EmbeddingDimensionMismatch, match="expected 3"):
        index_store._build_index_with_progress(
            "project-1", progress_callback=events.append,
        )
    assert [(event.stage, event.status) for event in events if event.status == "failed"] == [
        ("encode passages", "failed"),
    ]


def test_project_jobs_are_isolated_and_second_start_is_rejected(
    isolated_data_root, monkeypatch,
):
    started = {}

    def fake_rebuild(project_id, progress_callback=None, cancel_signal=None):
        started[project_id] = True
        if progress_callback:
            progress_callback(index_store.IndexProgress(
                stage="load shared model", status="started",
            ))
        while not cancel_signal.is_set():
            time.sleep(0.005)
        raise index_store.IndexBuildCancelled("test cancellation")

    monkeypatch.setattr(index_store, "rebuild_index", fake_rebuild)
    first = index_store.start_index_job("project-a")
    second = index_store.start_index_job("project-b")
    assert index_store.drain_index_job_progress("project-a")[0].stage == "queued"
    assert index_store.drain_index_job_progress("project-b")[0].stage == "queued"
    with pytest.raises(RuntimeError, match="project-a"):
        index_store.start_index_job("project-a")

    assert index_store.cancel_index_job("project-a") is True
    first.thread.join(timeout=1)
    assert first.status == "cancelled"
    deadline = time.monotonic() + 1
    while second.status == "queued" and time.monotonic() < deadline:
        time.sleep(0.005)
    assert second.status == "running"
    assert index_store.cancel_index_job("project-b") is True
    second.thread.join(timeout=1)
    assert second.status == "cancelled"
    assert started == {"project-a": True, "project-b": True}
    index_store.clear_finished_index_job("project-a")
    index_store.clear_finished_index_job("project-b")


def test_project_jobs_run_through_one_global_serial_queue(
    isolated_data_root, monkeypatch,
):
    release_first = Event()
    first_started = Event()
    second_started = Event()
    running = 0
    max_running = 0

    def fake_rebuild(project_id, **_kwargs):
        nonlocal running, max_running
        running += 1
        max_running = max(max_running, running)
        if project_id == "project-a":
            first_started.set()
            assert release_first.wait(timeout=1)
        else:
            second_started.set()
        running -= 1
        return ([], object(), np.empty((0, 0)))

    monkeypatch.setattr(index_store, "rebuild_index", fake_rebuild)
    first = index_store.start_index_job("project-a")
    assert first_started.wait(timeout=1)
    second = index_store.start_index_job("project-b")
    time.sleep(0.05)

    assert index_store.get_index_job_snapshot("project-b").status == "queued"
    assert not second_started.is_set()
    release_first.set()
    first.thread.join(timeout=1)
    second.thread.join(timeout=1)

    assert second_started.is_set()
    assert max_running == 1
    assert index_store.mark_index_job_terminal_delivered("project-a") is True
    assert index_store.mark_index_job_terminal_delivered("project-a") is False
    assert index_store.mark_index_job_terminal_delivered("project-b") is True
    index_store.clear_finished_index_job("project-a")
    index_store.clear_finished_index_job("project-b")


def test_queued_index_job_can_be_cancelled_before_worker_starts(
    isolated_data_root, monkeypatch,
):
    release_first = Event()
    called = []

    def fake_rebuild(project_id, **_kwargs):
        called.append(project_id)
        if project_id == "project-a":
            assert release_first.wait(timeout=1)
        return ([], object(), np.empty((0, 0)))

    monkeypatch.setattr(index_store, "rebuild_index", fake_rebuild)
    first = index_store.start_index_job("project-a")
    second = index_store.start_index_job("project-b")
    assert index_store.cancel_index_job("project-b") is True
    second.thread.join(timeout=1)
    release_first.set()
    first.thread.join(timeout=1)

    assert second.status == "cancelled"
    assert called == ["project-a"]
    for project_id in ("project-a", "project-b"):
        index_store.mark_index_job_terminal_delivered(project_id)
        index_store.clear_finished_index_job(project_id)


def test_ensure_index_via_job_detaches_waiter_without_cancelling_shared_job(
    isolated_data_root, monkeypatch,
):
    build_started = Event()
    release_build = Event()
    waiter_cancel = Event()

    monkeypatch.setattr(index_store, "load_index", lambda _pid: None)

    def fake_ensure(_project_id, **_kwargs):
        build_started.set()
        assert release_build.wait(timeout=1)
        return ([_chunk()], object(), np.asarray([[0.1, 0.2]]))

    monkeypatch.setattr(index_store, "ensure_index", fake_ensure)
    errors = []

    def wait_for_index():
        try:
            index_store.ensure_index_via_job("project-lazy", waiter_cancel)
        except Exception as exc:
            errors.append(exc)

    from threading import Thread
    waiter = Thread(target=wait_for_index)
    waiter.start()
    assert build_started.wait(timeout=1)
    waiter_cancel.set()
    waiter.join(timeout=1)

    job = index_store.get_index_job("project-lazy")
    assert isinstance(errors[0], index_store.IndexBuildCancelled)
    assert job.cancel_event.is_set() is False
    release_build.set()
    job.thread.join(timeout=1)
    assert job.status == "completed"
    index_store.mark_index_job_terminal_delivered("project-lazy")
    index_store.clear_finished_index_job("project-lazy")

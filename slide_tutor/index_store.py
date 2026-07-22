"""Vector index for slide retrieval — dual-backend: sentence-transformers
embeddings (preferred, semantic matching) or SKLearn TfidfVectorizer (fallback,
keyword matching).  Chunk text now comes from PageUnderstanding.clean_markdown
when available, with SlidePage.text as fallback for old projects."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import pickle
import re
import shutil
import tempfile
import time
import unicodedata
from datetime import UTC, datetime
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from queue import Empty, Queue
from threading import Condition, Event, Lock, RLock, Thread
from typing import Any, Callable, Literal, Union
from uuid import uuid4

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from slide_tutor import config
from slide_tutor.cache import get_project_dir, load_json, save_json
from slide_tutor.models import (
    PAGE_CONTENT_CONTRACT_VERSION,
    FileSlidesPayload,
    IndexProgress,
    PreindexStatus,
    RetrievalChunk,
)

logger = logging.getLogger(__name__)

# Index type: (chunks, backend, matrix)
#   TF-IDF mode:  backend = TfidfVectorizer, matrix = csr_matrix
#   Embedding mode: backend = SentenceTransformer, matrix = np.ndarray
Index = tuple[list[RetrievalChunk], Any, Union[csr_matrix, np.ndarray]]
_embedding_model_cache: tuple[tuple[str, str | None], Any | None] | None = None
_last_embedding_error: str | None = None
_last_embedding_reuse_stats: dict[str, Any] = {
    "reused": 0,
    "encoded": 0,
    "total": 0,
    "source": "none",
}
_project_locks: dict[str, RLock] = {}
_project_locks_guard = Lock()


ProgressCallback = Callable[[IndexProgress], None]
CancellationSignal = Event | Callable[[], bool] | None


class IndexBuildCancelled(RuntimeError):
    """Raised when a caller cancels a rebuild before activation."""


class EmbeddingDimensionMismatch(RuntimeError):
    """Raised when a shared model cannot satisfy an existing index contract."""


class EnhancedRouteUnavailable(RuntimeError):
    """Raised when no explicitly registered enhanced builder is available."""


@dataclass(frozen=True)
class ModelPreflight:
    """Result of checking whether the configured shared embedding model is usable."""

    available: bool
    model_id: str
    revision: str | None
    dimension: int | None
    reason: str
    backend: str


@dataclass(frozen=True)
class EnhancedSnapshot:
    """Complete enhanced page/index input for one atomic activation.

    The production v0.1.3 route intentionally does not choose an enhanced
    parser.  An explicitly authorized adapter may provide this snapshot to
    the background runner; the runner owns cancellation, lifecycle state and
    whole-snapshot activation.
    """

    index: Index
    page_stage_root: Path
    page_generation_id: str
    source_signature: str
    indexed_at: str | None = None


EnhancedSnapshotBuilder = Callable[
    [str, ProgressCallback, CancellationSignal], EnhancedSnapshot,
]


def _validate_enhanced_index_snapshot(
    project_id: str,
    index: Index,
    page_generation_id: str,
) -> None:
    """Reject an enhanced index that cannot belong to the active project/page.

    The enhanced builder is an integration seam, so validation must happen
    before any staged index is persisted.  Production projects have a manifest
    and therefore get file-id validation; compatibility fixtures without a
    manifest still receive project-id, generation-id and matrix-shape checks.
    """
    if not isinstance(index, tuple) or len(index) != 3:
        raise TypeError("enhanced snapshot index must be a (chunks, backend, matrix) tuple")
    chunks, _backend, matrix = index
    if not isinstance(chunks, list) or not all(
        isinstance(chunk, RetrievalChunk) for chunk in chunks
    ):
        raise TypeError("enhanced snapshot chunks must be RetrievalChunk models")
    shape = getattr(matrix, "shape", None)
    if shape is None or len(shape) < 1 or int(shape[0]) != len(chunks):
        raise ValueError(
            "enhanced snapshot matrix row count must match its chunk count"
        )

    expected_file_ids: set[str] | None = None
    try:
        from slide_tutor.projects import load_project

        expected_file_ids = {
            file.file_id for file in load_project(project_id).files
        }
    except FileNotFoundError:
        # Unit/compatibility fixtures may intentionally omit a manifest.
        expected_file_ids = None

    for chunk in chunks:
        if chunk.project_id != project_id:
            raise ValueError(
                "enhanced snapshot contains a chunk from another project"
            )
        if expected_file_ids is not None and chunk.file_id not in expected_file_ids:
            raise ValueError(
                f"enhanced snapshot references file outside project: {chunk.file_id}"
            )
        if chunk.generation_id and chunk.generation_id != page_generation_id:
            raise ValueError(
                "enhanced snapshot chunk generation does not match its page stage"
            )


def _validate_enhanced_page_stage(
    project_id: str,
    page_stage_root: Path,
    page_generation_id: str,
    source_signature: str,
) -> None:
    """Validate the staged page metadata before it can become active."""
    if not isinstance(page_stage_root, Path) or not page_stage_root.is_dir():
        raise ValueError("enhanced snapshot must include a complete page stage")
    page_meta = load_json(
        page_stage_root / "page_generation_meta.json",
        default=None,
    )
    if (
        not isinstance(page_meta, dict)
        or page_meta.get("project_id") not in {None, project_id}
        or page_meta.get("generation_id") != page_generation_id
        or page_meta.get("source_signature") != source_signature
        or not (page_stage_root / config.UNDERSTANDING_DIRNAME).is_dir()
    ):
        raise ValueError(
            "enhanced snapshot page metadata does not match its declared project/generation"
        )


@dataclass
class IndexJob:
    """Mutable worker state; consumers should read an IndexJobSnapshot."""

    project_id: str
    operation: Literal["explicit_preindex", "lazy_rebuild", "enhanced_preindex"]
    cancel_event: Event
    progress_queue: Queue[IndexProgress]
    queued_at: float
    enhanced_builder: EnhancedSnapshotBuilder | None = None
    thread: Thread | None = None
    status: Literal["queued", "running", "completed", "cancelled", "failed"] = "queued"
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    result: Index | None = None
    last_progress: IndexProgress | None = None
    terminal_delivered: bool = False
    lock: RLock = dataclass_field(default_factory=RLock)


@dataclass(frozen=True)
class IndexJobSnapshot:
    """Immutable cross-thread view of an index job."""

    project_id: str
    operation: Literal["explicit_preindex", "lazy_rebuild", "enhanced_preindex"]
    status: Literal["queued", "running", "completed", "cancelled", "failed"]
    queued_at: float
    started_at: float | None
    completed_at: float | None
    error: str | None
    last_progress: IndexProgress | None
    cancel_requested: bool
    terminal_delivered: bool


_index_jobs: dict[str, IndexJob] = {}
_index_jobs_lock = Lock()
_index_queue: list[IndexJob] = []
_index_queue_condition = Condition(Lock())


def _project_index_lock(project_id: str) -> RLock:
    """Return the process-local lock for one project's load/activation window."""
    with _project_locks_guard:
        return _project_locks.setdefault(project_id, RLock())


def _index_dir(project_id: str) -> Path:
    return get_project_dir(project_id) / config.INDEX_DIRNAME


def _chunks_path(project_id: str) -> Path:
    return get_project_dir(project_id) / config.CHUNKS_FILENAME


def _vec_pkl_path(project_id: str) -> Path:
    return _index_dir(project_id) / "vectorizer.pkl"


def _tfidf_matrix_path(project_id: str) -> Path:
    return _index_dir(project_id) / "tfidf_matrix.npz"


def _embeddings_path(project_id: str) -> Path:
    return _index_dir(project_id) / "embeddings.npy"


def _backend_flag_path(project_id: str) -> Path:
    return _index_dir(project_id) / "backend.txt"


def _index_meta_path(project_id: str) -> Path:
    return _index_dir(project_id) / "index_meta.json"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _cancel_requested(cancel_signal: CancellationSignal) -> bool:
    if cancel_signal is None:
        return False
    if isinstance(cancel_signal, Event):
        return cancel_signal.is_set()
    return bool(cancel_signal())


def _check_cancelled(cancel_signal: CancellationSignal) -> None:
    if _cancel_requested(cancel_signal):
        raise IndexBuildCancelled("Index rebuild cancelled before activation")


def _emit_progress(
    callback: ProgressCallback | None,
    operation_started: float,
    stage: str,
    status: Literal["started", "progress", "completed", "failed", "cancelled"],
    *,
    completed: int = 0,
    total: int = 0,
    reused: int = 0,
    backend: str | None = None,
    dimension: int | None = None,
    fallback_reason: str = "",
    detail: str = "",
) -> None:
    if callback is None:
        return
    callback(IndexProgress(
        stage=stage,
        status=status,
        completed=completed,
        total=total,
        reused=reused,
        elapsed_seconds=time.perf_counter() - operation_started,
        backend=backend,
        dimension=dimension,
        fallback_reason=fallback_reason,
        detail=detail,
    ))


def _queue_progress(job: IndexJob, event: IndexProgress) -> None:
    job.progress_queue.put(event)
    with job.lock:
        job.last_progress = event


def get_index_job(project_id: str) -> IndexJob | None:
    """Return the current in-process job for a project, if any."""
    with _index_jobs_lock:
        return _index_jobs.get(project_id)


def get_index_job_snapshot(project_id: str) -> IndexJobSnapshot | None:
    """Return an immutable snapshot suitable for UI reruns."""
    job = get_index_job(project_id)
    if job is None:
        return None
    with job.lock:
        return IndexJobSnapshot(
            project_id=job.project_id,
            operation=job.operation,
            status=job.status,
            queued_at=job.queued_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error=job.error,
            last_progress=job.last_progress,
            cancel_requested=job.cancel_event.is_set(),
            terminal_delivered=job.terminal_delivered,
        )


def drain_index_job_progress(project_id: str) -> list[IndexProgress]:
    """Drain queued progress events without touching Streamlit from the worker."""
    job = get_index_job(project_id)
    if job is None:
        return []
    events: list[IndexProgress] = []
    while True:
        try:
            event = job.progress_queue.get_nowait()
        except Empty:
            break
        with job.lock:
            job.last_progress = event
        events.append(event)
    return events


def cancel_index_job(project_id: str) -> bool:
    """Request cancellation for only the selected project's running job."""
    job = get_index_job(project_id)
    if job is None:
        return False
    with job.lock:
        if job.status not in {"queued", "running"}:
            return False
        job.cancel_event.set()
    with _index_queue_condition:
        _index_queue_condition.notify_all()
    return True


def _finish_index_job(job: IndexJob, status: str, *, error: str | None = None) -> None:
    with job.lock:
        job.status = status  # type: ignore[assignment]
        job.error = error
        job.completed_at = time.perf_counter()


def _run_enhanced_index_job(job: IndexJob) -> Index:
    """Build and atomically activate an explicitly supplied enhanced snapshot.

    The builder is deliberately injected: this keeps Marker v0.1.3 as the
    default production parser while giving an authorized adapter a real
    background lifecycle seam.  No page or index is activated until the
    builder returns a complete, source-matched ``EnhancedSnapshot``.
    """
    from slide_tutor.generation import (
        begin_enhancement,
        load_generation_lifecycle,
        mark_enhancement_failed,
    )

    builder = job.enhanced_builder
    if builder is None:
        raise EnhancedRouteUnavailable(
            "No enhanced parser builder was registered for this job"
        )
    source_signature = compute_source_signature(job.project_id)
    lifecycle = load_generation_lifecycle(job.project_id, source_signature)
    if lifecycle.baseline_status != "current":
        raise RuntimeError(
            "Cannot start enhanced indexing before a current baseline snapshot exists"
        )
    begin_enhancement(job.project_id, source_signature)
    snapshot: EnhancedSnapshot | None = None
    try:
        snapshot = builder(
            job.project_id,
            lambda event: _queue_progress(job, event),
            job.cancel_event,
        )
        if not isinstance(snapshot, EnhancedSnapshot):
            raise TypeError("enhanced builder must return EnhancedSnapshot")
        if snapshot.source_signature != source_signature:
            raise ValueError(
                "enhanced snapshot source signature does not match the active project"
            )
        if not snapshot.page_generation_id:
            raise ValueError("enhanced snapshot must include a page generation id")
        _validate_enhanced_page_stage(
            job.project_id,
            snapshot.page_stage_root,
            snapshot.page_generation_id,
            snapshot.source_signature,
        )
        _validate_enhanced_index_snapshot(
            job.project_id,
            snapshot.index,
            snapshot.page_generation_id,
        )
        _check_cancelled(job.cancel_event)
        activate_enhanced_index(
            job.project_id,
            snapshot.index,
            page_stage_root=snapshot.page_stage_root,
            page_generation_id=snapshot.page_generation_id,
            source_signature=snapshot.source_signature,
            indexed_at=snapshot.indexed_at,
            progress_callback=lambda event: _queue_progress(job, event),
            cancel_signal=job.cancel_event,
        )
        return snapshot.index
    except Exception as exc:
        if snapshot is not None and isinstance(snapshot.page_stage_root, Path):
            shutil.rmtree(snapshot.page_stage_root, ignore_errors=True)
        mark_enhancement_failed(
            job.project_id,
            f"{type(exc).__name__}: {exc}",
            cancelled=isinstance(exc, IndexBuildCancelled),
        )
        raise


def _run_index_job(job: IndexJob) -> None:
    with _index_queue_condition:
        while not _index_queue or _index_queue[0] is not job:
            if job.cancel_event.is_set():
                if job in _index_queue:
                    _index_queue.remove(job)
                _finish_index_job(job, "cancelled")
                _index_queue_condition.notify_all()
                return
            _index_queue_condition.wait(timeout=0.05)
        if job.cancel_event.is_set():
            _index_queue.pop(0)
            _finish_index_job(job, "cancelled")
            _index_queue_condition.notify_all()
            return
        with job.lock:
            job.status = "running"
            job.started_at = time.perf_counter()

    try:
        if job.operation == "enhanced_preindex":
            result = _run_enhanced_index_job(job)
        else:
            builder = rebuild_index if job.operation == "explicit_preindex" else ensure_index
            result = builder(
                job.project_id,
                progress_callback=lambda event: _queue_progress(job, event),
                cancel_signal=job.cancel_event,
            )
        with job.lock:
            job.result = result
    except IndexBuildCancelled:
        _finish_index_job(job, "cancelled")
    except Exception as exc:  # surfaced through the job object; never swallowed
        _finish_index_job(job, "failed", error=f"{type(exc).__name__}: {exc}")
    else:
        _finish_index_job(job, "completed")
    finally:
        with _index_queue_condition:
            if job in _index_queue:
                _index_queue.remove(job)
            _index_queue_condition.notify_all()


def _queue_index_job(
    project_id: str,
    operation: Literal["explicit_preindex", "lazy_rebuild", "enhanced_preindex"],
    *,
    enhanced_builder: EnhancedSnapshotBuilder | None = None,
) -> IndexJob:
    """Queue one globally serialized index worker for a project.

    The worker only calls index/page-understanding code and writes progress to a
    queue.  It never calls Streamlit APIs.  A second request for the same
    project is rejected so two activations cannot cross each other.
    """
    with _index_jobs_lock:
        existing = _index_jobs.get(project_id)
        if existing is not None:
            with existing.lock:
                if existing.status in {"queued", "running"}:
                    if operation == "lazy_rebuild":
                        return existing
                    raise RuntimeError(f"Index job already running for project {project_id}")
        job = IndexJob(
            project_id=project_id,
            operation=operation,
            enhanced_builder=enhanced_builder,
            cancel_event=Event(),
            progress_queue=Queue(),
            queued_at=time.perf_counter(),
        )
        queued = IndexProgress(stage="queued", status="started")
        job.progress_queue.put(queued)
        job.last_progress = queued
        worker = Thread(
            target=_run_index_job,
            args=(job,),
            name=f"slide-tutor-index-{project_id}",
            daemon=True,
        )
        job.thread = worker
        _index_jobs[project_id] = job
        with _index_queue_condition:
            _index_queue.append(job)
        worker.start()
        return job


def start_index_job(
    project_id: str,
    operation: Literal["explicit_preindex", "lazy_rebuild"] = "explicit_preindex",
) -> IndexJob:
    """Queue one baseline/lazy index worker for a project."""
    return _queue_index_job(project_id, operation)


def start_enhanced_index_job(
    project_id: str,
    enhanced_builder: EnhancedSnapshotBuilder,
) -> IndexJob:
    """Queue an opt-in enhanced worker without changing the default parser.

    The caller must provide the authorized parser/adapter as a complete
    snapshot builder.  This API never falls back to Marker v2, a remote
    service, or a partial page/index mutation on its own.
    """
    if not callable(enhanced_builder):
        raise TypeError("enhanced_builder must be callable")
    return _queue_index_job(
        project_id,
        "enhanced_preindex",
        enhanced_builder=enhanced_builder,
    )


def ensure_index_via_job(
    project_id: str,
    cancel_signal: CancellationSignal = None,
) -> Index:
    """Load an index or wait for a shared lazy rebuild without owning its cancellation."""
    cached = load_index(project_id)
    if cached is not None:
        return cached

    job = start_index_job(project_id, operation="lazy_rebuild")
    while True:
        if _cancel_requested(cancel_signal):
            raise IndexBuildCancelled("Stopped waiting for shared index rebuild")
        with job.lock:
            status = job.status
            result = job.result
            error = job.error
        if status == "completed":
            if result is not None:
                return result
            cached = load_index(project_id)
            if cached is not None:
                return cached
            raise RuntimeError("Index job completed without an activated index")
        if status == "cancelled":
            raise IndexBuildCancelled("Shared index rebuild was cancelled")
        if status == "failed":
            raise RuntimeError(error or "Shared index rebuild failed")
        time.sleep(0.05)


def mark_index_job_terminal_delivered(project_id: str) -> bool:
    """Mark a terminal result delivered once; return True only for the first caller."""
    job = get_index_job(project_id)
    if job is None:
        return False
    with job.lock:
        if job.status in {"queued", "running"} or job.terminal_delivered:
            return False
        job.terminal_delivered = True
        return True


def clear_finished_index_job(project_id: str) -> None:
    """Forget a finished job after its result has been rendered by the UI."""
    with _index_jobs_lock:
        job = _index_jobs.get(project_id)
        if job is not None:
            with job.lock:
                if job.status not in {"queued", "running"} and job.terminal_delivered:
                    _index_jobs.pop(project_id, None)


def _index_contract(backend_type: str) -> dict[str, Any]:
    """Return the persisted retrieval contract that determines cache compatibility."""
    contract: dict[str, Any] = {
        "retrieval_schema_version": config.RETRIEVAL_SCHEMA_VERSION,
        "chunk_text_version": config.CHUNK_TEXT_VERSION,
        "lexical_index_version": config.LEXICAL_INDEX_VERSION,
        "lexical_tokenizer_version": config.LEXICAL_TOKENIZER_VERSION,
        "backend": backend_type,
    }
    from slide_tutor.page_understanding import page_content_contract_signature
    contract["page_content_contract_signature"] = page_content_contract_signature()
    if backend_type == "embedding":
        contract.update({
            "embedding_model": config.EMBEDDING_MODEL,
            "embedding_model_revision": config.EMBEDDING_MODEL_REVISION,
            "embedding_query_prefix": config.EMBEDDING_QUERY_PREFIX,
            "embedding_passage_prefix": config.EMBEDDING_PASSAGE_PREFIX,
            "normalize_embeddings": config.EMBEDDING_NORMALIZE,
        })
    return contract


def _index_config_signature(contract: dict[str, Any]) -> str:
    """Hash a canonical contract so incompatible retrieval indexes are rejected."""
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_title(value: str) -> str:
    """Normalize display differences while preserving the title as metadata."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[\u2010-\u2015_]", "-", normalized)
    normalized = re.sub(r"\s*-\s*", "-", normalized)
    return re.sub(r"\s+", " ", normalized).strip(" \t\r\n:：。?？")


def _project_file_sha256(project_id: str, file_id: str) -> str:
    """Resolve a file's immutable source hash for content-addressed keys."""
    try:
        from slide_tutor.projects import load_project

        project = load_project(project_id)
        for file in project.files:
            if file.file_id == file_id:
                return file.sha256
    except Exception:
        pass
    # Compatibility/test callers may not have a persisted project manifest.
    return file_id


def canonical_text_hash(text: str) -> str:
    """Hash the exact canonical chunk text, including its title prefix."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_chunk_key(
    file_sha256: str,
    local_page_number: int,
    canonical_text: str,
    chunker_contract: str | None = None,
) -> str:
    """Return the stable PIO-9 key for one page-level retrieval chunk."""
    payload = {
        "file_sha256": file_sha256,
        "local_page_number": local_page_number,
        "canonical_text_hash": canonical_text_hash(canonical_text),
        "chunker_contract": chunker_contract or config.CHUNK_TEXT_VERSION,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def compute_embedding_key(canonical_text: str) -> str:
    """Return the stable key for an embedding input contract."""
    payload = {
        "canonical_text_hash": canonical_text_hash(canonical_text),
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_model_revision": config.EMBEDDING_MODEL_REVISION,
        "embedding_passage_prefix": config.EMBEDDING_PASSAGE_PREFIX,
        "normalize_embeddings": config.EMBEDDING_NORMALIZE,
        "retrieval_schema_version": config.RETRIEVAL_SCHEMA_VERSION,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _chunk_with_content_keys(
    chunk: RetrievalChunk,
    *,
    file_sha256: str | None = None,
) -> RetrievalChunk:
    """Backfill stable keys on legacy or test-created chunks."""
    resolved_file_sha = file_sha256 or _project_file_sha256(chunk.project_id, chunk.file_id)
    chunk_key = compute_chunk_key(
        resolved_file_sha,
        chunk.page_number,
        chunk.text,
        config.CHUNK_TEXT_VERSION,
    )
    embedding_key = compute_embedding_key(chunk.text)
    if chunk.chunk_key == chunk_key and chunk.embedding_key == embedding_key:
        return chunk
    return chunk.model_copy(update={"chunk_key": chunk_key, "embedding_key": embedding_key})


# ── Source signature (Phase 4 Advance: pre-index reliability) ──


def compute_source_signature(project_id: str) -> str:
    """Stable hash of every ready file in the project.  The signature covers
    (file_id, source_type, sha256, page_count) sorted by file_id so file
    order changes do NOT invalidate the index.  Returns empty string when
    there are no ready files."""
    from slide_tutor.projects import load_project
    try:
        project = load_project(project_id)
    except Exception:
        return ""
    ready = sorted(
        [f for f in project.files if f.status == "ready"],
        key=lambda f: f.file_id,
    )
    if not ready:
        return ""
    root = hashlib.sha256()
    for f in ready:
        root.update(f.file_id.encode())
        root.update(b"\0")
        root.update(getattr(f, "source_type", "pdf").encode())
        root.update(b"\0")
        root.update(f.sha256.encode())
        root.update(b"\0")
        root.update(str(f.page_count).encode())
        root.update(b"\n")
    return root.hexdigest()


def get_preindex_status(project_id: str) -> PreindexStatus:
    """Classify the project's index state for sidebar display.

    Returns:
      - not_ready: no ready PDFs in the project
      - not_indexed: ready PDFs exist but no explicit pre-index was done
      - current: explicit pre-index completed and source signature matches
      - stale: source files changed since the last explicit pre-index
    """
    from slide_tutor.projects import load_project
    try:
        project = load_project(project_id)
    except Exception:
        return "not_ready"

    ready_files = [f for f in project.files if f.status == "ready"]
    if not ready_files:
        return "not_ready"

    meta = load_json(_index_meta_path(project_id), default={}) or {}
    current_sig = compute_source_signature(project_id)
    stored_sig = meta.get("source_signature")

    if not stored_sig:
        return "not_indexed"       # old project, no signature at all
    if stored_sig != current_sig:
        return "stale"             # files changed
    from slide_tutor.generation import generation_state
    generation = generation_state(project_id, current_sig)
    if generation.page_status != "current" or generation.index_status == "stale":
        return "stale"
    if meta.get("indexed_at"):
        return "current"           # explicit pre-index done, still matches
    return "lazy_current"           # lazy index exists but never explicitly pre-indexed


# ── Chunk building ──


def build_chunks(
    project_id: str,
    progress_callback: ProgressCallback | None = None,
    cancel_signal: CancellationSignal = None,
    ensure_understanding: bool = False,
    understanding_root: Path | None = None,
) -> list[RetrievalChunk]:
    """Scan every PDF's slides.json in the project.  For each page, try to read
    its PageUnderstanding.clean_markdown first; fall back to SlidePage.text for
    old projects that lack understanding cache.  Empty-text pages are skipped."""
    from slide_tutor.projects import load_project
    from slide_tutor.page_understanding import evidence_eligibility, load_page_understanding
    from slide_tutor.generation import load_page_from_root
    if ensure_understanding:
        from slide_tutor.page_understanding import ensure_page_understanding

    project = load_project(project_id)
    chunks: list[RetrievalChunk] = []
    total_pages = sum(
        getattr(file, "page_count", 0)
        for file in project.files if file.status == "ready"
    )
    completed_pages = 0
    operation_started = time.perf_counter()

    for file in project.files:
        if file.status != "ready":
            continue
        file_sha256 = getattr(file, "sha256", "") or file.file_id
        slides_data = load_json(
            get_project_dir(project_id) / "files" / file.file_id / config.SLIDES_FILENAME
        )
        if not slides_data:
            continue
        payload = FileSlidesPayload.model_validate(slides_data)
        for slide in payload.slides:
            _check_cancelled(cancel_signal)
            if ensure_understanding:
                # Explicit pre-index owns the page-understanding pass.  The
                # check before and after each page makes cancellation bounded
                # by the current parser call and keeps this worker UI-free.
                ensure_page_understanding(project_id, file.file_id, slide)
                _check_cancelled(cancel_signal)
            # Keep title as metadata and include it in the unified retrieval text.
            ud = (
                load_page_from_root(understanding_root, file.file_id, slide.slide_id)
                if understanding_root is not None
                else load_page_understanding(project_id, file.file_id, slide.slide_id)
            )
            canonical_text = getattr(ud, "canonical_markdown", "") if ud is not None else ""
            legacy_text = getattr(ud, "clean_markdown", "") if ud is not None else ""
            if ud is not None and (canonical_text or legacy_text).strip():
                body_text = (canonical_text or legacy_text).strip()
            elif slide.text.strip():
                body_text = slide.text.strip()
            else:
                body_text = ""

            title = (slide.title_guess or "").strip()
            text = f"{title}\n\n{body_text}".strip() if title else body_text
            if not text:
                completed_pages += 1
                _emit_progress(
                    progress_callback,
                    operation_started,
                    "build chunks",
                    "progress",
                    completed=completed_pages,
                    total=total_pages,
                )
                continue

            snippet = text.strip()[:150]
            chunk_fields = {
                "chunk_key": compute_chunk_key(
                    file_sha256,
                    slide.page_number,
                    text,
                    config.CHUNK_TEXT_VERSION,
                ),
                "embedding_key": compute_embedding_key(text),
                "page_quality": getattr(ud, "quality", "ok") if ud is not None else "ok",
                "route_action": getattr(ud, "route_action", "select") if ud is not None else "select",
                "selected_provider": getattr(ud, "selected_provider", "pymupdf") if ud is not None else "pymupdf",
                "evidence_eligibility": evidence_eligibility(ud) if ud is not None and hasattr(ud, "route_action") else "eligible",
                "page_content_contract_version": (
                    getattr(ud, "page_content_contract_version", PAGE_CONTENT_CONTRACT_VERSION)
                    if ud is not None else PAGE_CONTENT_CONTRACT_VERSION
                ),
                "quality_warnings": list(getattr(ud, "warnings", [])) if ud is not None else [],
                "generation_id": getattr(ud, "generation_id", None) if ud is not None else None,
            }
            chunks.append(
                RetrievalChunk(
                    chunk_id=slide.slide_id,
                    project_id=project_id,
                    file_id=file.file_id,
                    slide_id=slide.slide_id,
                    source_file=slide.source_file,
                    page_number=slide.page_number,
                    title=title,
                    normalized_title=normalize_title(title),
                    text=text,
                    snippet_reference=snippet,
                    **chunk_fields,
                )
            )
            completed_pages += 1
            _emit_progress(
                progress_callback,
                operation_started,
                "build chunks",
                "progress",
                completed=completed_pages,
                total=total_pages,
            )

    return chunks


# ── Embedding helpers ──


def _load_embedding_model() -> Any | None:
    """Try to load the configured sentence-transformers model.  Returns the model
    or None if the package is not installed or the model fails to load."""
    global _embedding_model_cache, _last_embedding_error
    model_name = config.EMBEDDING_MODEL
    model_key = (model_name, config.EMBEDDING_MODEL_REVISION)
    if _embedding_model_cache is not None and _embedding_model_cache[0] == model_key:
        return _embedding_model_cache[1]

    offline_previous = {
        key: os.environ.get(key)
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    if config.EMBEDDING_LOCAL_FILES_ONLY:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from sentence_transformers import SentenceTransformer
        kwargs = {}
        if config.EMBEDDING_MODEL_REVISION:
            kwargs["revision"] = config.EMBEDDING_MODEL_REVISION
        if config.EMBEDDING_LOCAL_FILES_ONLY:
            kwargs["model_kwargs"] = {"local_files_only": True}
            kwargs["processor_kwargs"] = {"local_files_only": True}
        try:
            model = SentenceTransformer(model_name, **kwargs)
        finally:
            for key, value in offline_previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        # Loading is kept separate from passage encoding so preflight can
        # report model availability without hiding a long warm-up in it.
        _embedding_model_cache = (model_key, model)
        _last_embedding_error = None
        return model
    except Exception as exc:
        for key, value in offline_previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _last_embedding_error = f"{type(exc).__name__}: {exc}"
        logger.info("sentence-transformers unavailable, using TF-IDF fallback (%s)", exc)
        _embedding_model_cache = (model_key, None)
        return None


def reset_embedding_model_cache() -> None:
    """Forget a failed shared-model load so a user-initiated retry can work."""
    global _embedding_model_cache, _last_embedding_error
    _embedding_model_cache = None
    _last_embedding_error = None


def _model_dimension(model: Any) -> int | None:
    getter = getattr(model, "get_embedding_dimension", None)
    if not callable(getter):
        getter = getattr(model, "get_sentence_embedding_dimension", None)
    if callable(getter):
        value = getter()
        return int(value) if value is not None else None
    return None


def _expected_embedding_dimension(project_id: str) -> int | None:
    """Use an old dimension only when model identity/revision is unchanged."""
    meta = load_json(_index_meta_path(project_id), default={}) or {}
    if meta.get("backend") != "embedding":
        return None
    if meta.get("embedding_model") != config.EMBEDDING_MODEL:
        return None
    if meta.get("embedding_model_revision") != config.EMBEDDING_MODEL_REVISION:
        return None
    dimension = meta.get("embedding_dimension")
    return int(dimension) if dimension is not None else None


def _load_embedding_model_with_timeout() -> Any | None:
    """Bound local model preflight so a slow/corrupt cache cannot freeze Q&A."""
    global _embedding_model_cache, _last_embedding_error
    model_key = (config.EMBEDDING_MODEL, config.EMBEDDING_MODEL_REVISION)
    if _embedding_model_cache is not None and _embedding_model_cache[0] == model_key:
        return _embedding_model_cache[1]

    result: dict[str, Any | None] = {}

    def load() -> None:
        result["model"] = _load_embedding_model()

    worker = Thread(target=load, name="slide-tutor-model-preflight", daemon=True)
    worker.start()
    worker.join(timeout=max(0.1, config.EMBEDDING_PREFLIGHT_TIMEOUT_SECONDS))
    if worker.is_alive():
        _last_embedding_error = (
            "model preflight timed out after "
            f"{config.EMBEDDING_PREFLIGHT_TIMEOUT_SECONDS:g}s"
        )
        # Make the current build choose the explicit fallback immediately;
        # the daemon worker may finish later and a subsequent retry can use it.
        _embedding_model_cache = (model_key, None)
        return None
    return result.get("model")


def preflight_shared_model(expected_dimension: int | None = None) -> ModelPreflight:
    """Check shared model availability, revision and embedding dimension.

    The check is intentionally explicit: callers can show the reason to the
    user and decide whether an explicit TF-IDF fallback is acceptable.
    """
    model = _load_embedding_model_with_timeout()
    if model is None:
        return ModelPreflight(
            available=False,
            model_id=config.EMBEDDING_MODEL,
            revision=config.EMBEDDING_MODEL_REVISION,
            dimension=None,
            reason=_last_embedding_error or "model unavailable",
            backend="tfidf",
        )

    dimension: int | None = None
    try:
        dimension = _model_dimension(model)
        if dimension is None:
            probe = model.encode(
                [f"{config.EMBEDDING_QUERY_PREFIX}preflight"],
                convert_to_numpy=True,
                normalize_embeddings=config.EMBEDDING_NORMALIZE,
            )
            dimension = int(np.asarray(probe).shape[-1])
    except Exception as exc:
        return ModelPreflight(
            available=False,
            model_id=config.EMBEDDING_MODEL,
            revision=config.EMBEDDING_MODEL_REVISION,
            dimension=None,
            reason=f"dimension probe failed: {type(exc).__name__}: {exc}",
            backend="tfidf",
        )

    if expected_dimension is not None and dimension != expected_dimension:
        return ModelPreflight(
            available=False,
            model_id=config.EMBEDDING_MODEL,
            revision=config.EMBEDDING_MODEL_REVISION,
            dimension=dimension,
            reason=(
                f"embedding dimension mismatch: expected {expected_dimension}, "
                f"got {dimension}"
            ),
            backend="tfidf",
        )

    return ModelPreflight(
        available=True,
        model_id=config.EMBEDDING_MODEL,
        revision=config.EMBEDDING_MODEL_REVISION,
        dimension=dimension,
        reason="ready",
        backend="embedding",
    )


# ── Index building ──


def _build_tfidf_index(chunks: list[RetrievalChunk]) -> tuple[TfidfVectorizer, csr_matrix]:
    """Build TF-IDF vocabulary and matrix from chunk texts."""
    texts = [chunk.text for chunk in chunks]
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix


def _build_embedding_index(
    chunks: list[RetrievalChunk],
    model: Any,
    progress_callback: ProgressCallback | None = None,
    cancel_signal: CancellationSignal = None,
) -> tuple[Any, np.ndarray]:
    """Encode all chunk texts into a dense embedding matrix."""
    global _last_embedding_reuse_stats
    texts = [f"{config.EMBEDDING_PASSAGE_PREFIX}{chunk.text}" for chunk in chunks]
    if not texts:
        _last_embedding_reuse_stats = {
            "reused": 0, "encoded": 0, "total": 0, "source": "fresh"
        }
        return model, np.empty((0, 0), dtype=np.float32)

    batch_size = max(1, int(getattr(config, "RETRIEVAL_EMBEDDING_BATCH_SIZE", 32)))
    batches: list[np.ndarray] = []
    operation_started = time.perf_counter()
    for start in range(0, len(texts), batch_size):
        _check_cancelled(cancel_signal)
        batch = model.encode(
            texts[start:start + batch_size],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=config.EMBEDDING_NORMALIZE,
        )
        batches.append(np.asarray(batch))
        completed = min(start + batch_size, len(texts))
        _emit_progress(
            progress_callback,
            operation_started,
            "encode passages",
            "progress",
            completed=completed,
            total=len(texts),
            backend="embedding",
        )
    embeddings = np.vstack(batches)
    _last_embedding_reuse_stats = {
        "reused": 0,
        "encoded": len(texts),
        "total": len(texts),
        "source": "fresh",
    }
    return model, embeddings


def _load_previous_embedding_snapshot(
    project_id: str,
) -> tuple[list[RetrievalChunk], np.ndarray] | None:
    """Read a prior embedding snapshot without requiring its source signature.

    The current source may have added/replaced/reordered files.  Contract and
    model identity still have to match, but individual rows are selected by
    their stable embedding keys rather than by the old chunk position.
    """
    try:
        meta = load_json(_index_meta_path(project_id), default={}) or {}
        if meta.get("backend") != "embedding":
            return None
        current_contract = _index_contract("embedding")
        if meta.get("index_config_signature") != _index_config_signature(current_contract):
            return None
        chunks_json = load_json(_chunks_path(project_id), default=None)
        if not isinstance(chunks_json, list):
            return None
        chunks = [RetrievalChunk.model_validate(raw) for raw in chunks_json]
        embeddings_path = _embeddings_path(project_id)
        if not embeddings_path.exists():
            return None
        matrix = np.asarray(np.load(str(embeddings_path), allow_pickle=False))
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
            return None
        if meta.get("embedding_dimension") is not None and int(meta["embedding_dimension"]) != matrix.shape[1]:
            return None
        return chunks, matrix
    except Exception:
        return None


def _build_embedding_index_reusing(
    project_id: str,
    chunks: list[RetrievalChunk],
    model: Any,
    progress_callback: ProgressCallback | None = None,
    cancel_signal: CancellationSignal = None,
) -> tuple[Any, np.ndarray]:
    """Build embeddings while reusing rows whose stable keys are unchanged."""
    global _last_embedding_reuse_stats
    if not chunks or not all(isinstance(chunk, RetrievalChunk) for chunk in chunks):
        _last_embedding_reuse_stats = {
            "reused": 0, "encoded": len(chunks), "total": len(chunks), "source": "fresh"
        }
        return _build_embedding_index(chunks, model, progress_callback, cancel_signal)

    current_chunks = [
        _chunk_with_content_keys(chunk) for chunk in chunks
    ]
    previous = _load_previous_embedding_snapshot(project_id)
    previous_rows: dict[str, np.ndarray] = {}
    previous_source = "fresh"
    if previous is not None:
        previous_chunks, previous_matrix = previous
        for row, previous_chunk in enumerate(previous_chunks):
            normalized = _chunk_with_content_keys(previous_chunk)
            key = normalized.embedding_key
            if key and key not in previous_rows:
                previous_rows[key] = previous_matrix[row]
        previous_source = "previous_index"

    dimension = _model_dimension(model)
    if dimension is None and previous is not None:
        dimension = int(previous[1].shape[1])
    if dimension is None:
        probe = model.encode(
            [f"{config.EMBEDDING_QUERY_PREFIX}preflight"],
            convert_to_numpy=True,
            normalize_embeddings=config.EMBEDDING_NORMALIZE,
        )
        dimension = int(np.asarray(probe).shape[-1])

    matrix = np.empty((len(current_chunks), dimension), dtype=np.float32)
    missing: list[tuple[int, RetrievalChunk]] = []
    reused = 0
    for row, chunk in enumerate(current_chunks):
        previous_row = previous_rows.get(chunk.embedding_key)
        if previous_row is not None and previous_row.shape[-1] == dimension:
            matrix[row] = previous_row
            reused += 1
        else:
            missing.append((row, chunk))

    batch_size = max(1, int(getattr(config, "RETRIEVAL_EMBEDDING_BATCH_SIZE", 32)))
    encoded = 0
    operation_started = time.perf_counter()
    for start in range(0, len(missing), batch_size):
        _check_cancelled(cancel_signal)
        batch = missing[start:start + batch_size]
        texts = [
            f"{config.EMBEDDING_PASSAGE_PREFIX}{chunk.text}"
            for _row, chunk in batch
        ]
        encoded_rows = np.asarray(
            model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=config.EMBEDDING_NORMALIZE,
            )
        )
        if encoded_rows.ndim != 2 or encoded_rows.shape[1] != dimension:
            raise EmbeddingDimensionMismatch(
                f"embedding reuse produced shape {encoded_rows.shape}; expected (*, {dimension})"
            )
        for (row, _chunk), encoded_row in zip(batch, encoded_rows):
            matrix[row] = encoded_row
        encoded += len(batch)
        _emit_progress(
            progress_callback,
            operation_started,
            "encode passages",
            "progress",
            completed=min(reused + encoded, len(current_chunks)),
            total=len(current_chunks),
            reused=reused,
            backend="embedding",
            dimension=dimension,
            detail=f"reused {reused}; encoded {encoded}",
        )

    if not missing:
        _emit_progress(
            progress_callback,
            operation_started,
            "encode passages",
            "progress",
            completed=len(current_chunks),
            total=len(current_chunks),
            reused=reused,
            backend="embedding",
            dimension=dimension,
            detail=f"reused {reused}; encoded 0",
        )
    _last_embedding_reuse_stats = {
        "reused": reused,
        "encoded": encoded,
        "total": len(current_chunks),
        "source": previous_source,
    }
    return model, matrix


def build_index(
    chunks: list[RetrievalChunk],
    progress_callback: ProgressCallback | None = None,
    cancel_signal: CancellationSignal = None,
) -> Index:
    """Build the best available index with an explicit TF-IDF fallback."""
    _check_cancelled(cancel_signal)
    model = _load_embedding_model()
    if model is not None:
        backend, matrix = _build_embedding_index(
            chunks, model, progress_callback, cancel_signal,
        )
        return chunks, backend, matrix

    _check_cancelled(cancel_signal)
    backend, matrix = _build_tfidf_index(chunks)
    return chunks, backend, matrix


_DEFAULT_BUILD_INDEX = build_index


# ── Persistence ──


def _write_staged_index(
    project_id: str,
    index: Index,
    stage_root: Path,
    indexed_at: str | None = None,
    generation_id: str | None = None,
    page_generation_id: str | None = None,
) -> None:
    """Write a complete index generation below ``stage_root`` only."""
    chunks, backend, matrix = index
    index_dir = stage_root / config.INDEX_DIRNAME
    index_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = stage_root / config.CHUNKS_FILENAME

    # chunks are always saved as JSON
    save_json(chunks_path, [c.model_dump(mode="json") for c in chunks])

    if isinstance(matrix, np.ndarray):
        # Embedding mode
        backend_type = "embedding"
        np.save(str(index_dir / "embeddings.npy"), matrix)
        # The model is loaded from the shared Hugging Face/application cache.
        # Project indexes persist only embeddings and the encoding contract.
        (index_dir / "backend.txt").write_text("embedding", encoding="utf-8")
    else:
        # TF-IDF mode
        backend_type = "tfidf"
        with open(index_dir / "vectorizer.pkl", "wb") as f:
            pickle.dump(backend, f)
        save_npz(str(index_dir / "tfidf_matrix.npz"), matrix)
        (index_dir / "backend.txt").write_text("tfidf", encoding="utf-8")

    source_signature = compute_source_signature(project_id)
    previous_meta = load_json(_index_meta_path(project_id), default={}) or {}
    contract = _index_contract(backend_type)
    metadata = {
        "schema_version": config.SCHEMA_VERSION,
        **contract,
        "index_config_signature": _index_config_signature(contract),
        "embedding_dimension": int(matrix.shape[1]) if isinstance(matrix, np.ndarray) else None,
        "chunk_count": len(chunks),
        "created_at": _utc_now(),
        "generation_id": generation_id or f"index-{uuid4().hex}",
        "page_generation_id": page_generation_id,
        "source_signature": source_signature,
        "page_content_contract_signature": contract.get("page_content_contract_signature"),
        "retrieval_contract_signature": _index_config_signature(contract),
        "model_preflight": {
            "model_id": config.EMBEDDING_MODEL,
            "revision": config.EMBEDDING_MODEL_REVISION,
            "backend": backend_type,
            "dimension": int(matrix.shape[1]) if isinstance(matrix, np.ndarray) else None,
            "reason": "ready" if backend_type == "embedding" else (
                _last_embedding_error or "model unavailable; using TF-IDF fallback"
            ),
        },
        "embedding_reuse": dict(_last_embedding_reuse_stats),
    }
    if indexed_at:
        metadata["indexed_at"] = indexed_at
    # Algorithm-only rebuilds should not erase a completed pre-index marker.
    elif previous_meta.get("source_signature") == source_signature and previous_meta.get("indexed_at"):
        metadata["indexed_at"] = previous_meta["indexed_at"]
    save_json(index_dir / "index_meta.json", metadata)


def _activate_staged_index(
    project_id: str,
    stage_root: Path,
    *,
    page_stage_root: Path | None = None,
    page_generation_id: str | None = None,
    activation_layer: Literal["baseline", "enhanced"] = "baseline",
) -> None:
    """Switch a fully written generation into place, with rollback protection."""
    project_dir = get_project_dir(project_id)
    project_dir.mkdir(parents=True, exist_ok=True)
    final_chunks = _chunks_path(project_id)
    final_index = _index_dir(project_id)
    backup_root = Path(tempfile.mkdtemp(prefix=".index-rollback-", dir=project_dir))
    backup_chunks = backup_root / config.CHUNKS_FILENAME
    backup_index = backup_root / config.INDEX_DIRNAME
    page_meta_path = get_project_dir(project_id) / "page_generation_meta.json"
    backup_page_meta = backup_root / "page_generation_meta.json"
    lifecycle_path = get_project_dir(project_id) / config.GENERATION_LIFECYCLE_FILENAME
    backup_lifecycle = backup_root / config.GENERATION_LIFECYCLE_FILENAME
    generations_dir = get_project_dir(project_id) / "generations"
    active_page_generation_dir = None
    backup_page_generation_dir = None
    old_chunks = final_chunks.exists()
    old_index = final_index.exists()
    new_chunks_activated = False
    new_index_activated = False
    new_page_activated = False
    new_lifecycle_activated = False
    lifecycle_backed_up = False

    try:
        if old_chunks:
            os.replace(final_chunks, backup_chunks)
        if old_index:
            os.replace(final_index, backup_index)
        if lifecycle_path.exists():
            os.replace(lifecycle_path, backup_lifecycle)
            lifecycle_backed_up = True

        if page_stage_root is not None and page_generation_id:
            generations_dir.mkdir(parents=True, exist_ok=True)
            active_meta = load_json(page_meta_path, default={}) or {}
            old_storage_root = active_meta.get("storage_root")
            if page_meta_path.exists():
                os.replace(page_meta_path, backup_page_meta)
            if old_storage_root:
                old_dir = get_project_dir(project_id) / old_storage_root
                if old_dir.exists():
                    backup_page_generation_dir = backup_root / "old-page-generation"
                    os.replace(old_dir, backup_page_generation_dir)
            active_page_generation_dir = generations_dir / page_generation_id
            os.replace(page_stage_root / config.UNDERSTANDING_DIRNAME, active_page_generation_dir)
            new_page_activated = True
            staged_meta = load_json(page_stage_root / "page_generation_meta.json", default={}) or {}
            staged_meta.update({
                "storage_root": str(active_page_generation_dir.relative_to(get_project_dir(project_id))),
                "status": "current",
            })
            save_json(page_meta_path, staged_meta)

        os.replace(stage_root / config.CHUNKS_FILENAME, final_chunks)
        new_chunks_activated = True
        os.replace(stage_root / config.INDEX_DIRNAME, final_index)
        new_index_activated = True
        # The page/index pair is now coherent.  Persist the lifecycle marker
        # only after both filesystem roots have moved, so a failed activation
        # cannot advertise an enhanced/current snapshot that readers cannot load.
        staged_index_meta = load_json(final_index / "index_meta.json", default={}) or {}
        active_page_meta = load_json(page_meta_path, default={}) or {}
        new_lifecycle_activated = True
        if activation_layer == "enhanced":
            from slide_tutor.generation import mark_enhancement_current
            mark_enhancement_current(
                project_id,
                page_generation_id=active_page_meta.get("generation_id") or page_generation_id,
                index_generation_id=staged_index_meta.get("generation_id"),
                source_signature=staged_index_meta.get("source_signature", ""),
            )
        else:
            from slide_tutor.generation import mark_baseline_current
            mark_baseline_current(
                project_id,
                page_generation_id=active_page_meta.get("generation_id"),
                index_generation_id=staged_index_meta.get("generation_id"),
                source_signature=staged_index_meta.get("source_signature", ""),
            )
        # An active generation change invalidates the project-level lexical corpus.
        from slide_tutor.retrieval import clear_bm25_cache
        clear_bm25_cache(project_id)
    except Exception:
        # Remove any new generation that was moved into the final location.
        if new_chunks_activated and final_chunks.exists():
            final_chunks.unlink(missing_ok=True)
        if new_index_activated and final_index.exists():
            shutil.rmtree(final_index, ignore_errors=True)
        if new_page_activated:
            if page_meta_path.exists():
                page_meta_path.unlink(missing_ok=True)
            if active_page_generation_dir is not None and active_page_generation_dir.exists():
                shutil.rmtree(active_page_generation_dir, ignore_errors=True)
        if (new_lifecycle_activated or lifecycle_backed_up) and lifecycle_path.exists():
            lifecycle_path.unlink(missing_ok=True)
        # Restore the previous complete generation where possible.
        if backup_chunks.exists():
            os.replace(backup_chunks, final_chunks)
        if backup_index.exists():
            os.replace(backup_index, final_index)
        if backup_page_generation_dir is not None and backup_page_generation_dir.exists():
            # Restore the previous generation at the same relative path.
            old_meta = load_json(backup_page_meta, default={}) or {}
            old_storage_root = old_meta.get("storage_root")
            if old_storage_root:
                restored_dir = get_project_dir(project_id) / old_storage_root
                restored_dir.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup_page_generation_dir, restored_dir)
        if backup_page_meta.exists():
            os.replace(backup_page_meta, page_meta_path)
        if backup_lifecycle.exists():
            os.replace(backup_lifecycle, lifecycle_path)
        raise
    finally:
        shutil.rmtree(backup_root, ignore_errors=True)


def _persist_index_atomic(
    project_id: str,
    index: Index,
    *,
    indexed_at: str | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_signal: CancellationSignal = None,
    page_stage_root: Path | None = None,
    page_generation_id: str | None = None,
    activation_layer: Literal["baseline", "enhanced"] = "baseline",
) -> None:
    """Persist a complete generation, then activate it with rollback protection.

    ``chunks.json`` and ``index/`` are switched in two filesystem operations;
    the project lock prevents same-process readers from crossing that window,
    while the rollback directory restores the previous complete generation on
    any activation failure.
    """
    project_dir = get_project_dir(project_id)
    project_dir.mkdir(parents=True, exist_ok=True)
    operation_started = time.perf_counter()
    stage_root = Path(tempfile.mkdtemp(prefix=".index-build-", dir=project_dir))
    backend = "embedding" if isinstance(index[2], np.ndarray) else "tfidf"
    dimension = int(index[2].shape[1]) if backend == "embedding" else None
    fallback_reason = (
        (_last_embedding_error or "model unavailable; using TF-IDF fallback")
        if backend == "tfidf" else ""
    )
    try:
        _emit_progress(
            progress_callback,
            operation_started,
            "persist index",
            "started",
            total=len(index[0]),
            backend=backend,
            dimension=dimension,
            fallback_reason=fallback_reason,
        )
        _check_cancelled(cancel_signal)
        _write_staged_index(
            project_id,
            index,
            stage_root,
            indexed_at=indexed_at,
            generation_id=page_generation_id,
            page_generation_id=page_generation_id,
        )
        _check_cancelled(cancel_signal)
        _emit_progress(
            progress_callback,
            operation_started,
            "persist index",
            "completed",
            completed=len(index[0]),
            total=len(index[0]),
            backend=backend,
            dimension=dimension,
            fallback_reason=fallback_reason,
        )
    except IndexBuildCancelled:
        shutil.rmtree(stage_root, ignore_errors=True)
        if page_stage_root is not None:
            shutil.rmtree(page_stage_root, ignore_errors=True)
        _emit_progress(
            progress_callback,
            operation_started,
            "persist index",
            "cancelled",
            backend=backend,
            dimension=dimension,
            fallback_reason=fallback_reason,
            detail="cancelled before activation; staged generation discarded",
        )
        raise
    except Exception as exc:
        shutil.rmtree(stage_root, ignore_errors=True)
        if page_stage_root is not None:
            shutil.rmtree(page_stage_root, ignore_errors=True)
        _emit_progress(
            progress_callback,
            operation_started,
            "persist index",
            "failed",
            backend=backend,
            dimension=dimension,
            fallback_reason=fallback_reason,
            detail=f"{type(exc).__name__}: {exc}",
        )
        raise

    _emit_progress(
        progress_callback,
        operation_started,
        "activate index",
        "started",
        total=len(index[0]),
        backend=backend,
        dimension=dimension,
        fallback_reason=fallback_reason,
    )
    try:
        _check_cancelled(cancel_signal)
        _activate_staged_index(
            project_id,
            stage_root,
            page_stage_root=page_stage_root,
            page_generation_id=page_generation_id,
            activation_layer=activation_layer,
        )
        _emit_progress(
            progress_callback,
            operation_started,
            "activate index",
            "completed",
            completed=len(index[0]),
            total=len(index[0]),
            backend=backend,
            dimension=dimension,
            fallback_reason=fallback_reason,
        )
    except IndexBuildCancelled:
        _emit_progress(
            progress_callback,
            operation_started,
            "activate index",
            "cancelled",
            backend=backend,
            dimension=dimension,
            fallback_reason=fallback_reason,
            detail="cancelled before staged activation; previous index preserved",
        )
        raise
    except Exception as exc:
        _emit_progress(
            progress_callback,
            operation_started,
            "activate index",
            "failed",
            backend=backend,
            dimension=dimension,
            fallback_reason=fallback_reason,
            detail=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)
        if page_stage_root is not None:
            shutil.rmtree(page_stage_root, ignore_errors=True)


def save_index(
    project_id: str,
    index: Index,
    *,
    indexed_at: str | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_signal: CancellationSignal = None,
    page_stage_root: Path | None = None,
    page_generation_id: str | None = None,
    activation_layer: Literal["baseline", "enhanced"] = "baseline",
) -> None:
    """Persist a generation through staged activation under the project lock."""
    with _project_index_lock(project_id):
        _persist_index_atomic(
            project_id,
            index,
            indexed_at=indexed_at,
            progress_callback=progress_callback,
            cancel_signal=cancel_signal,
            page_stage_root=page_stage_root,
            page_generation_id=page_generation_id,
            activation_layer=activation_layer,
        )


def activate_enhanced_index(
    project_id: str,
    index: Index,
    *,
    page_stage_root: Path,
    page_generation_id: str,
    source_signature: str = "",
    indexed_at: str | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_signal: CancellationSignal = None,
) -> None:
    """Activate a complete enhanced page/index stage after baseline exists.

    Enhanced callers must provide a full page stage and matching generation id;
    index-only mutation is intentionally rejected because it would make the
    active page/index pair incoherent.  Until activation succeeds, the current
    baseline remains the only active query surface.
    """
    if page_stage_root is None or not page_generation_id:
        raise ValueError("Enhanced activation requires a complete page stage")
    effective_source_signature = source_signature or compute_source_signature(project_id)
    try:
        _validate_enhanced_page_stage(
            project_id,
            page_stage_root,
            page_generation_id,
            effective_source_signature,
        )
        _validate_enhanced_index_snapshot(
            project_id,
            index,
            page_generation_id,
        )
    except Exception:
        # The caller handed us a staged root; an invalid snapshot must never
        # leave an unowned page stage behind for a later activation attempt.
        shutil.rmtree(page_stage_root, ignore_errors=True)
        raise
    from slide_tutor.generation import (
        begin_enhancement,
        mark_enhancement_failed,
    )
    with _project_index_lock(project_id):
        begin_enhancement(project_id, effective_source_signature)
        try:
            _persist_index_atomic(
                project_id,
                index,
                indexed_at=indexed_at,
                progress_callback=progress_callback,
                cancel_signal=cancel_signal,
                page_stage_root=page_stage_root,
                page_generation_id=page_generation_id,
                activation_layer="enhanced",
            )
        except Exception as exc:
            mark_enhancement_failed(
                project_id,
                f"{type(exc).__name__}: {exc}",
                cancelled=isinstance(exc, IndexBuildCancelled),
            )
            raise


def _load_index_unlocked(project_id: str) -> Index | None:
    """Load an index while the caller holds the project lock."""
    chunks_json = load_json(_chunks_path(project_id))
    if not chunks_json:
        return None
    chunks = [RetrievalChunk.model_validate(c) for c in chunks_json]

    flag_path = _backend_flag_path(project_id)
    if not flag_path.exists():
        return None
    backend_type = flag_path.read_text(encoding="utf-8").strip()
    meta = load_json(_index_meta_path(project_id), default={}) or {}
    if meta.get("backend") and meta["backend"] != backend_type:
        return None
    if meta.get("chunk_count") is not None and meta["chunk_count"] != len(chunks):
        return None
    if meta.get("page_generation_id"):
        from slide_tutor.page_understanding import active_page_generation
        active_page = active_page_generation(project_id)
        if active_page.get("generation_id") != meta.get("page_generation_id"):
            logger.info("Page/index generation mismatch — triggering rebuild for %s", project_id)
            return None
    # Phase 4 Advance: missing or stale signature → force rebuild
    stored_sig = meta.get("source_signature")
    if not stored_sig:
        logger.info("No source signature — old index discarded for %s", project_id)
        return None
    if stored_sig != compute_source_signature(project_id):
        logger.info("Source signature changed — triggering index rebuild for %s", project_id)
        return None

    contract = _index_contract(backend_type)
    if meta.get("index_config_signature") != _index_config_signature(contract):
        logger.info("Retrieval contract changed — triggering index rebuild for %s", project_id)
        return None

    if backend_type == "embedding":
        if meta.get("embedding_model") != config.EMBEDDING_MODEL:
            return None
        emb_path = _embeddings_path(project_id)
        if not emb_path.exists():
            return None
        matrix = np.load(str(emb_path))
        if matrix.ndim != 2 or meta.get("embedding_dimension") != matrix.shape[1]:
            return None
        # Try to reload the model; if it's not available, fall back to TF-IDF rebuild
        model = _load_embedding_model_with_timeout()
        if model is None:
            return None  # trigger rebuild with TF-IDF
        model_dimension = _model_dimension(model)
        if model_dimension is None or model_dimension != matrix.shape[1]:
            logger.warning(
                "Embedding dimension mismatch while loading %s: metadata=%s, "
                "matrix=%s, model=%s",
                project_id,
                meta.get("embedding_dimension"),
                matrix.shape[1],
                model_dimension,
            )
            return None
        return chunks, model, matrix
    else:
        # If embeddings are now available, rebuild old TF-IDF indexes into the
        # preferred semantic backend instead of keeping the weaker cache forever.
        if _load_embedding_model_with_timeout() is not None:
            return None
        vec_path = _vec_pkl_path(project_id)
        mat_path = _tfidf_matrix_path(project_id)
        if not vec_path.exists() or not mat_path.exists():
            return None
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
        tfidf_matrix = load_npz(str(mat_path))
        return chunks, vectorizer, tfidf_matrix


def load_index(project_id: str) -> Index | None:
    """Load a complete generation under the project's process-local lock."""
    with _project_index_lock(project_id):
        return _load_index_unlocked(project_id)


def _call_with_build_options(
    function: Callable[..., Any],
    *args: Any,
    progress_callback: ProgressCallback | None,
    cancel_signal: CancellationSignal,
    ensure_understanding: bool | None = None,
    understanding_root: Path | None = None,
) -> Any:
    signature = inspect.signature(function)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs: dict[str, Any] = {}
    if progress_callback is not None and (
        accepts_kwargs or "progress_callback" in signature.parameters
    ):
        kwargs["progress_callback"] = progress_callback
    if cancel_signal is not None and (
        accepts_kwargs or "cancel_signal" in signature.parameters
    ):
        kwargs["cancel_signal"] = cancel_signal
    if ensure_understanding is not None and (
        accepts_kwargs or "ensure_understanding" in signature.parameters
    ):
        kwargs["ensure_understanding"] = ensure_understanding
    if understanding_root is not None and (
        accepts_kwargs or "understanding_root" in signature.parameters
    ):
        kwargs["understanding_root"] = understanding_root
    return function(*args, **kwargs)


def _maybe_stage_page_generation(
    project_id: str,
    *,
    cancel_signal: CancellationSignal,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Path | None, str | None]:
    """Prepare canonical pages only when the active page generation is stale."""
    from slide_tutor.generation import (
        begin_baseline_build,
        build_page_generation,
        mark_baseline_recovered,
        validate_active_page_generation,
    )
    from slide_tutor.page_understanding import active_page_generation
    from slide_tutor.projects import load_file_slides, load_project

    try:
        project = load_project(project_id)
    except Exception:
        # Compatibility seam for legacy/injected unit-test builders.
        return None, None
    source_signature = compute_source_signature(project_id)
    files_and_slides = [
        (file.file_id, load_file_slides(project_id, file.file_id))
        for file in project.files if file.status == "ready"
    ]
    if not any(slides for _file_id, slides in files_and_slides):
        mark_baseline_recovered(project_id, "no ready file page manifests")
        return None, None
    begin_baseline_build(project_id, source_signature)
    integrity = validate_active_page_generation(
        project_id, files_and_slides, source_signature=source_signature,
    )
    if integrity.status == "current":
        active_meta = active_page_generation(project_id)
        if progress_callback is not None:
            progress_callback(IndexProgress(
                stage="build page content",
                status="completed",
                completed=integrity.expected_count,
                total=integrity.expected_count,
                reused=integrity.expected_count,
                detail="active page generation reused",
            ))
        return None, active_meta["generation_id"]
    try:
        root, metadata = build_page_generation(
            project_id,
            files_and_slides,
            source_signature=source_signature,
            cancel_signal=cancel_signal,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        mark_baseline_recovered(
            project_id,
            f"{type(exc).__name__}: {exc}",
            cancelled=isinstance(exc, IndexBuildCancelled),
        )
        raise
    return root, metadata.generation_id


def _build_index_with_progress(
    project_id: str,
    *,
    progress_callback: ProgressCallback | None = None,
    cancel_signal: CancellationSignal = None,
    ensure_understanding: bool = False,
    understanding_root: Path | None = None,
) -> Index:
    global _last_embedding_reuse_stats
    _last_embedding_reuse_stats = {
        "reused": 0,
        "encoded": 0,
        "total": 0,
        "source": "not_embedding",
    }
    operation_started = time.perf_counter()

    def emit_stage_error(stage: str, exc: Exception) -> None:
        status: Literal["failed", "cancelled"] = (
            "cancelled" if isinstance(exc, IndexBuildCancelled) else "failed"
        )
        _emit_progress(
            progress_callback,
            operation_started,
            stage,
            status,
            detail=f"{type(exc).__name__}: {exc}",
        )

    _emit_progress(
        progress_callback,
        operation_started,
        "load shared model",
        "started",
    )
    try:
        _check_cancelled(cancel_signal)
        if build_index is _DEFAULT_BUILD_INDEX:
            preflight = preflight_shared_model(
                expected_dimension=_expected_embedding_dimension(project_id),
            )
            if (
                not preflight.available
                and preflight.dimension is not None
                and "dimension mismatch" in preflight.reason
            ):
                raise EmbeddingDimensionMismatch(preflight.reason)
        else:
            # Unit tests and controlled benchmark backends can inject a builder;
            # they must never trigger a real model download as a side effect.
            preflight = ModelPreflight(
                available=True,
                model_id=config.EMBEDDING_MODEL,
                revision=config.EMBEDDING_MODEL_REVISION,
                dimension=None,
                reason="injected backend",
                backend="unknown",
            )
    except Exception as exc:
        emit_stage_error("load shared model", exc)
        raise
    _emit_progress(
        progress_callback,
        operation_started,
        "load shared model",
        "completed",
        backend=preflight.backend,
        dimension=preflight.dimension,
        fallback_reason=(preflight.reason if preflight.backend == "tfidf" else ""),
        detail=preflight.reason,
    )

    _emit_progress(
        progress_callback,
        operation_started,
        "build chunks",
        "started",
    )
    try:
        chunks = _call_with_build_options(
            build_chunks,
            project_id,
            progress_callback=progress_callback,
            cancel_signal=cancel_signal,
            ensure_understanding=ensure_understanding,
            understanding_root=understanding_root,
        )
        _check_cancelled(cancel_signal)
        chunks = [
            _chunk_with_content_keys(chunk)
            if isinstance(chunk, RetrievalChunk) else chunk
            for chunk in chunks
        ]
    except Exception as exc:
        emit_stage_error("build chunks", exc)
        raise
    _emit_progress(
        progress_callback,
        operation_started,
        "build chunks",
        "completed",
        completed=len(chunks),
        total=len(chunks),
    )
    _check_cancelled(cancel_signal)

    _emit_progress(
        progress_callback,
        operation_started,
        "encode passages",
        "started",
        total=len(chunks),
        backend=preflight.backend,
        dimension=preflight.dimension,
        fallback_reason=(preflight.reason if preflight.backend == "tfidf" else ""),
    )
    try:
        if build_index is _DEFAULT_BUILD_INDEX and preflight.backend == "embedding":
            model = _load_embedding_model()
            if model is None:
                index = _call_with_build_options(
                    build_index,
                    chunks,
                    progress_callback=progress_callback,
                    cancel_signal=cancel_signal,
                )
            else:
                backend, matrix = _build_embedding_index_reusing(
                    project_id,
                    chunks,
                    model,
                    progress_callback=progress_callback,
                    cancel_signal=cancel_signal,
                )
                index = chunks, backend, matrix
        else:
            index = _call_with_build_options(
                build_index,
                chunks,
                progress_callback=progress_callback,
                cancel_signal=cancel_signal,
            )
        _check_cancelled(cancel_signal)
        actual_backend = "embedding" if isinstance(index[2], np.ndarray) else "tfidf"
        actual_dimension = int(index[2].shape[1]) if actual_backend == "embedding" else None
        if (
            actual_backend == "embedding"
            and preflight.dimension is not None
            and actual_dimension != preflight.dimension
        ):
            raise EmbeddingDimensionMismatch(
                "embedding dimension mismatch after encoding: "
                f"expected {preflight.dimension}, got {actual_dimension}; "
                f"model={preflight.model_id}, revision={preflight.revision}"
            )
    except Exception as exc:
        emit_stage_error("encode passages", exc)
        raise
    backend = "embedding" if isinstance(index[2], np.ndarray) else "tfidf"
    _emit_progress(
        progress_callback,
        operation_started,
        "encode passages",
        "completed",
        completed=len(chunks),
        total=len(chunks),
        backend=backend,
        dimension=actual_dimension,
        fallback_reason=(preflight.reason if backend == "tfidf" else ""),
        detail=(preflight.reason if backend == "tfidf" else "ready"),
    )
    return index


def ensure_index(
    project_id: str,
    progress_callback: ProgressCallback | None = None,
    cancel_signal: CancellationSignal = None,
) -> Index:
    """Load a cached index or rebuild it with observable progress."""
    with _project_index_lock(project_id):
        cached = _load_index_unlocked(project_id)
        if cached is not None:
            return cached
        page_stage_root, page_generation_id = _maybe_stage_page_generation(
            project_id,
            cancel_signal=cancel_signal,
            progress_callback=progress_callback,
        )
        try:
            index = _build_index_with_progress(
                project_id,
                progress_callback=progress_callback,
                cancel_signal=cancel_signal,
                ensure_understanding=page_stage_root is None,
                understanding_root=page_stage_root
                if page_stage_root is not None else None,
            )
        except Exception as exc:
            if page_stage_root is not None:
                shutil.rmtree(page_stage_root, ignore_errors=True)
            from slide_tutor.generation import mark_baseline_recovered
            mark_baseline_recovered(
                project_id,
                f"{type(exc).__name__}: {exc}",
                cancelled=isinstance(exc, IndexBuildCancelled),
            )
            raise
        try:
            _persist_index_atomic(
                project_id,
                index,
                progress_callback=progress_callback,
                cancel_signal=cancel_signal,
                page_stage_root=page_stage_root,
                page_generation_id=page_generation_id,
            )
        except Exception as exc:
            from slide_tutor.generation import mark_baseline_recovered
            mark_baseline_recovered(
                project_id,
                f"{type(exc).__name__}: {exc}",
                cancelled=isinstance(exc, IndexBuildCancelled),
            )
            raise
        finally:
            if page_stage_root is not None and page_stage_root.exists():
                shutil.rmtree(page_stage_root, ignore_errors=True)
        from slide_tutor.generation import mark_baseline_current
        from slide_tutor.page_understanding import active_page_generation
        active_page_meta = active_page_generation(project_id)
        index_meta = load_json(_index_meta_path(project_id), default={}) or {}
        mark_baseline_current(
            project_id,
            page_generation_id=active_page_meta.get("generation_id"),
            index_generation_id=index_meta.get("generation_id"),
            source_signature=index_meta.get("source_signature", ""),
        )
        return index


def rebuild_index(
    project_id: str,
    progress_callback: ProgressCallback | None = None,
    cancel_signal: CancellationSignal = None,
) -> Index:
    """Force-rebuild chunks + index from the latest PageUnderstanding cache.
    Old index artefacts are overwritten.  Used by the 'Pre-index all pages' button.

    Writes indexed_at as part of the same metadata write so the pre-index
    timestamp and source_signature are always consistent."""
    with _project_index_lock(project_id):
        page_stage_root, page_generation_id = _maybe_stage_page_generation(
            project_id,
            cancel_signal=cancel_signal,
            progress_callback=progress_callback,
        )
        try:
            index = _build_index_with_progress(
                project_id,
                progress_callback=progress_callback,
                cancel_signal=cancel_signal,
                ensure_understanding=page_stage_root is None,
                understanding_root=page_stage_root
                if page_stage_root is not None else None,
            )
        except Exception as exc:
            if page_stage_root is not None:
                shutil.rmtree(page_stage_root, ignore_errors=True)
            from slide_tutor.generation import mark_baseline_recovered
            mark_baseline_recovered(
                project_id,
                f"{type(exc).__name__}: {exc}",
                cancelled=isinstance(exc, IndexBuildCancelled),
            )
            raise
        from slide_tutor.projects import utc_now
        indexed_at = utc_now()
        try:
            if (
                page_stage_root is not None
                or page_generation_id is not None
                or progress_callback is not None
                or cancel_signal is not None
            ):
                _persist_index_atomic(
                    project_id,
                    index,
                    indexed_at=indexed_at,
                    progress_callback=progress_callback,
                    cancel_signal=cancel_signal,
                    page_stage_root=page_stage_root,
                    page_generation_id=page_generation_id,
                )
            else:
                # Keep the small compatibility seam used by existing integrations
                # that monkeypatch save_index, while the normal implementation
                # receives indexed_at atomically.
                try:
                    save_index(project_id, index, indexed_at=indexed_at)
                except TypeError as exc:
                    if "indexed_at" not in str(exc):
                        raise
                    save_index(project_id, index)
                    meta = load_json(_index_meta_path(project_id), default={}) or {}
                    meta["indexed_at"] = indexed_at
                    save_json(_index_meta_path(project_id), meta)
            from slide_tutor.generation import mark_baseline_current
            from slide_tutor.page_understanding import active_page_generation
            active_page_meta = active_page_generation(project_id)
            index_meta = load_json(_index_meta_path(project_id), default={}) or {}
            mark_baseline_current(
                project_id,
                page_generation_id=active_page_meta.get("generation_id"),
                index_generation_id=index_meta.get("generation_id"),
                source_signature=index_meta.get("source_signature", ""),
            )
            return index
        except Exception as exc:
            from slide_tutor.generation import mark_baseline_recovered
            mark_baseline_recovered(
                project_id,
                f"{type(exc).__name__}: {exc}",
                cancelled=isinstance(exc, IndexBuildCancelled),
            )
            raise
        finally:
            if page_stage_root is not None and page_stage_root.exists():
                shutil.rmtree(page_stage_root, ignore_errors=True)

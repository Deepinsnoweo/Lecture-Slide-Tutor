"""Rerun-safe, Streamlit-free Q&A generation jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Event, Lock, RLock, Thread
from typing import Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from slide_tutor.llm import LLMCancelled
from slide_tutor.models import QAActionType, QAContextBundle, QAMemoryTurn, SlidePage


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class QAJobRequest(BaseModel):
    """Immutable input snapshot captured before a Streamlit rerun can occur."""

    task_id: str = Field(default_factory=lambda: f"task-{uuid4().hex}")
    session_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    file_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    action_type: QAActionType
    language: str = Field(min_length=1)
    question: str = ""
    current_slide: SlidePage
    file_page_count: int = Field(ge=1)
    memory: list[dict] = Field(default_factory=list)


class QAJobResult(BaseModel):
    answer_markdown: str = ""
    context_bundle: QAContextBundle | None = None
    quality_warning: str | None = None


@dataclass(frozen=True)
class QAJobSnapshot:
    task_id: str
    session_id: str
    project_id: str
    file_id: str
    source_file: str
    page_number: int
    action_type: QAActionType
    language: str
    question: str
    status: Literal["pending", "running", "completed", "cancelled", "failed"]
    partial_text: str
    context_bundle: QAContextBundle | None
    quality_warning: str | None
    error: str | None
    created_at: str
    started_at: str | None
    completed_at: str | None
    cancel_requested: bool


Pipeline = Callable[[QAJobRequest, Event, Callable[[str], None]], QAJobResult]


@dataclass
class _QAJob:
    request: QAJobRequest
    cancel_event: Event = field(default_factory=Event)
    status: str = "pending"
    partial_text: str = ""
    context_bundle: QAContextBundle | None = None
    quality_warning: str | None = None
    error: str | None = None
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    completed_at: str | None = None
    thread: Thread | None = None
    lock: RLock = field(default_factory=RLock)

    def snapshot(self) -> QAJobSnapshot:
        with self.lock:
            return QAJobSnapshot(
                task_id=self.request.task_id,
                session_id=self.request.session_id,
                project_id=self.request.project_id,
                file_id=self.request.file_id,
                source_file=self.request.source_file,
                page_number=self.request.page_number,
                action_type=self.request.action_type,
                language=self.request.language,
                question=self.request.question,
                status=self.status,  # type: ignore[arg-type]
                partial_text=self.partial_text,
                context_bundle=self.context_bundle,
                quality_warning=self.quality_warning,
                error=self.error,
                created_at=self.created_at,
                started_at=self.started_at,
                completed_at=self.completed_at,
                cancel_requested=self.cancel_event.is_set(),
            )


class QAJobRegistry:
    """Thread-safe in-process registry; no Streamlit session state is touched."""

    def __init__(self, *, finished_limit: int = 32) -> None:
        self._jobs: dict[str, _QAJob] = {}
        self._lock = Lock()
        self.finished_limit = max(1, finished_limit)

    def _active_for_session(self, session_id: str) -> _QAJob | None:
        for job in self._jobs.values():
            if job.request.session_id == session_id and job.status in {"pending", "running"}:
                return job
        return None

    def start(self, request: QAJobRequest, pipeline: Pipeline) -> QAJobSnapshot:
        with self._lock:
            if self._active_for_session(request.session_id) is not None:
                raise RuntimeError("This session already has an active generation task")
            if request.task_id in self._jobs:
                raise RuntimeError(f"Task already exists: {request.task_id}")
            job = _QAJob(request=request)
            self._jobs[request.task_id] = job
            worker = Thread(
                target=self._run,
                args=(job, pipeline),
                name=f"slide-tutor-qa-{request.task_id}",
                daemon=True,
            )
            job.thread = worker
            worker.start()
            return job.snapshot()

    def _run(self, job: _QAJob, pipeline: Pipeline) -> None:
        with job.lock:
            job.status = "running"
            job.started_at = _now()

        def emit(delta: str) -> None:
            if not delta:
                return
            with job.lock:
                # Append-only updates make rerun snapshots monotonic.
                job.partial_text += delta

        try:
            result = pipeline(job.request, job.cancel_event, emit)
            with job.lock:
                job.context_bundle = result.context_bundle
                job.quality_warning = result.quality_warning
                if result.answer_markdown and not job.partial_text:
                    job.partial_text = result.answer_markdown
                if job.cancel_event.is_set():
                    job.status = "cancelled"
                elif not job.partial_text.strip():
                    job.status = "failed"
                    job.error = "Provider returned an empty completion"
                else:
                    job.status = "completed"
                job.completed_at = _now()
        except LLMCancelled:
            with job.lock:
                job.status = "cancelled"
                job.completed_at = _now()
        except Exception as exc:  # surfaced through snapshot, never swallowed
            with job.lock:
                if job.cancel_event.is_set():
                    job.status = "cancelled"
                else:
                    job.status = "failed"
                    job.error = f"{type(exc).__name__}: {exc}"
                job.completed_at = _now()

    def snapshot(self, task_id: str) -> QAJobSnapshot | None:
        with self._lock:
            job = self._jobs.get(task_id)
        return job.snapshot() if job is not None else None

    def active_snapshot(self, session_id: str) -> QAJobSnapshot | None:
        with self._lock:
            jobs = [
                job for job in self._jobs.values()
                if job.request.session_id == session_id and job.status in {"pending", "running"}
            ]
        return jobs[0].snapshot() if jobs else None

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(task_id)
        if job is None:
            return False
        with job.lock:
            if job.status not in {"pending", "running"}:
                return False
            job.cancel_event.set()
            return True

    def cleanup(self, *, session_id: str | None = None) -> int:
        """Remove only finished jobs, retaining a bounded recent history."""
        with self._lock:
            finished = [
                job for job in self._jobs.values()
                if job.status in {"completed", "cancelled", "failed"}
                and (session_id is None or job.request.session_id == session_id)
            ]
            finished.sort(key=lambda job: job.completed_at or job.created_at)
            remove = finished[:-self.finished_limit]
            for job in remove:
                self._jobs.pop(job.request.task_id, None)
            return len(remove)

    def clear(self) -> None:
        with self._lock:
            running = [job for job in self._jobs.values() if job.status in {"pending", "running"}]
            if running:
                raise RuntimeError("Cannot clear registry while jobs are running")
            self._jobs.clear()


qa_job_registry = QAJobRegistry()


__all__ = [
    "QAJobRequest", "QAJobResult", "QAJobSnapshot", "QAJobRegistry", "qa_job_registry",
]

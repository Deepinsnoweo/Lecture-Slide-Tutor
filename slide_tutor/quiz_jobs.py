"""Rerun-safe background generation for per-PDF quizzes and final exams."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from threading import Event, Lock, RLock, Thread
from typing import Any, Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from slide_tutor.models import QUIZ_PROMPT_VERSION, GenerationSnapshot, QuizSet


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class QuizGenerationRequest(BaseModel):
    """Immutable generation inputs captured before Streamlit reruns."""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(default_factory=lambda: f"quiz-task-{uuid4().hex}")
    session_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    quiz_type: Literal["per_pdf", "final_exam"]
    file_id: str | None = None
    target_count: int = Field(ge=1)
    language: str = Field(min_length=1)
    question_type_mix: str = Field(min_length=1)
    quality_mode: Literal["fast", "reviewed"] = "fast"
    generation_snapshot: GenerationSnapshot | None = None
    blueprint_schema_version: str = ""
    planner_prompt_version: str = ""
    review_prompt_version: str = ""

    @model_validator(mode="after")
    def validate_file_scope(self) -> "QuizGenerationRequest":
        if self.quiz_type == "per_pdf" and not self.file_id:
            raise ValueError("per_pdf quiz generation requires file_id")
        if self.quality_mode == "reviewed" and self.quiz_type != "final_exam":
            raise ValueError("Reviewed quality mode is only available for final exams")
        return self

    @property
    def request_signature(self) -> str:
        raw = "|".join([
            self.project_id, self.quiz_type, self.file_id or "", str(self.target_count),
            self.language, self.question_type_mix, self.quality_mode,
            self.blueprint_schema_version, self.planner_prompt_version,
            self.review_prompt_version,
            QUIZ_PROMPT_VERSION,
            "fast:medium" if self.quality_mode == "fast" else "planner:high,generation:medium,review:high",
            self.generation_snapshot.model_dump_json() if self.generation_snapshot else "",
        ])
        return sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class QuizGenerationSnapshot:
    task_id: str
    session_id: str
    project_id: str
    quiz_type: Literal["per_pdf", "final_exam"]
    file_id: str | None
    target_count: int
    language: str
    question_type_mix: str
    quality_mode: Literal["fast", "reviewed"]
    generation_snapshot: GenerationSnapshot | None
    request_signature: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    result: QuizSet | None
    error: str | None
    created_at: str
    started_at: str | None
    completed_at: str | None
    terminal_claimed: bool
    phase: str
    phase_detail: str
    progress_current: int
    progress_total: int
    cancel_requested: bool


QuizGenerator = Callable[..., QuizSet]


class QuizGenerationCancelled(RuntimeError):
    """Internal terminal signal used to avoid activating a cancelled result."""


def generate_quiz_request(
    request: QuizGenerationRequest,
    cancel_signal: Event | None = None,
    report: Callable[[str, str], None] | None = None,
) -> QuizSet:
    """Production generator kept Streamlit-free for worker use."""
    from slide_tutor import config
    from slide_tutor.models import QUIZ_PROMPT_VERSION, QuizReuseSummary
    from slide_tutor.projects import load_project, ordered_files, utc_now
    from slide_tutor.quiz import (
        _normalize_question_type_mix,
        collect_reusable_wrong_questions,
        finalize_fast_quiz_set,
        generate_final_exam,
        generate_per_pdf_quiz,
        save_cached_quiz,
    )

    def cancelled() -> bool:
        return bool(cancel_signal and cancel_signal.is_set())

    def stage(name: str, detail: str = "") -> None:
        if cancelled():
            raise QuizGenerationCancelled("Quiz generation cancelled")
        if report:
            report(name, detail)

    if request.quality_mode == "reviewed":
        from slide_tutor.quiz_quality import run_reviewed_final_exam
        reviewed_mix = _normalize_question_type_mix(
            request.question_type_mix, request.target_count,
        )
        try:
            result = run_reviewed_final_exam(
                request.project_id,
                target_count=request.target_count,
                language=request.language,
                question_type_mix=reviewed_mix,
                generation_snapshot=request.generation_snapshot,
                progress=report,
                cancel_requested=cancelled,
            )
        except Exception as exc:
            if cancelled():
                raise QuizGenerationCancelled("Quiz generation cancelled after current Provider batch") from exc
            raise
        result.generation_request_signature = request.request_signature
    else:
        required_mix = _normalize_question_type_mix(
            request.question_type_mix, request.target_count,
        )
        if request.quiz_type == "final_exam":
            project = load_project(request.project_id)
            file_ids = [file.file_id for file in ordered_files(project) if file.status == "ready"]
        else:
            file_ids = [request.file_id] if request.file_id else []
        reused, source_attempt_ids = collect_reusable_wrong_questions(
            request.project_id,
            file_ids=file_ids,
            language=request.language,
            question_type_mix=request.question_type_mix,
            request_signature=request.request_signature,
            generation_snapshot=request.generation_snapshot,
            required_mix=required_mix,
        )
        reused_counts = {question_type: 0 for question_type in required_mix}
        for question in reused:
            reused_counts[question.question_type] += 1
        missing_mix = {
            question_type: max(required_mix[question_type] - reused_counts[question_type], 0)
            for question_type in required_mix
        }
        missing_count = sum(missing_mix.values())
        generated: list[Any] = []
        base: QuizSet | None = None
        if missing_count:
            stage(
                "generation",
                f"reusing {len(reused)} wrong question(s); generating {missing_count} new",
            )
            if request.quiz_type == "final_exam":
                base = generate_final_exam(
                    request.project_id, missing_count, request.language,
                    missing_mix, request.generation_snapshot, allow_partial=True,
                )
            else:
                base = generate_per_pdf_quiz(
                    request.project_id, request.file_id, missing_count,
                    request.language, missing_mix, request.generation_snapshot,
                    allow_partial=True,
                )
            generated = list(base.questions)
        else:
            stage("generation", f"reusing {len(reused)} unresolved wrong question(s)")
        if base is None:
            base = QuizSet(
                project_id=request.project_id,
                file_ids=file_ids,
                total_count=request.target_count,
                difficulty_mix="review of unresolved wrong questions",
                question_type_mix=request.question_type_mix,
                questions=[],
                created_at=utc_now(),
                model=config.LLM_MODEL,
                prompt_version=QUIZ_PROMPT_VERSION,
                language=request.language,
                quality_mode="fast",
                reuse_summary=QuizReuseSummary(),
            )
        base.questions = [*reused, *generated]
        base.total_count = len(base.questions)
        base.file_ids = file_ids
        base.question_type_mix = request.question_type_mix
        from slide_tutor.topics import resolve_quiz_topics
        result = resolve_quiz_topics(base)
        result = finalize_fast_quiz_set(
            result,
            request_signature=request.request_signature,
            generation_snapshot=request.generation_snapshot,
            reused_count=len(reused),
            generated_count=len(generated),
            source_attempt_ids=source_attempt_ids,
        )
    if cancelled():
        raise QuizGenerationCancelled("Quiz generation cancelled before cache activation")
    stage("finalizing", "saving complete quiz cache")
    save_cached_quiz(result)
    return result


@dataclass
class _QuizGenerationJob:
    request: QuizGenerationRequest
    status: str = "pending"
    result: QuizSet | None = None
    error: str | None = None
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    completed_at: str | None = None
    terminal_claimed: bool = False
    phase: str = "queued"
    phase_detail: str = ""
    progress_current: int = 0
    progress_total: int = 0
    cancel_event: Event = field(default_factory=Event)
    thread: Thread | None = None
    lock: RLock = field(default_factory=RLock)

    def snapshot(self) -> QuizGenerationSnapshot:
        with self.lock:
            return QuizGenerationSnapshot(
                task_id=self.request.task_id,
                session_id=self.request.session_id,
                project_id=self.request.project_id,
                quiz_type=self.request.quiz_type,
                file_id=self.request.file_id,
                target_count=self.request.target_count,
                language=self.request.language,
                question_type_mix=self.request.question_type_mix,
                quality_mode=self.request.quality_mode,
                generation_snapshot=self.request.generation_snapshot,
                request_signature=self.request.request_signature,
                status=self.status,  # type: ignore[arg-type]
                result=self.result,
                error=self.error,
                created_at=self.created_at,
                started_at=self.started_at,
                completed_at=self.completed_at,
                terminal_claimed=self.terminal_claimed,
                phase=self.phase,
                phase_detail=self.phase_detail,
                progress_current=self.progress_current,
                progress_total=self.progress_total,
                cancel_requested=self.cancel_event.is_set(),
            )


class QuizGenerationRegistry:
    """One active Quiz/Exam generation per browser session."""

    def __init__(self) -> None:
        self._jobs: dict[str, _QuizGenerationJob] = {}
        self._lock = Lock()

    def _active_for_session(self, session_id: str) -> _QuizGenerationJob | None:
        return next((
            job for job in self._jobs.values()
            if job.request.session_id == session_id and job.status in {"pending", "running"}
        ), None)

    def start(
        self,
        request: QuizGenerationRequest,
        generator: QuizGenerator = generate_quiz_request,
    ) -> QuizGenerationSnapshot:
        with self._lock:
            active = self._active_for_session(request.session_id)
            if active is not None:
                if active.request.request_signature == request.request_signature:
                    raise RuntimeError("This quiz generation request is already active")
                raise RuntimeError("This session already has an active quiz generation task")
            if request.task_id in self._jobs:
                raise RuntimeError(f"Quiz task already exists: {request.task_id}")
            job = _QuizGenerationJob(request=request)
            self._jobs[request.task_id] = job
            worker = Thread(
                target=self._run, args=(job, generator),
                name=f"slide-tutor-quiz-{request.task_id}", daemon=True,
            )
            job.thread = worker
            worker.start()
            return job.snapshot()

    @staticmethod
    def _invoke_generator(
        generator: QuizGenerator,
        request: QuizGenerationRequest,
        cancel_event: Event,
        report: Callable[[str, str], None],
    ) -> QuizSet:
        """Support old one-argument test generators while exposing stage hooks."""
        import inspect

        parameters = inspect.signature(generator).parameters.values()
        accepts_varargs = any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters)
        positional = [
            parameter for parameter in parameters
            if parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        ]
        if accepts_varargs or len(positional) >= 3:
            return generator(request, cancel_event, report)
        if len(positional) >= 2:
            return generator(request, cancel_event)
        return generator(request)

    @staticmethod
    def _run(job: _QuizGenerationJob, generator: QuizGenerator) -> None:
        with job.lock:
            job.status = "running"
            job.started_at = _now()
            job.phase = "running"

        def report(phase: str, detail: str = "", current: int = 0, total: int = 0) -> None:
            with job.lock:
                job.phase = phase
                job.phase_detail = detail
                job.progress_current = current
                job.progress_total = total
            if job.cancel_event.is_set():
                raise QuizGenerationCancelled("Quiz generation cancelled")
        try:
            result = QuizGenerationRegistry._invoke_generator(
                generator, job.request, job.cancel_event, report,
            )
            if job.cancel_event.is_set():
                raise QuizGenerationCancelled("Quiz generation cancelled")
        except QuizGenerationCancelled as exc:
            with job.lock:
                job.status = "cancelled"
                job.error = str(exc)
                job.completed_at = _now()
                job.phase = "cancelled"
        except Exception as exc:
            with job.lock:
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.completed_at = _now()
                job.phase = "failed"
        else:
            with job.lock:
                job.result = result
                job.status = "completed"
                job.completed_at = _now()
                job.phase = "completed"

    def snapshot(self, task_id: str) -> QuizGenerationSnapshot | None:
        with self._lock:
            job = self._jobs.get(task_id)
        return job.snapshot() if job is not None else None

    def active_snapshot(self, session_id: str) -> QuizGenerationSnapshot | None:
        with self._lock:
            job = self._active_for_session(session_id)
        return job.snapshot() if job is not None else None

    def claim_terminal(self, task_id: str) -> QuizGenerationSnapshot | None:
        """Claim one terminal result once so UI reruns cannot double-commit it."""
        with self._lock:
            job = self._jobs.get(task_id)
        if job is None:
            return None
        with job.lock:
            if job.status in {"pending", "running"} or job.terminal_claimed:
                return None
            job.terminal_claimed = True
            return job.snapshot()

    def cancel(self, task_id: str) -> bool:
        """Request cancellation; the worker observes it at every stage boundary."""
        with self._lock:
            job = self._jobs.get(task_id)
        if job is None:
            return False
        with job.lock:
            if job.status not in {"pending", "running"}:
                return False
            job.cancel_event.set()
            job.phase_detail = "cancellation requested"
            return True


quiz_generation_registry = QuizGenerationRegistry()


__all__ = [
    "QuizGenerationCancelled", "QuizGenerationRegistry", "QuizGenerationRequest", "QuizGenerationSnapshot",
    "generate_quiz_request", "quiz_generation_registry",
]

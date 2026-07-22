from __future__ import annotations

from threading import Event

import pytest

from slide_tutor.models import QuizSet
from slide_tutor.quiz_jobs import (
    QuizGenerationRegistry,
    QuizGenerationRequest,
    generate_quiz_request,
)


def _request(*, task_id: str = "task-1", language: str = "中文") -> QuizGenerationRequest:
    return QuizGenerationRequest(
        task_id=task_id,
        session_id="session-1",
        project_id="project-1",
        quiz_type="per_pdf",
        file_id="file-1",
        target_count=3,
        language=language,
        question_type_mix="mcq:2,true_false:1,fill_blank:0,short_answer:0",
    )


def _result(language: str = "中文") -> QuizSet:
    return QuizSet(
        quiz_set_id="quiz-1", project_id="project-1", file_ids=["file-1"],
        total_count=0, difficulty_mix="mixed",
        question_type_mix="mcq:0,true_false:0,fill_blank:0,short_answer:0",
        language=language,
    )


def test_background_quiz_job_completes_and_terminal_is_claimed_once():
    registry = QuizGenerationRegistry()
    request = _request(language="日本語")
    snapshot = registry.start(request, lambda req: _result(req.language))
    job = registry._jobs[snapshot.task_id]
    job.thread.join(timeout=1)

    completed = registry.snapshot(snapshot.task_id)
    assert completed.status == "completed"
    assert completed.result.language == "日本語"
    assert registry.claim_terminal(snapshot.task_id) is not None
    assert registry.claim_terminal(snapshot.task_id) is None


def test_duplicate_active_signature_and_other_active_request_are_rejected():
    registry = QuizGenerationRegistry()
    release = Event()
    first = _request()
    registry.start(first, lambda _req: (release.wait(timeout=1), _result())[1])

    with pytest.raises(RuntimeError, match="already active"):
        registry.start(first.model_copy(update={"task_id": "task-2"}), lambda _req: _result())
    with pytest.raises(RuntimeError, match="active quiz generation"):
        registry.start(
            _request(task_id="task-3", language="English"), lambda _req: _result(),
        )
    release.set()
    registry._jobs[first.task_id].thread.join(timeout=1)


def test_background_failure_is_stable_in_snapshot():
    registry = QuizGenerationRegistry()

    def fail(_request):
        raise ValueError("provider broke")

    snapshot = registry.start(_request(), fail)
    registry._jobs[snapshot.task_id].thread.join(timeout=1)
    failed = registry.snapshot(snapshot.task_id)

    assert failed.status == "failed"
    assert "provider broke" in failed.error
    assert registry.claim_terminal(snapshot.task_id).status == "failed"


def test_reviewed_request_isolated_and_cancelled_before_activation():
    registry = QuizGenerationRegistry()
    started = Event()
    release = Event()
    request = _request().model_copy(update={
        "quiz_type": "final_exam",
        "file_id": None,
        "quality_mode": "reviewed",
        "question_type_mix": "mcq:3,true_false:0,fill_blank:0,short_answer:0",
    })

    def generator(_request, cancel_event):
        started.set()
        while not cancel_event.is_set():
            release.wait(timeout=0.01)
        return _result()

    snapshot = registry.start(request, generator)
    assert started.wait(timeout=1)
    assert registry.cancel(snapshot.task_id) is True
    registry._jobs[snapshot.task_id].thread.join(timeout=1)
    terminal = registry.claim_terminal(snapshot.task_id)
    assert terminal is not None
    assert terminal.status == "cancelled"
    assert terminal.cancel_requested is True


def test_job_snapshot_exposes_batch_progress():
    registry = QuizGenerationRegistry()
    request = QuizGenerationRequest(
        session_id="progress-session", project_id="p", quiz_type="per_pdf",
        file_id="f", target_count=3, language="English",
        question_type_mix="mcq:3,true_false:0,fill_blank:0,short_answer:0",
    )

    def generator(_request, _cancel_event, report):
        report("generation", "batch 1/3", 1, 3)
        return _result("English")

    started = registry.start(request, generator=generator)
    registry._jobs[started.task_id].thread.join(timeout=1)
    terminal = registry.snapshot(started.task_id)
    assert terminal.progress_current == 1
    assert terminal.progress_total == 3


def test_reviewed_worker_forwards_requested_mixed_type_quota(monkeypatch):
    captured = {}

    def fake_reviewed(*_args, **kwargs):
        captured.update(kwargs)
        return _result().model_copy(update={"quality_mode": "reviewed"})

    monkeypatch.setattr("slide_tutor.quiz_quality.run_reviewed_final_exam", fake_reviewed)
    monkeypatch.setattr("slide_tutor.quiz.save_cached_quiz", lambda _quiz_set: None)
    request = QuizGenerationRequest(
        session_id="mixed-reviewed", project_id="p", quiz_type="final_exam",
        target_count=4, language="English", quality_mode="reviewed",
        question_type_mix="mcq:1,true_false:1,fill_blank:1,short_answer:1",
    )

    generate_quiz_request(request)

    assert captured["question_type_mix"] == {
        "mcq": 1, "true_false": 1, "fill_blank": 1, "short_answer": 1,
    }

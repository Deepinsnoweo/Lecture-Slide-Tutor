from __future__ import annotations

from dataclasses import replace
import time
from threading import Event

import pytest

from slide_tutor.llm import LLMCancelled
from slide_tutor.models import QAContextBundle, QAContextSource, SlidePage
from slide_tutor.qa_jobs import QAJobRegistry, QAJobRequest, QAJobResult
from slide_tutor.qa_records import (
    append_memory_turn,
    finalize_qa_snapshot,
    memory_turn_from_record,
    record_from_snapshot,
    should_enter_memory,
    upsert_answer_record,
)


def _request(session="s", action="ask"):
    slide = SlidePage(
        slide_id="slide-1", module_id="p", project_id="p", file_id="f",
        source_file="lecture.pdf", source_type="pdf", page_number=1,
        global_index=1, title_guess="Title", text="Body", text_quality="ok",
    )
    return QAJobRequest(
        session_id=session, project_id="p", file_id="f", source_file="lecture.pdf",
        page_number=1, action_type=action, language="中文", question="What?",
        current_slide=slide, file_page_count=3,
    )


def test_registry_rejects_second_active_task_and_snapshot_is_monotonic():
    registry = QAJobRegistry()

    def pipeline(request, cancel, emit):
        emit("a")
        time.sleep(0.02)
        emit("b")
        return QAJobResult(answer_markdown="ab")

    first = registry.start(_request(), pipeline)
    with pytest.raises(RuntimeError):
        registry.start(_request(), pipeline)
    lengths = [len(first.partial_text)]
    while True:
        snapshot = registry.snapshot(first.task_id)
        assert snapshot is not None
        lengths.append(len(snapshot.partial_text))
        if snapshot.status not in {"pending", "running"}:
            break
        time.sleep(0.005)
    assert lengths == sorted(lengths)
    assert snapshot.status == "completed"
    assert snapshot.partial_text == "ab"


def test_cancelled_stream_preserves_partial_text_and_is_not_failed():
    registry = QAJobRegistry()

    def pipeline(request, cancel, emit):
        emit("partial")
        cancel.wait(timeout=1)
        raise LLMCancelled("stopped")

    initial = registry.start(_request(), pipeline)
    assert registry.cancel(initial.task_id) is True
    while True:
        snapshot = registry.snapshot(initial.task_id)
        if snapshot.status not in {"pending", "running"}:
            break
        time.sleep(0.005)
    assert snapshot.status == "cancelled"
    assert snapshot.partial_text == "partial"
    assert snapshot.error is None


def test_worker_exception_is_readable_and_cleanup_keeps_running_jobs():
    registry = QAJobRegistry(finished_limit=1)

    def fail(request, cancel, emit):
        raise ValueError("provider broke")

    task = registry.start(_request("failure"), fail)
    task_snapshot = registry.snapshot(task.task_id)
    while task_snapshot.status in {"pending", "running"}:
        time.sleep(0.005)
        task_snapshot = registry.snapshot(task.task_id)
    assert task_snapshot.status == "failed"
    assert "provider broke" in task_snapshot.error
    assert registry.cleanup(session_id="failure") == 0


def test_empty_completion_becomes_failed_terminal_snapshot():
    registry = QAJobRegistry()

    def empty(request, cancel, emit):
        return QAJobResult(answer_markdown="")

    task = registry.start(_request("empty"), empty)
    snapshot = registry.snapshot(task.task_id)
    while snapshot.status in {"pending", "running"}:
        time.sleep(0.005)
        snapshot = registry.snapshot(task.task_id)
    assert snapshot.status == "failed"
    assert snapshot.error == "Provider returned an empty completion"
    record = record_from_snapshot(snapshot)
    assert record.status == "failed"


def test_cancel_without_partial_text_remains_a_valid_cancelled_record():
    registry = QAJobRegistry()

    def wait_for_cancel(request, cancel, emit):
        cancel.wait(timeout=1)
        raise LLMCancelled("stopped")

    task = registry.start(_request("empty-cancel"), wait_for_cancel)
    assert registry.cancel(task.task_id)
    snapshot = registry.snapshot(task.task_id)
    while snapshot.status in {"pending", "running"}:
        time.sleep(0.005)
        snapshot = registry.snapshot(task.task_id)
    record = record_from_snapshot(snapshot)
    assert record.status == "cancelled"
    assert record.answer_markdown == ""


def test_finalization_transition_is_atomic_and_idempotent():
    registry = QAJobRegistry()

    def pipeline(request, cancel, emit):
        emit("answer")
        return QAJobResult(answer_markdown="answer")

    task = registry.start(_request("finalize"), pipeline)
    snapshot = registry.snapshot(task.task_id)
    while snapshot.status in {"pending", "running"}:
        time.sleep(0.005)
        snapshot = registry.snapshot(task.task_id)
    first = finalize_qa_snapshot(snapshot, [], [], set())
    assert first.changed
    assert len(first.records) == 1
    assert len(first.memory) == 1
    second = finalize_qa_snapshot(
        snapshot, first.records, first.memory, first.submitted_task_ids,
    )
    assert not second.changed
    assert len(second.records) == 1
    assert len(second.memory) == 1


def test_finalization_error_is_preserved_as_visible_failed_record():
    registry = QAJobRegistry()

    def pipeline(request, cancel, emit):
        emit("answer")
        return QAJobResult(answer_markdown="answer")

    task = registry.start(_request("bad-finalize"), pipeline)
    snapshot = registry.snapshot(task.task_id)
    while snapshot.status in {"pending", "running"}:
        time.sleep(0.005)
        snapshot = registry.snapshot(task.task_id)
    malformed = replace(snapshot, partial_text="")
    transition = finalize_qa_snapshot(malformed, [], [], set())
    assert transition.changed
    assert transition.record is not None
    assert transition.record.status == "failed"
    assert "FinalizationError" in (transition.record.error or "")
    assert transition.memory == []


def test_answer_record_and_memory_policy_are_idempotent():
    registry = QAJobRegistry()

    def pipeline(request, cancel, emit):
        emit("answer")
        return QAJobResult(
            answer_markdown="answer",
            context_bundle=QAContextBundle(
                prompt_context="body",
                sources=[QAContextSource(kind="selected_page", source_file="lecture.pdf", page_number=1)],
            ),
        )

    task = registry.start(_request(action="ask"), pipeline)
    snapshot = registry.snapshot(task.task_id)
    while snapshot.status in {"pending", "running"}:
        time.sleep(0.005)
        snapshot = registry.snapshot(task.task_id)
    record = record_from_snapshot(snapshot)
    records = upsert_answer_record([], record)
    records = upsert_answer_record(records, record)
    assert len(records) == 1
    assert should_enter_memory(record)
    assert memory_turn_from_record(record) is not None
    assert len(append_memory_turn([], record)) == 1

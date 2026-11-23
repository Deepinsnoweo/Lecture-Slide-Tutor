"""Visible answer records and their intentionally separate memory policy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import os
from threading import RLock
from typing import Any

from slide_tutor import cache
from slide_tutor.models import (
    AnswerHistoryScope,
    QAAnswerPageGroup,
    QAAnswerProjection,
    QAAnswerRecord,
    QAAnswerHistoryBundle,
    QACurrentResponse,
    QAMemoryTurn,
)
from slide_tutor.qa_jobs import QAJobSnapshot


_HISTORY_FILENAME = "qa_answer_history_v1.json"
_HISTORY_LOCK = RLock()


@dataclass(frozen=True)
class QAFinalizationState:
    """A fully computed session-state transition for one terminal snapshot."""

    record: QAAnswerRecord | None
    records: list[QAAnswerRecord]
    memory: list[dict[str, Any]]
    submitted_task_ids: set[str]
    changed: bool


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _history_path(project_id: str):
    return cache.get_project_dir(project_id) / _HISTORY_FILENAME


def _persistent_record(record: QAAnswerRecord) -> QAAnswerRecord:
    """Strip prompt text and short-term memory while preserving provenance."""
    context = record.context_bundle
    if context is not None:
        context = context.model_copy(deep=True, update={
            "prompt_context": "",
            "memory_turns": [],
        })
    return record.model_copy(deep=True, update={"context_bundle": context})


def load_project_answer_history(project_id: str) -> list[QAAnswerRecord]:
    """Load one project history bundle; legacy raw record lists remain readable."""
    raw = cache.load_json(_history_path(project_id), default=None)
    if raw is None:
        return []
    try:
        if isinstance(raw, list):
            records = [QAAnswerRecord.model_validate(item) for item in raw]
            return [record for record in records if record.project_id == project_id]
        return QAAnswerHistoryBundle.model_validate(raw).records
    except Exception:
        return []


def save_project_answer_history(
    project_id: str,
    records: list[QAAnswerRecord],
) -> list[QAAnswerRecord]:
    """Atomically replace one project bundle after complete serialization."""
    project_records = [
        _persistent_record(record)
        for record in records
        if record.project_id == project_id
    ]
    project_records.sort(key=lambda item: item.created_at, reverse=True)
    bundle = QAAnswerHistoryBundle(
        project_id=project_id,
        records=project_records,
        updated_at=_now(),
    )
    path = _history_path(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.urandom(6).hex()}")
    with _HISTORY_LOCK:
        try:
            cache.save_json(temporary, bundle)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return project_records


def persist_answer_record(record: QAAnswerRecord) -> list[QAAnswerRecord]:
    """Idempotently commit one terminal task into its project history."""
    with _HISTORY_LOCK:
        records = load_project_answer_history(record.project_id)
        records = upsert_answer_record(records, _persistent_record(record))
        return save_project_answer_history(record.project_id, records)


def delete_project_answer_record(project_id: str, record_id: str) -> list[QAAnswerRecord]:
    with _HISTORY_LOCK:
        records = [
            record for record in load_project_answer_history(project_id)
            if record.record_id != record_id
        ]
        return save_project_answer_history(project_id, records)


def clear_file_answer_history(
    project_id: str, file_id: str, record_ids: frozenset[str],
) -> list[QAAnswerRecord]:
    """Delete the confirmed records from this file in the latest saved history."""
    with _HISTORY_LOCK:
        records = load_project_answer_history(project_id)
        remaining = [
            record for record in records
            if record.file_id != file_id or record.record_id not in record_ids
        ]
        return save_project_answer_history(project_id, remaining)


def clear_project_answer_history(project_id: str) -> list[QAAnswerRecord]:
    return save_project_answer_history(project_id, [])


def record_from_snapshot(snapshot: QAJobSnapshot) -> QAAnswerRecord:
    """Convert one terminal job snapshot into an independently validated record."""
    status = snapshot.status
    error = snapshot.error
    if status == "failed" and not error:
        error = "Generation failed without a provider error"
    return QAAnswerRecord(
        record_id=snapshot.task_id,
        task_id=snapshot.task_id,
        action_type=snapshot.action_type,
        status=status,
        project_id=snapshot.project_id,
        file_id=snapshot.file_id,
        source_file=snapshot.source_file,
        page_number=snapshot.page_number,
        language=snapshot.language,
        question=snapshot.question,
        answer_markdown=snapshot.partial_text,
        context_bundle=snapshot.context_bundle,
        created_at=snapshot.created_at,
        completed_at=snapshot.completed_at or _now(),
        error=error,
    )


def current_response_from_snapshot(snapshot: QAJobSnapshot) -> QACurrentResponse:
    """Build the stable project output projection from one immutable job snapshot."""
    return QACurrentResponse(
        task_id=snapshot.task_id,
        project_id=snapshot.project_id,
        file_id=snapshot.file_id,
        source_file=snapshot.source_file,
        page_number=snapshot.page_number,
        action_type=snapshot.action_type,
        language=snapshot.language,
        question=snapshot.question,
        status=snapshot.status,
        answer_markdown=snapshot.partial_text,
        context_bundle=snapshot.context_bundle,
        quality_warning=snapshot.quality_warning,
        error=snapshot.error,
        created_at=snapshot.created_at,
        completed_at=snapshot.completed_at,
    )


def _failed_record_from_finalization_error(
    snapshot: QAJobSnapshot,
    exc: Exception,
) -> QAAnswerRecord:
    """Preserve a terminal task as a visible failure instead of locking the UI."""
    return QAAnswerRecord(
        record_id=snapshot.task_id,
        task_id=snapshot.task_id,
        action_type=snapshot.action_type,
        status="failed",
        project_id=snapshot.project_id,
        file_id=snapshot.file_id,
        source_file=snapshot.source_file,
        page_number=snapshot.page_number,
        language=snapshot.language,
        question=snapshot.question,
        answer_markdown=snapshot.partial_text,
        context_bundle=snapshot.context_bundle,
        created_at=snapshot.created_at,
        completed_at=snapshot.completed_at or _now(),
        error=f"FinalizationError: {type(exc).__name__}: {exc}",
    )


def upsert_answer_record(records: list[QAAnswerRecord], record: QAAnswerRecord) -> list[QAAnswerRecord]:
    """Idempotently submit a completed/cancelled/failed record by task id."""
    result = [item for item in records if item.task_id != record.task_id]
    result.append(record)
    result.sort(key=lambda item: item.created_at, reverse=True)
    return result


def project_answer_records(
    records: list[QAAnswerRecord],
    *,
    project_id: str,
    scope: AnswerHistoryScope,
    current_file_id: str,
    current_page_number: int,
) -> QAAnswerProjection:
    """Project immutable audit records into page/tool-oriented UI groups."""
    project_records = [record for record in records if record.project_id == project_id]
    if scope == "current_page":
        visible = [
            record for record in project_records
            if record.file_id == current_file_id
            and record.page_number == current_page_number
        ]
    elif scope == "current_file":
        visible = [
            record for record in project_records if record.file_id == current_file_id
        ]
    else:
        visible = list(project_records)

    by_page: dict[tuple[str, int], list[QAAnswerRecord]] = {}
    for record in visible:
        by_page.setdefault((record.file_id, record.page_number), []).append(record)

    groups: list[QAAnswerPageGroup] = []
    for (file_id, page_number), page_records in by_page.items():
        newest_first = sorted(page_records, key=lambda item: item.created_at, reverse=True)
        completed_explains = [
            item for item in newest_first
            if item.status == "completed" and item.action_type == "explain"
        ]
        completed_translations = [
            item for item in newest_first
            if item.status == "completed" and item.action_type == "translate"
        ]
        latest_translation_by_language: dict[str, QAAnswerRecord] = {}
        previous_versions = list(completed_explains[1:])
        for item in completed_translations:
            language_key = item.language.strip().casefold()
            if language_key not in latest_translation_by_language:
                latest_translation_by_language[language_key] = item
            else:
                previous_versions.append(item)

        groups.append(QAAnswerPageGroup(
            project_id=project_id,
            file_id=file_id,
            source_file=newest_first[0].source_file,
            page_number=page_number,
            latest_activity_at=newest_first[0].created_at,
            explain=completed_explains[0] if completed_explains else None,
            translations=sorted(
                latest_translation_by_language.values(),
                key=lambda item: (item.language.casefold(), item.created_at),
            ),
            asks=sorted(
                [
                    item for item in page_records
                    if item.status == "completed" and item.action_type == "ask"
                ],
                key=lambda item: item.created_at,
            ),
            previous_versions=sorted(
                previous_versions, key=lambda item: item.created_at, reverse=True,
            ),
            recent_tasks=[
                item for item in newest_first if item.status in {"failed", "cancelled"}
            ],
        ))
    if scope == "current_file":
        groups.sort(key=lambda item: item.page_number)
    else:
        groups.sort(key=lambda item: item.latest_activity_at, reverse=True)
    return QAAnswerProjection(
        scope=scope,
        groups=groups,
        project_record_count=len(project_records),
        visible_record_count=len(visible),
    )


def remove_answer_records(
    records: list[QAAnswerRecord],
    *,
    project_id: str,
    scope: AnswerHistoryScope,
    current_file_id: str,
) -> list[QAAnswerRecord]:
    """Delete only visible answer records; memory/current response are separate stores."""
    if scope == "whole_project":
        return [record for record in records if record.project_id != project_id]
    if scope == "current_file":
        return [
            record for record in records
            if not (record.project_id == project_id and record.file_id == current_file_id)
        ]
    raise ValueError("Bulk deletion supports current_file or whole_project only")


def should_enter_memory(record: QAAnswerRecord) -> bool:
    return record.status == "completed" and record.action_type in {"ask", "explain"}


def memory_turn_from_record(record: QAAnswerRecord) -> QAMemoryTurn | None:
    if not should_enter_memory(record) or not record.answer_markdown.strip():
        return None
    return QAMemoryTurn(
        question=record.question or f"{record.action_type.title()} page {record.page_number}",
        answer=record.answer_markdown,
        file_id=record.file_id,
        page_number=record.page_number,
        source_file=record.source_file,
        language=record.language,
    )


def append_memory_turn(
    memory: list[dict[str, Any] | QAMemoryTurn],
    record: QAAnswerRecord,
    *,
    max_rounds: int = 5,
) -> list[dict[str, Any]]:
    turn = memory_turn_from_record(record)
    normalized: list[dict[str, Any]] = []
    for item in memory:
        if isinstance(item, QAMemoryTurn):
            normalized.append(item.model_dump(mode="json"))
        elif isinstance(item, dict):
            normalized.append(item)
    if turn is not None:
        normalized.append(turn.model_dump(mode="json"))
    return normalized[-max(1, max_rounds):]


def finalize_qa_snapshot(
    snapshot: QAJobSnapshot,
    records: list[QAAnswerRecord],
    memory: list[dict[str, Any] | QAMemoryTurn],
    submitted_task_ids: set[str],
) -> QAFinalizationState:
    """Purely compute one idempotent terminal transition before UI mutation."""
    if snapshot.task_id in submitted_task_ids:
        return QAFinalizationState(
            record=None,
            records=list(records),
            memory=[
                item.model_dump(mode="json") if isinstance(item, QAMemoryTurn) else dict(item)
                for item in memory
            ],
            submitted_task_ids=set(submitted_task_ids),
            changed=False,
        )

    try:
        record = record_from_snapshot(snapshot)
    except Exception as exc:
        record = _failed_record_from_finalization_error(snapshot, exc)

    next_records = upsert_answer_record(records, record)
    next_memory = append_memory_turn(memory, record)
    next_submitted = set(submitted_task_ids)
    next_submitted.add(snapshot.task_id)
    return QAFinalizationState(
        record=record,
        records=next_records,
        memory=next_memory,
        submitted_task_ids=next_submitted,
        changed=True,
    )


def migrate_legacy_answer_state(state: dict[str, Any]) -> QAAnswerRecord | None:
    """Convert the old single-output slot once, without treating it as truth later."""
    answer = state.get("qa_last_answer")
    question = state.get("qa_last_question")
    bundle = state.get("qa_last_bundle")
    if not answer or not question or not isinstance(bundle, dict):
        return None
    source = next((source for source in bundle.get("sources", []) if isinstance(source, dict)), {})
    try:
        return QAAnswerRecord(
            record_id="legacy-qa-last",
            task_id="legacy-qa-last",
            action_type="ask",
            status="completed",
            project_id=str(state.get("selected_project_id") or "legacy-project"),
            file_id=str(state.get("selected_file_id") or "legacy-file"),
            source_file=str(source.get("source_file") or "legacy"),
            page_number=int(source.get("page_number") or state.get("selected_page_number") or 1),
            language=str(state.get("qa_target_language") or "中文"),
            question=str(question),
            answer_markdown=str(answer),
            created_at=_now(),
        )
    except Exception:
        return None

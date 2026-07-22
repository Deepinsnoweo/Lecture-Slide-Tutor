from __future__ import annotations

from slide_tutor.models import AgentRunSummary, QAAnswerRecord, QAContextBundle, SlidePage
from slide_tutor.qa_jobs import QAJobRequest, QAJobSnapshot
from slide_tutor.qa_records import (
    clear_project_answer_history,
    current_response_from_snapshot,
    delete_project_answer_record,
    load_project_answer_history,
    persist_answer_record,
    project_answer_records,
    remove_answer_records,
    should_enter_memory,
)


def _record(
    record_id: str,
    *,
    action: str,
    created_at: str,
    file_id: str = "file-a",
    page: int = 2,
    language: str = "中文",
    status: str = "completed",
    project_id: str = "project-a",
) -> QAAnswerRecord:
    return QAAnswerRecord(
        record_id=record_id,
        task_id=record_id,
        action_type=action,
        status=status,
        project_id=project_id,
        file_id=file_id,
        source_file=f"{file_id}.pdf",
        page_number=page,
        language=language,
        question=f"question-{record_id}",
        answer_markdown="answer" if status != "failed" else "",
        created_at=created_at,
        error="provider failed" if status == "failed" else None,
    )


def test_current_page_projection_uses_complete_page_identity():
    records = [
        _record("current", action="ask", created_at="2026-07-13T01:00:00Z"),
        _record("same-page-other-file", action="ask", created_at="2026-07-13T02:00:00Z", file_id="file-b"),
        _record("other-page", action="ask", created_at="2026-07-13T03:00:00Z", page=3),
        _record("other-project", action="ask", created_at="2026-07-13T04:00:00Z", project_id="project-b"),
    ]
    projection = project_answer_records(
        records, project_id="project-a", scope="current_page",
        current_file_id="file-a", current_page_number=2,
    )
    assert projection.visible_record_count == 1
    assert projection.project_record_count == 3
    assert [record.record_id for record in projection.groups[0].asks] == ["current"]


def test_projection_keeps_latest_tools_and_chronological_asks():
    records = [
        _record("explain-old", action="explain", created_at="2026-07-13T01:00:00Z"),
        _record("ask-new", action="ask", created_at="2026-07-13T06:00:00Z"),
        _record("translate-zh-old", action="translate", created_at="2026-07-13T02:00:00Z"),
        _record("ask-old", action="ask", created_at="2026-07-13T03:00:00Z"),
        _record("explain-new", action="explain", created_at="2026-07-13T05:00:00Z"),
        _record("translate-en", action="translate", created_at="2026-07-13T04:00:00Z", language="English"),
        _record("translate-zh-new", action="translate", created_at="2026-07-13T07:00:00Z"),
        _record("failed", action="ask", created_at="2026-07-13T08:00:00Z", status="failed"),
    ]
    group = project_answer_records(
        records, project_id="project-a", scope="current_page",
        current_file_id="file-a", current_page_number=2,
    ).groups[0]
    assert group.explain.record_id == "explain-new"
    assert {record.record_id for record in group.translations} == {
        "translate-en", "translate-zh-new",
    }
    assert [record.record_id for record in group.asks] == ["ask-old", "ask-new"]
    assert {record.record_id for record in group.previous_versions} == {
        "explain-old", "translate-zh-old",
    }
    assert [record.record_id for record in group.recent_tasks] == ["failed"]


def test_whole_project_groups_pages_by_latest_activity():
    records = [
        _record("page-2", action="ask", created_at="2026-07-13T01:00:00Z"),
        _record("page-3", action="ask", created_at="2026-07-13T02:00:00Z", page=3),
    ]
    projection = project_answer_records(
        records, project_id="project-a", scope="whole_project",
        current_file_id="file-a", current_page_number=2,
    )
    assert [group.page_number for group in projection.groups] == [3, 2]


def test_current_file_projection_excludes_other_pdf_and_keeps_all_pages():
    records = [
        _record("file-a-page-2", action="ask", created_at="2026-07-13T03:00:00Z"),
        _record("file-a-page-3", action="ask", created_at="2026-07-13T01:00:00Z", page=3),
        _record("file-a-page-1", action="ask", created_at="2026-07-13T02:00:00Z", page=1),
        _record(
            "file-b-page-2", action="ask", created_at="2026-07-13T04:00:00Z",
            file_id="file-b",
        ),
    ]
    projection = project_answer_records(
        records, project_id="project-a", scope="current_file",
        current_file_id="file-a", current_page_number=2,
    )
    assert projection.visible_record_count == 3
    assert [group.page_number for group in projection.groups] == [1, 2, 3]


def test_current_response_preserves_captured_language_and_source_identity():
    slide = SlidePage(
        slide_id="slide-2", module_id="project-a", project_id="project-a",
        file_id="file-a", source_file="lecture.pdf", page_number=2,
        global_index=2, source_type="pdf", text_quality="ok",
    )
    request = QAJobRequest(
        task_id="task-1", session_id="session-1", project_id="project-a",
        file_id="file-a", source_file="lecture.pdf", page_number=2,
        action_type="translate", language="日本語", current_slide=slide,
        file_page_count=10,
    )
    snapshot = QAJobSnapshot(
        task_id=request.task_id, session_id=request.session_id,
        project_id=request.project_id, file_id=request.file_id,
        source_file=request.source_file, page_number=request.page_number,
        action_type=request.action_type, language=request.language,
        question=request.question, status="completed", partial_text="翻訳",
        context_bundle=None, quality_warning=None, error=None,
        created_at="2026-07-13T00:00:00Z", started_at=None,
        completed_at="2026-07-13T00:00:01Z", cancel_requested=False,
    )

    response = current_response_from_snapshot(snapshot)

    assert response.project_id == "project-a"
    assert response.source_file == "lecture.pdf"
    assert response.page_number == 2
    assert response.language == "日本語"
    assert response.answer_markdown == "翻訳"


def test_agent_summary_round_trips_through_answer_record_and_cancelled_not_memory():
    summary = AgentRunSummary(
        mode="bounded_agent_v2",
        rounds=2,
        executed_tool_calls=1,
        baseline_source_count=2,
        stop_reason="finish_requested",
        usage_status="unavailable",
        elapsed_ms=2,
    )
    context = QAContextBundle(
        prompt_context="grounded",
        planner_mode="bounded_agent_v2",
        agent_summary=summary,
    )
    record = _record("agent", action="ask", created_at="2026-07-13T01:00:00Z")
    record = record.model_copy(update={"context_bundle": context})
    loaded = QAAnswerRecord.model_validate(record.model_dump(mode="json"))
    assert loaded.context_bundle is not None
    assert loaded.context_bundle.agent_summary is not None
    assert loaded.context_bundle.agent_summary.executed_tool_calls == 1

    cancelled = record.model_copy(update={
        "record_id": "cancelled-agent",
        "task_id": "cancelled-agent",
        "status": "cancelled",
        "answer_markdown": "",
    })
    assert should_enter_memory(cancelled) is False


def test_bulk_answer_removal_only_changes_requested_record_scope():
    records = [
        _record("file-a", action="ask", created_at="2026-07-13T01:00:00Z"),
        _record(
            "file-b", action="ask", created_at="2026-07-13T02:00:00Z",
            file_id="file-b",
        ),
        _record(
            "project-b", action="ask", created_at="2026-07-13T03:00:00Z",
            project_id="project-b",
        ),
    ]

    after_file = remove_answer_records(
        records, project_id="project-a", scope="current_file", current_file_id="file-a",
    )
    assert {record.record_id for record in after_file} == {"file-b", "project-b"}
    after_project = remove_answer_records(
        records, project_id="project-a", scope="whole_project", current_file_id="file-a",
    )
    assert [record.record_id for record in after_project] == ["project-b"]


def test_project_history_persists_sanitized_records_without_memory(
    isolated_data_root,
):
    context = QAContextBundle(
        prompt_context="full private assembled prompt",
        memory_turns=[],
        planner_mode="deterministic",
    )
    record = _record(
        "persisted", action="ask", created_at="2026-07-13T01:00:00Z",
    ).model_copy(update={"context_bundle": context})

    persist_answer_record(record)
    persist_answer_record(record)
    loaded = load_project_answer_history("project-a")

    assert [item.record_id for item in loaded] == ["persisted"]
    assert loaded[0].answer_markdown == "answer"
    assert loaded[0].context_bundle is not None
    assert loaded[0].context_bundle.prompt_context == ""
    assert loaded[0].context_bundle.memory_turns == []


def test_project_history_isolated_delete_and_clear(isolated_data_root):
    first = _record("a-1", action="ask", created_at="2026-07-13T01:00:00Z")
    second = _record("a-2", action="explain", created_at="2026-07-13T02:00:00Z")
    other = _record(
        "b-1", action="ask", created_at="2026-07-13T03:00:00Z",
        project_id="project-b",
    )
    for record in (first, second, other):
        persist_answer_record(record)

    remaining = delete_project_answer_record("project-a", "a-1")
    assert [record.record_id for record in remaining] == ["a-2"]
    assert [record.record_id for record in load_project_answer_history("project-b")] == ["b-1"]
    clear_project_answer_history("project-a")
    assert load_project_answer_history("project-a") == []
    assert [record.record_id for record in load_project_answer_history("project-b")] == ["b-1"]

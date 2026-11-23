from __future__ import annotations

from slide_tutor.ui import (
    _answer_record_widget_key,
    _apply_page_jump,
    _agent_summary_text,
    _ensure_preview_preference_state,
    _ensure_qa_language_state,
    _find_cited_file,
    _generation_lifecycle_label,
    _has_history_or_memory,
    _page_jump_key,
    _page_browser_window_key,
    _page_browser_window,
    _page_browser_window_start,
    _persist_preview_preference,
    _persist_qa_language_state,
    _pdf_reorder_options,
    _preview_image_data_uri,
    _preview_preference_key,
    _preview_state_key,
    _quiz_generation_quality_notice,
    _select_pdf,
    _set_selected_page,
)
from slide_tutor.models import AgentRunSummary


def test_preview_data_uri_allows_only_expected_image_mime(tmp_path):
    png = tmp_path / "page.png"
    png.write_bytes(b"\x89PNG\r\n")
    uri = _preview_image_data_uri(png)
    assert uri is not None
    assert uri.startswith("data:image/png;base64,")
    assert str(png) not in uri

    gif = tmp_path / "page.gif"
    gif.write_bytes(b"GIF89a")
    assert _preview_image_data_uri(gif) is None


def test_preview_state_is_scoped_to_project_and_pdf_not_page():
    assert _preview_state_key("project-a", "file-a") == _preview_state_key(
        "project-a", "file-a",
    )
    assert _preview_state_key("project-a", "file-a") != _preview_state_key(
        "project-a", "file-b",
    )
    assert _preview_state_key("project-a", "file-a") != _preview_state_key(
        "project-b", "file-a",
    )


def test_preview_preference_survives_widget_cleanup_and_remains_pdf_scoped():
    state: dict = {}
    widget_key, preference_key = _ensure_preview_preference_state(
        state, "project-a", "file-a",
    )
    assert state[widget_key] is False
    assert state[preference_key] is False

    state[widget_key] = True
    _persist_preview_preference(state, "project-a", "file-a", widget_key)
    state.pop(widget_key)

    restored_widget, restored_preference = _ensure_preview_preference_state(
        state, "project-a", "file-a",
    )
    assert state[restored_widget] is True
    assert state[restored_preference] is True
    assert _preview_preference_key("project-a", "file-b") not in state


def test_page_navigation_callbacks_commit_canonical_and_jump_state():
    state: dict = {}
    jump_key = _page_jump_key("file-a")

    _set_selected_page(state, 3, 5, jump_key)
    assert state["selected_page_number"] == 3
    assert state[jump_key] == 3

    _set_selected_page(state, 99, 5, jump_key)
    assert state["selected_page_number"] == 5
    assert state[jump_key] == 5

    state[jump_key] = 2
    _apply_page_jump(state, jump_key, 5)
    assert state["selected_page_number"] == 2


def test_pdf_selection_resets_page_only_when_file_changes():
    state = {
        "selected_file_id": "file-a",
        "selected_page_number": 12,
        "qa_file_clear_confirmation": ("project-a", "file-a", frozenset({"answer-1"})),
    }

    _select_pdf(state, "file-a")
    assert state["selected_page_number"] == 12
    assert "qa_file_clear_confirmation" in state

    _select_pdf(state, "file-b")
    assert state["selected_file_id"] == "file-b"
    assert state["selected_page_number"] == 1
    assert "qa_file_clear_confirmation" not in state


def test_citation_resolves_by_file_identity_and_rejects_ambiguous_legacy_names():
    from types import SimpleNamespace

    first = SimpleNamespace(file_id="first", original_name="lecture.pdf")
    second = SimpleNamespace(file_id="second", original_name="lecture.pdf")
    other = SimpleNamespace(file_id="other", original_name="other.pdf")
    project = SimpleNamespace(project_id="project-a", files=[first, second, other])

    assert _find_cited_file(project, "lecture.pdf", file_id="second") is second
    assert _find_cited_file(project, "lecture.pdf", file_id="deleted") is None
    assert _find_cited_file(project, "lecture.pdf", file_id="second", project_id="project-b") is None
    assert _find_cited_file(project, "lecture.pdf") is None
    assert _find_cited_file(project, "lecture") is None
    assert _find_cited_file(project, " other.PDF ") is other


def test_pdf_reorder_options_are_unique_and_do_not_expose_hashes():
    class File:
        def __init__(self, file_id: str, name: str, pages: int, sha256: str):
            self.file_id = file_id
            self.original_name = name
            self.page_count = pages
            self.sha256 = sha256

    files = [
        File("file-a", "lecture.pdf", 10, "secret-a"),
        File("file-b", "lecture.pdf", 20, "secret-b"),
    ]
    labels, ids_by_label = _pdf_reorder_options(files)

    assert labels == [
        "1. lecture.pdf · 10 pages",
        "2. lecture.pdf · 20 pages",
    ]
    assert [ids_by_label[label] for label in labels] == ["file-a", "file-b"]
    assert all("secret" not in label for label in labels)


def test_page_browser_window_is_bounded_and_handles_last_group():
    slides = list(range(1, 54))

    assert _page_browser_window_start(1, 53) == 1
    assert _page_browser_window_start(13, 53) == 13
    assert _page_browser_window_start(53, 53) == 49
    assert _page_browser_window(slides, 1) == list(range(1, 13))
    assert _page_browser_window(slides, 49) == list(range(49, 54))
    assert len(_page_browser_window(slides, 13)) <= 12


def test_page_browser_state_is_scoped_to_project_and_pdf():
    project_a_file_a = _page_browser_window_key("project-a", "file-a")
    assert project_a_file_a == _page_browser_window_key("project-a", "file-a")
    assert project_a_file_a != _page_browser_window_key("project-a", "file-b")
    assert project_a_file_a != _page_browser_window_key("project-b", "file-a")


def test_qa_language_survives_widget_cleanup_and_remains_project_scoped():
    state: dict = {}
    project_a_key = _ensure_qa_language_state(state, "project-a")
    assert state[project_a_key] == "简体中文"

    state[project_a_key] = "English"
    _persist_qa_language_state(state, "project-a", project_a_key)
    state.pop(project_a_key)

    assert _ensure_qa_language_state(state, "project-a") == project_a_key
    assert state[project_a_key] == "English"

    project_b_key = _ensure_qa_language_state(state, "project-b")
    assert state[project_b_key] == "简体中文"


def test_qa_language_migrates_legacy_chinese_label():
    state = {"qa_target_languages": {"project-a": "中文"}}
    key = _ensure_qa_language_state(state, "project-a")

    assert state[key] == "简体中文"
    assert state["qa_target_languages"]["project-a"] == "简体中文"


def test_history_and_memory_section_is_conditional():
    class Record:
        def __init__(self, project_id: str):
            self.project_id = project_id

    assert not _has_history_or_memory([], "project-a", [])
    assert _has_history_or_memory([], "project-a", [{"question": "Q"}])
    assert _has_history_or_memory([Record("project-a")], "project-a", [])
    assert not _has_history_or_memory([Record("project-b")], "project-a", [])


def test_answer_history_widget_keys_are_namespaced_by_render_surface():
    record_id = "task-shared-record"

    local_key = _answer_record_widget_key("answer-history", record_id)
    project_key = _answer_record_widget_key("project-history", record_id)

    assert local_key != project_key
    assert local_key.endswith(record_id)
    assert project_key.endswith(record_id)
    assert local_key == _answer_record_widget_key("answer-history", record_id)


def test_quiz_surfaces_warn_when_only_baseline_is_active():
    notices = [
        _quiz_generation_quality_notice("baseline_current"),
        _quiz_generation_quality_notice("enhancing"),
        _quiz_generation_quality_notice("enhancement_failed"),
    ]
    assert all(notice for notice in notices)
    assert all(not any("\u4e00" <= char <= "\u9fff" for char in notice) for notice in notices)
    assert _quiz_generation_quality_notice("enhanced_current") is None


def test_generation_lifecycle_labels_are_english_only():
    statuses = (
        "baseline_building",
        "baseline_current",
        "enhancing",
        "enhanced_current",
        "enhancement_failed",
        "stale",
        "unknown",
    )
    labels = [_generation_lifecycle_label(status) for status in statuses]

    assert labels[-1] == "Generation status unavailable"
    assert all(not any("\u4e00" <= char <= "\u9fff" for char in label) for label in labels)


def test_agent_summary_text_exposes_only_sanitized_run_facts():
    exact = AgentRunSummary(
        mode="bounded_agent_v2",
        rounds=2,
        executed_tool_calls=1,
        baseline_source_count=2,
        stop_reason="finish_requested",
        usage_status="exact",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        elapsed_ms=2,
    )
    fallback = exact.model_copy(update={
        "mode": "bounded_agent_v2_fallback",
        "usage_status": "unavailable",
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    })
    assert "Agent v2 · rounds=2 · tools=1" in _agent_summary_text(exact)
    assert "tokens=15" in _agent_summary_text(exact)
    assert "Agent v2 → v1" in _agent_summary_text(fallback)
    assert "reasoning" not in _agent_summary_text(exact).lower()

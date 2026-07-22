from slide_tutor import ui
from slide_tutor.models import ProjectFile, SlidePage


def _project_file() -> ProjectFile:
    return ProjectFile(
        file_id="file-1",
        original_name="Lecture01.pdf",
        safe_name="Lecture01.pdf",
        sha256="abc123",
        size_bytes=123,
        page_count=10,
        created_at="2026-07-09T00:00:00Z",
        status="ready",
    )


def _slide(page_number: int = 5) -> SlidePage:
    return SlidePage(
        slide_id=f"slide-{page_number}",
        module_id="project-1",
        project_id="project-1",
        file_id="file-1",
        source_file="Lecture01.pdf",
        source_type="pdf",
        page_number=page_number,
        global_index=page_number,
        text_quality="ok",
    )


def test_memory_turn_preserves_text_beyond_legacy_preview_limits():
    question = "Q" * 250
    answer = ("A" * 350) + "TAIL"

    turn = ui._build_memory_turn(question, answer, _project_file(), _slide(), "中文")

    assert turn["question"] == question
    assert turn["answer"] == answer
    assert turn["page_number"] == 5
    assert turn["source_file"] == "Lecture01.pdf"


def test_memory_turn_still_caps_accidental_huge_answers():
    answer = "A" * (ui._MEMORY_STORED_ANSWER_CHAR_LIMIT + 100)

    turn = ui._build_memory_turn("What is this?", answer, _project_file(), _slide(), "中文")

    assert len(turn["answer"]) == ui._MEMORY_STORED_ANSWER_CHAR_LIMIT
    assert turn["answer"].endswith("...")


def test_prompt_memory_is_bounded_without_mutating_stored_answer():
    answer = ("A" * 500) + "TAIL"
    memory = [ui._build_memory_turn("What is MPG?", answer, _project_file(), _slide(), "中文")]

    prompt_block = ui._format_memory_for_prompt(memory)

    assert "Recent conversation memory" in prompt_block
    assert "TAIL" not in prompt_block
    assert len(prompt_block) < ui._MEMORY_PROMPT_CHAR_BUDGET + 200
    assert memory[0]["answer"].endswith("TAIL")


def test_clear_memory_is_visible_even_after_answer_is_cleared():
    memory = [ui._build_memory_turn("Q", "A", _project_file(), _slide(), "中文")]

    actions = ui._qa_action_visibility(answer=None, memory=memory)

    assert actions["clear_memory"] is True
    assert actions["clear_answer"] is False


def test_clear_answer_also_clears_its_context_bundle():
    state = {
        "qa_last_answer": "Answer",
        "qa_last_question": "Question",
        "qa_last_bundle": {"sources": ["old context"]},
        "qa_memory": [{"question": "Keep this memory"}],
    }

    ui._clear_qa_output_state(state)

    assert "qa_last_answer" not in state
    assert "qa_last_question" not in state
    assert "qa_last_bundle" not in state
    assert state["qa_memory"] == [{"question": "Keep this memory"}]

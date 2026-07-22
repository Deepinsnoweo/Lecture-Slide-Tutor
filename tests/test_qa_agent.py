from __future__ import annotations

from slide_tutor import projects, qa_context, retrieval
from slide_tutor.agent_policy import AgentExecutionPolicy
from slide_tutor.models import QAContextBundle, RetrievalChunk, SlidePage, ToolCall
from slide_tutor.qa_agent import (
    build_agent_controller_messages,
    create_agent_accumulator,
    execute_qa_selection_calls,
)


def _slide(page: int, text: str | None = None) -> SlidePage:
    return SlidePage(
        slide_id=f"slide-{page}",
        module_id="project-1",
        project_id="project-1",
        file_id="file-1",
        source_file="Lecture01.pdf",
        source_type="pdf",
        page_number=page,
        global_index=page,
        text=text or f"Page {page} text",
        text_quality="ok",
    )


def _chunk(page: int, text: str) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=f"chunk-{page}",
        project_id="project-1",
        file_id="file-1",
        slide_id=f"slide-{page}",
        source_file="Lecture01.pdf",
        page_number=page,
        text=text,
        snippet_reference=text[:80],
    )


def _patch_context(monkeypatch, slides, retrieved=None, seen=None):
    monkeypatch.setattr(projects, "load_file_slides", lambda *args: slides)
    monkeypatch.setattr(
        qa_context,
        "_slide_text_for_context",
        lambda _project, _file, slide: slide.text,
    )
    monkeypatch.setattr(
        retrieval,
        "retrieve_for_question",
        lambda *args, **kwargs: (
            seen.append(kwargs.get("lexical_query")) if seen is not None else None
        ) or (retrieved or []),
    )


def _accumulator(
    monkeypatch,
    *,
    question="Explain this page",
    retrieved=None,
    seen=None,
    long_page: int | None = None,
):
    slides = [
        _slide(
            page,
            text=(f"Page {page} text " + "x" * 500)
            if page == long_page else None,
        )
        for page in range(1, 21)
    ]
    _patch_context(monkeypatch, slides, retrieved=retrieved, seen=seen)
    return create_agent_accumulator(
        question,
        slides[4],
        20,
        [{
            "question": "previous",
            "answer": "memory",
            "file_id": "file-1",
            "source_file": "Lecture01.pdf",
            "page_number": 4,
            "language": "中文",
        }],
        ([], object(), object()),
        AgentExecutionPolicy(
            max_observation_chars_per_call=200,
            max_total_observation_chars=500,
        ),
    )


def test_controller_prompt_teaches_short_query_and_finish_preferences():
    messages = build_agent_controller_messages(
        "How does the model classify regression?",
        QAContextBundle(prompt_context="direct evidence"),
    )
    prompt = "\n".join(message.content for message in messages)
    assert "3-10 high-signal terms" in prompt
    assert "world model 两种 世界モデル" in prompt
    assert "express listener reactions assessment backchannels" in prompt
    assert "regression classification target output" in prompt
    assert "no-new-context" in prompt
    assert "confirmation round" in prompt


def test_baseline_includes_current_positional_and_memory_before_tools(monkeypatch):
    accumulator = _accumulator(
        monkeypatch,
        question="比较第17页和下一页",
    )

    source_pages = [source.page_number for source in accumulator.bundle.sources]
    assert source_pages[:3] == [5, 17, 6]
    assert any(source.kind == "memory" for source in accumulator.bundle.sources)
    assert accumulator.baseline_source_count >= 4
    assert accumulator.executed_call_count == 0


def test_page_observation_contains_real_text_and_updates_final_bundle(monkeypatch):
    accumulator = _accumulator(monkeypatch)
    observations, traces = execute_qa_selection_calls(
        [ToolCall(
            call_id="call-1",
            name="load_pages",
            arguments={"page_numbers": [7]},
        )],
        accumulator=accumulator,
    )

    assert traces[0].status == "executed"
    assert observations[0].status == "executed"
    assert "Page 7 text" in observations[0].content_for_model
    assert observations[0].sources[0].page_number == 7
    assert any(source.page_number == 7 for source in accumulator.bundle.sources)
    assert accumulator.executed_call_count == 1


def test_neighbor_observation_uses_relative_page_and_search_preserves_original_query(
    monkeypatch,
):
    seen_queries = []
    accumulator = _accumulator(
        monkeypatch,
        question="What is the relationship?",
        retrieved=[_chunk(9, "retrieved concept")],
        seen=seen_queries,
    )

    neighbor, _ = execute_qa_selection_calls(
        [ToolCall(
            call_id="call-1",
            name="load_neighbor_pages",
            arguments={"offsets": [1]},
        )],
        accumulator=accumulator,
    )
    search, _ = execute_qa_selection_calls(
        [ToolCall(
            call_id="call-2",
            name="search_slides",
            arguments={"query": "concept X"},
        )],
        accumulator=accumulator,
    )

    assert neighbor[0].sources[0].page_number == 6
    assert search[0].sources[0].page_number == 9
    assert seen_queries == ["What is the relationship?"]


def test_duplicate_calls_receive_observations_without_duplicate_sources(monkeypatch):
    accumulator = _accumulator(monkeypatch)
    calls = [
        ToolCall(call_id="call-1", name="load_pages", arguments={"page_numbers": [7]}),
        ToolCall(call_id="call-2", name="load_pages", arguments={"page_numbers": [7]}),
    ]
    observations, traces = execute_qa_selection_calls(calls, accumulator=accumulator)

    assert [trace.status for trace in traces] == ["executed", "deduplicated"]
    assert [observation.status for observation in observations] == [
        "executed", "deduplicated",
    ]
    page7_sources = [
        source for source in accumulator.bundle.sources if source.page_number == 7
    ]
    assert len(page7_sources) == 1


def test_duplicate_across_rounds_is_deduplicated(monkeypatch):
    accumulator = _accumulator(monkeypatch)
    first, _ = execute_qa_selection_calls(
        [ToolCall(call_id="call-1", name="load_pages", arguments={"page_numbers": [7]})],
        accumulator=accumulator,
    )
    second, traces = execute_qa_selection_calls(
        [ToolCall(call_id="call-2", name="load_pages", arguments={"page_numbers": [7]})],
        accumulator=accumulator,
    )

    assert first[0].status == "executed"
    assert traces[0].status == "deduplicated"
    assert second[0].status == "deduplicated"
    assert accumulator.executed_call_count == 1


def test_observation_is_truncated_without_changing_final_bundle(monkeypatch):
    accumulator = _accumulator(monkeypatch, long_page=7)
    observation, _ = execute_qa_selection_calls(
        [ToolCall(call_id="call-1", name="load_pages", arguments={"page_numbers": [7]})],
        accumulator=accumulator,
    )

    assert len(observation[0].content_for_model) <= 200
    assert "tool=load_pages" in observation[0].content_for_model
    assert "new_sources=1" in observation[0].content_for_model
    assert "source=Lecture01.pdf, page 7" in observation[0].content_for_model
    assert "Page 7 text" in accumulator.bundle.prompt_context


def test_total_observation_budget_is_separate_from_final_bundle_budget(monkeypatch):
    accumulator = _accumulator(monkeypatch, long_page=7)
    accumulator.policy = AgentExecutionPolicy(
        max_observation_chars_per_call=500,
        max_total_observation_chars=500,
    )
    first, _ = execute_qa_selection_calls(
        [ToolCall(call_id="call-1", name="load_pages", arguments={"page_numbers": [7]})],
        accumulator=accumulator,
    )
    second, _ = execute_qa_selection_calls(
        [ToolCall(call_id="call-2", name="load_pages", arguments={"page_numbers": [8]})],
        accumulator=accumulator,
    )

    assert len(first[0].content_for_model) <= 500
    assert "loaded_prompt_chars=" in first[0].content_for_model
    assert len(first[0].content_for_model) + len(second[0].content_for_model) <= 500
    assert "source=Lecture01.pdf, page 8" in second[0].content_for_model
    assert "Page 8 text" in accumulator.bundle.prompt_context

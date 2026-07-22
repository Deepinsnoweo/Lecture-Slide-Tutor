from slide_tutor import config, projects, qa_context, retrieval
from slide_tutor.models import (
    QAContextPlan,
    QAMemoryTurn,
    RetrievalChunk,
    SlidePage,
    ToolCall,
)


def _slide(page_number: int = 5, text: str = "Current page text") -> SlidePage:
    return SlidePage(
        slide_id=f"slide-{page_number}",
        module_id="project-1",
        project_id="project-1",
        file_id="file-1",
        source_file="Lecture01.pdf",
        source_type="pdf",
        page_number=page_number,
        global_index=page_number,
        text=text,
        text_quality="ok",
    )


def _chunk(page_number: int, text: str) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=f"chunk-{page_number}",
        project_id="project-1",
        file_id="file-1",
        slide_id=f"slide-{page_number}",
        source_file="Lecture01.pdf",
        page_number=page_number,
        text=text,
        snippet_reference=text[:150],
    )


def _patch_bundle_dependencies(monkeypatch, slides, retrieved):
    monkeypatch.setattr(projects, "load_file_slides", lambda *args: slides)
    monkeypatch.setattr(
        qa_context,
        "_slide_text_for_context",
        lambda project_id, file_id, slide: slide.text,
    )
    monkeypatch.setattr(retrieval, "retrieve_for_question", lambda *args, **kwargs: retrieved)


def _build_bundle(plan: QAContextPlan, current_slide: SlidePage, memory, index=None):
    return qa_context.build_qa_context_bundle(
        plan,
        "project-1",
        current_slide,
        "file-1",
        20,
        memory,
        index or ([], object(), object()),
    )


def test_planner_uses_lightweight_model_and_sanitizes_tool_args(monkeypatch):
    calls = {}
    monkeypatch.setattr(config, "QA_PLANNER_MODEL", "planner-light")

    def fake_complete_json(system_prompt, user_prompt, **kwargs):
        calls["kwargs"] = kwargs
        calls["user_prompt"] = user_prompt
        return QAContextPlan(
            include_current_page=True,
            page_numbers=[17, 0, 999, 17],
            page_offsets=[-1, 0, 2, 99, -99, -1],
            semantic_query="relationship between current page and page 17",
            use_memory=True,
        )

    monkeypatch.setattr(qa_context, "complete_json", fake_complete_json)

    plan = qa_context.plan_qa_context(
        "这页和17页有什么联系？",
        _slide(page_number=5),
        max_page=20,
        memory=[{"question": "mpg是什么？"}],
    )

    assert calls["kwargs"]["model"] == "planner-light"
    assert calls["kwargs"]["temperature"] == 0.0
    assert calls["kwargs"]["max_retries"] == 1
    assert calls["kwargs"]["pydantic_model"] is QAContextPlan
    assert "mpg是什么" in calls["user_prompt"]
    assert plan.page_numbers == [17]
    assert plan.page_offsets == [-1, 2]


def test_planner_fallback_survives_llm_failure_and_filters_impossible_neighbors(monkeypatch):
    def fail_complete_json(*args, **kwargs):
        raise RuntimeError("planner unavailable")

    monkeypatch.setattr(qa_context, "complete_json", fail_complete_json)

    plan = qa_context.plan_qa_context(
        "这页和上一页什么区别？",
        _slide(page_number=1),
        max_page=1,
        memory=[{"bad": "old shape"}, "not a dict"],
    )

    assert plan.include_current_page is True
    assert plan.page_numbers == []
    assert plan.page_offsets == []
    assert plan.semantic_query == "这页和上一页什么区别？"
    assert plan.use_memory is True


def test_planner_accepts_dict_response_and_handles_missing_memory_question(monkeypatch):
    captured = {}

    def fake_complete_json(system_prompt, user_prompt, **kwargs):
        captured["user_prompt"] = user_prompt
        return {
            "include_current_page": True,
            "page_numbers": [2],
            "page_offsets": [1],
            "semantic_query": "",
            "use_memory": False,
            "ambiguity_note": None,
        }

    monkeypatch.setattr(qa_context, "complete_json", fake_complete_json)

    plan = qa_context.plan_qa_context(
        "下一页继续讲什么？",
        _slide(page_number=1),
        max_page=3,
        memory=[{}],
    )

    assert "(missing question)" in captured["user_prompt"]
    assert plan.page_numbers == [2]
    assert plan.page_offsets == [1]
    assert plan.use_memory is False


def test_function_calling_planner_returns_plan_and_tool_trace(monkeypatch):
    monkeypatch.setattr(config, "QA_TOOL_CALLING_ENABLED", True)
    captured = {}

    def fake_complete_tool_calls(system_prompt, user_prompt, tools, **kwargs):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        captured["tools"] = tools
        captured["kwargs"] = kwargs
        return [
            ToolCall(
                call_id="call-1",
                name="load_pages",
                arguments={"page_numbers": [17]},
            ),
            ToolCall(
                call_id="call-2",
                name="load_neighbor_pages",
                arguments={"offsets": [-1]},
            ),
            ToolCall(
                call_id="call-3",
                name="search_slides",
                arguments={"query": "LTD mechanism"},
            ),
        ]

    monkeypatch.setattr(qa_context, "complete_tool_calls", fake_complete_tool_calls)

    decision = qa_context.plan_qa_context_with_tools(
        "这页和17页有什么联系？",
        _slide(page_number=5),
        max_page=20,
        memory=[],
    )

    assert decision.mode == "function_calling"
    assert decision.plan.page_numbers == [17]
    assert decision.plan.page_offsets == [-1]
    assert decision.plan.semantic_query == "LTD mechanism"
    assert [trace.status for trace in decision.tool_traces] == [
        "executed", "executed", "executed",
    ]
    assert [tool["function"]["name"] for tool in captured["tools"]] == [
        "load_pages", "load_neighbor_pages", "search_slides",
    ]
    assert captured["kwargs"]["temperature"] == 0.0
    assert "tool calls" in captured["system_prompt"]
    assert "Output ONLY valid JSON" not in captured["user_prompt"]


def test_function_calling_planner_falls_back_to_legacy_json_planner(monkeypatch):
    monkeypatch.setattr(config, "QA_TOOL_CALLING_ENABLED", True)
    monkeypatch.setattr(
        qa_context,
        "complete_tool_calls",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("tools unavailable")),
    )
    legacy_plan = QAContextPlan(
        include_current_page=True,
        page_offsets=[-1],
        semantic_query="legacy question",
    )
    monkeypatch.setattr(qa_context, "plan_qa_context", lambda *args, **kwargs: legacy_plan)

    decision = qa_context.plan_qa_context_with_tools(
        "question", _slide(page_number=5), max_page=20, memory=[],
    )

    assert decision.mode == "function_calling_fallback"
    assert decision.plan is legacy_plan
    assert decision.tool_traces == []


def test_function_calling_planner_respects_disabled_flag(monkeypatch):
    monkeypatch.setattr(config, "QA_TOOL_CALLING_ENABLED", False)
    legacy_plan = QAContextPlan(semantic_query="legacy")
    monkeypatch.setattr(qa_context, "plan_qa_context", lambda *args, **kwargs: legacy_plan)
    monkeypatch.setattr(
        qa_context,
        "complete_tool_calls",
        lambda *args, **kwargs: pytest.fail("disabled planner must not call tools"),
    )

    decision = qa_context.plan_qa_context_with_tools(
        "question", _slide(page_number=5), max_page=20, memory=[],
    )

    assert decision.mode == "json_planner"
    assert decision.plan is legacy_plan


def test_function_calling_marks_additional_search_trace_as_deduplicated(monkeypatch):
    monkeypatch.setattr(config, "QA_TOOL_CALLING_ENABLED", True)
    monkeypatch.setattr(
        qa_context,
        "complete_tool_calls",
        lambda *args, **kwargs: [
            ToolCall(call_id="call-1", name="search_slides", arguments={"query": "first"}),
            ToolCall(call_id="call-2", name="search_slides", arguments={"query": "second"}),
        ],
    )

    decision = qa_context.plan_qa_context_with_tools(
        "search", _slide(page_number=5), max_page=20, memory=[],
    )

    assert decision.plan.semantic_query == "first"
    assert [trace.status for trace in decision.tool_traces] == [
        "executed", "deduplicated",
    ]


def test_bundle_dedupes_loaded_pages_before_semantic_search(monkeypatch):
    current = _slide(5, "Current page content")
    related = _slide(7, "Related page content")
    _patch_bundle_dependencies(
        monkeypatch,
        [current, related],
        [_chunk(5, current.text), _chunk(7, related.text)],
    )

    bundle = _build_bundle(
        QAContextPlan(include_current_page=True, semantic_query="related", use_memory=False),
        current,
        [],
    )

    assert [source.page_number for source in bundle.sources] == [5, 7]
    assert [source.kind for source in bundle.sources] == ["selected_page", "retrieved_page"]
    assert "Related pages (semantic search)" in bundle.prompt_context


def test_bundle_preserves_retrieval_methods_for_context_transparency(monkeypatch):
    current = _slide(5, "Current page content")
    related = _chunk(50, "Long-Term Depression")
    related = related.model_copy(update={
        "retrieval_methods": ["exact_title", "bm25", "dense"],
    })
    _patch_bundle_dependencies(monkeypatch, [current], [related])

    bundle = qa_context.build_qa_context_bundle(
        QAContextPlan(include_current_page=False, semantic_query="synaptic plasticity"),
        "project-1", current, "file-1", 60, [],
        ([], object(), object()),
        original_question="Long-Term Depression 在哪一页？",
    )

    source = bundle.sources[0]
    assert source.retrieval_methods == ["exact_title", "bm25", "dense"]
    assert "exact_title + bm25 + dense match" in source.reason


def test_bundle_reports_only_semantic_chunks_that_enter_prompt(monkeypatch):
    current = _slide(5, "Current page content")
    long_chunk = _chunk(7, "A" * 5000)
    skipped_chunk = _chunk(8, "Short related page")
    _patch_bundle_dependencies(monkeypatch, [current], [long_chunk, skipped_chunk])

    bundle = _build_bundle(
        QAContextPlan(include_current_page=False, semantic_query="topic", use_memory=False),
        current,
        [],
    )

    assert "Lecture01.pdf, page 7" in bundle.prompt_context
    assert "Lecture01.pdf, page 8" not in bundle.prompt_context
    assert [(source.kind, source.page_number) for source in bundle.sources] == [
        ("retrieved_page", 7),
    ]


def test_bundle_keeps_all_explicit_pages_and_dedupes_nearby_pages(monkeypatch):
    slides = [_slide(page, f"Page {page} content") for page in range(4, 9)]
    current = next(slide for slide in slides if slide.page_number == 5)
    _patch_bundle_dependencies(monkeypatch, slides, [])

    bundle = _build_bundle(
        QAContextPlan(
            include_current_page=True,
            page_numbers=[4, 6, 7, 8],
            page_offsets=[-1, 1],
            use_memory=False,
        ),
        current,
        [],
    )

    assert [source.page_number for source in bundle.sources] == [5, 4, 6, 7, 8]
    assert [source.kind for source in bundle.sources] == [
        "selected_page", "explicit_page", "explicit_page", "explicit_page", "explicit_page",
    ]
    for page_number in (4, 6, 7, 8):
        assert f"Page {page_number} (Lecture01.pdf)" in bundle.prompt_context


def test_bundle_skips_invalid_memory_and_reports_only_included_turns(monkeypatch):
    current = _slide(5, "Current page content")
    _patch_bundle_dependencies(monkeypatch, [current], [])
    memory = [
        {
            "question": "What is MPG?",
            "answer": "Miles per gallon.",
            "file_id": "file-1",
            "page_number": 5,
            "source_file": "Lecture01.pdf",
            "language": "中文",
        },
        {"bad": "old shape"},
        "not a memory turn",
        {
            "question": "Why does it matter?",
            "answer": "It measures fuel efficiency.",
            "file_id": "file-1",
            "page_number": 6,
            "source_file": "Lecture01.pdf",
            "language": "中文",
        },
    ]

    bundle = _build_bundle(
        QAContextPlan(include_current_page=False, use_memory=True),
        current,
        memory,
    )

    assert all(isinstance(turn, QAMemoryTurn) for turn in bundle.memory_turns)
    assert [turn.question for turn in bundle.memory_turns] == [
        "What is MPG?", "Why does it matter?",
    ]
    assert [source.kind for source in bundle.sources] == ["memory", "memory"]
    assert "What is MPG?" in bundle.prompt_context
    assert "old shape" not in bundle.prompt_context


def test_mpg_follow_up_uses_memory_context_without_planner(monkeypatch):
    current = _slide(15, "MPG applications")
    _patch_bundle_dependencies(monkeypatch, [current], [])
    memory = [{
        "question": "mpg 是什么？",
        "answer": "MPG means miles per gallon.",
        "file_id": "file-1",
        "page_number": 14,
        "source_file": "Lecture01.pdf",
        "language": "中文",
    }]
    bundle = _build_bundle(
        QAContextPlan(include_current_page=True, use_memory=True),
        current,
        memory,
    )
    assert bundle.memory_turns[0].question == "mpg 是什么？"
    assert "MPG means miles per gallon" in bundle.prompt_context
    assert any(source.kind == "memory" for source in bundle.sources)

from __future__ import annotations

from types import SimpleNamespace

from slide_tutor import qa_pipelines
from slide_tutor.models import (
    AgentRunSummary,
    QAContextBundle,
    QAContextPlanningResult,
    QAContextSource,
    QAContextPlan,
    QATaskStatus,
    SlidePage,
    ToolTrace,
)
from slide_tutor.agent_runtime import AgentRuntimeResult
from slide_tutor.qa_jobs import QAJobRequest


def _slide() -> SlidePage:
    return SlidePage(
        slide_id="slide-1",
        module_id="project-1",
        project_id="project-1",
        file_id="file-1",
        source_file="Lecture01.pdf",
        source_type="pdf",
        page_number=3,
        global_index=3,
        text="Current page",
        text_quality="ok",
    )


def _request() -> QAJobRequest:
    return QAJobRequest(
        session_id="session-1",
        project_id="project-1",
        file_id="file-1",
        source_file="Lecture01.pdf",
        page_number=3,
        action_type="ask",
        language="简体中文",
        question="What is this?",
        current_slide=_slide(),
        file_page_count=10,
    )


def test_ask_pipeline_persists_function_calling_provenance(monkeypatch):
    planning = QAContextPlanningResult(
        plan=QAContextPlan(
            include_current_page=True,
            semantic_query="this concept",
        ),
        mode="function_calling",
        tool_traces=[ToolTrace(
            call_id="call-1",
            name="search_slides",
            status="executed",
            summary="tool executed",
        )],
    )
    bundle = QAContextBundle(
        prompt_context="context",
        sources=[QAContextSource(
            kind="selected_page",
            source_file="Lecture01.pdf",
            page_number=3,
            reason="current",
        )],
    )
    monkeypatch.setattr(
        qa_pipelines,
        "get_page_text_for_prompt",
        lambda *args: SimpleNamespace(text="page text", quality="ok", warning=None),
    )
    monkeypatch.setattr(qa_pipelines, "ensure_index_via_job", lambda *args, **kwargs: ([], object(), object()))
    monkeypatch.setattr(qa_pipelines, "plan_qa_context_with_tools", lambda *args: planning)
    monkeypatch.setattr(qa_pipelines, "build_qa_context_bundle", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(qa_pipelines, "build_qa_system_prompt", lambda *args, **kwargs: "system")
    monkeypatch.setattr(qa_pipelines, "PROMPT_FREE_QA", "{context}")
    monkeypatch.setattr(qa_pipelines, "stream_text", lambda *args, **kwargs: iter(["answer"]))

    emitted: list[str] = []
    result = qa_pipelines.run_qa_pipeline(_request(), None, emitted.append)

    assert result.answer_markdown == "answer"
    assert emitted == ["answer"]
    assert result.context_bundle is not None
    assert result.context_bundle.planner_mode == "function_calling"
    assert result.context_bundle.tool_traces[0].name == "search_slides"


def _agent_summary(stop_reason="finish_requested"):
    return AgentRunSummary(
        mode="bounded_agent_v2",
        rounds=1,
        executed_tool_calls=0,
        baseline_source_count=1,
        stop_reason=stop_reason,
        usage_status="unavailable",
        elapsed_ms=1,
    )


def test_ask_pipeline_uses_agent_only_when_v2_flag_is_on(monkeypatch):
    bundle = QAContextBundle(
        prompt_context="agent context",
        sources=[QAContextSource(
            kind="selected_page",
            source_file="Lecture01.pdf",
            page_number=3,
        )],
    )
    agent_result = AgentRuntimeResult(bundle=bundle, summary=_agent_summary())
    monkeypatch.setattr(qa_pipelines.config, "QA_AGENT_ENABLED", True)
    monkeypatch.setattr(qa_pipelines, "run_free_ask_agent", lambda **kwargs: agent_result)
    monkeypatch.setattr(qa_pipelines, "agent_policy_from_config", lambda: object())
    monkeypatch.setattr(qa_pipelines, "get_page_text_for_prompt", lambda *args: SimpleNamespace(text="page text", quality="ok", warning=None))
    monkeypatch.setattr(qa_pipelines, "ensure_index_via_job", lambda *args, **kwargs: ([], object(), object()))
    monkeypatch.setattr(qa_pipelines, "build_qa_system_prompt", lambda *args, **kwargs: "system")
    monkeypatch.setattr(qa_pipelines, "PROMPT_FREE_QA", "{context}")
    monkeypatch.setattr(qa_pipelines, "stream_text", lambda *args, **kwargs: iter(["answer"]))

    result = qa_pipelines.run_qa_pipeline(_request(), None, lambda _delta: None)

    assert result.answer_markdown == "answer"
    assert result.context_bundle is not None
    assert result.context_bundle.planner_mode == "bounded_agent_v2"
    assert result.context_bundle.agent_summary is not None


def test_agent_no_progress_falls_back_to_v1_once(monkeypatch):
    agent_summary = _agent_summary("fallback_v1").model_copy(update={
        "mode": "bounded_agent_v2_fallback",
    })
    agent_result = AgentRuntimeResult(
        bundle=QAContextBundle(prompt_context="baseline"),
        summary=agent_summary,
        fallback_requested=True,
    )
    planning = QAContextPlanningResult(
        plan=QAContextPlan(include_current_page=True),
        mode="json_planner",
    )
    v1_bundle = QAContextBundle(prompt_context="v1 context")
    calls = []
    monkeypatch.setattr(qa_pipelines.config, "QA_AGENT_ENABLED", True)
    monkeypatch.setattr(qa_pipelines, "run_free_ask_agent", lambda **kwargs: agent_result)
    monkeypatch.setattr(qa_pipelines, "agent_policy_from_config", lambda: object())
    monkeypatch.setattr(qa_pipelines, "_v1_ask_bundle", lambda *args: (calls.append(True) or (planning, v1_bundle)))
    monkeypatch.setattr(qa_pipelines, "get_page_text_for_prompt", lambda *args: SimpleNamespace(text="page text", quality="ok", warning=None))
    monkeypatch.setattr(qa_pipelines, "ensure_index_via_job", lambda *args, **kwargs: ([], object(), object()))
    monkeypatch.setattr(qa_pipelines, "build_qa_system_prompt", lambda *args, **kwargs: "system")
    monkeypatch.setattr(qa_pipelines, "PROMPT_FREE_QA", "{context}")
    monkeypatch.setattr(qa_pipelines, "stream_text", lambda *args, **kwargs: iter(["answer"]))

    result = qa_pipelines.run_qa_pipeline(_request(), None, lambda _delta: None)

    assert calls == [True]
    assert result.context_bundle is not None
    assert result.context_bundle.prompt_context == "v1 context"
    assert result.context_bundle.agent_summary is not None
    assert result.context_bundle.planner_mode == "json_planner"


def test_explain_and_translate_never_enter_agent_loop(monkeypatch):
    monkeypatch.setattr(qa_pipelines.config, "QA_AGENT_ENABLED", True)
    monkeypatch.setattr(
        qa_pipelines,
        "run_free_ask_agent",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Agent called")),
    )
    monkeypatch.setattr(
        qa_pipelines,
        "get_page_text_for_prompt",
        lambda *args: SimpleNamespace(text="page text", quality="ok", warning=None),
    )
    monkeypatch.setattr(
        qa_pipelines,
        "ensure_index_via_job",
        lambda *args, **kwargs: ([], object(), object()),
    )
    monkeypatch.setattr(
        qa_pipelines,
        "retrieve_for_current_page",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(qa_pipelines, "build_qa_system_prompt", lambda *args, **kwargs: "system")
    monkeypatch.setattr(qa_pipelines, "stream_text", lambda *args, **kwargs: iter(["answer"]))

    explain = qa_pipelines.run_qa_pipeline(
        _request().model_copy(update={"action_type": "explain"}),
        None,
        lambda _delta: None,
    )
    translate = qa_pipelines.run_qa_pipeline(
        _request().model_copy(update={"action_type": "translate"}),
        None,
        lambda _delta: None,
    )

    assert explain.answer_markdown == "answer"
    assert translate.answer_markdown == "answer"

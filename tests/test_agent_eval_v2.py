from __future__ import annotations

import time

from slide_tutor import agent_eval_v2
from slide_tutor.agent_eval import AgentEvalCase
from slide_tutor.models import (
    AgentRoundTrace,
    AgentRunSummary,
    LLMCallMetrics,
    ProjectFile,
    ProjectManifest,
    QAContextBundle,
    QAContextSource,
    SlidePage,
    ToolTrace,
)
from slide_tutor.qa_jobs import QAJobResult


def _case(**updates) -> AgentEvalCase:
    payload = dict(
        case_id="case-1",
        category="semantic_search",
        source_file="Lecture01.pdf",
        current_page=1,
        question="What are the expert rules?",
        expected_tools=["search_slides"],
        required_context_pages=[7],
        citation_gold_pages=[7],
    )
    payload.update(updates)
    return AgentEvalCase(**payload)


def _bundle(*, with_summary=True) -> QAContextBundle:
    summary = None
    if with_summary:
        summary = AgentRunSummary(
            mode="bounded_agent_v2",
            rounds=2,
            executed_tool_calls=1,
            baseline_source_count=1,
            stop_reason="finish_requested",
            round_traces=[AgentRoundTrace(
                round_number=1,
                tool_traces=[ToolTrace(
                    call_id="call-1",
                    name="search_slides",
                    status="executed",
                )],
                controller_metrics=LLMCallMetrics(
                    operation="agent_turn",
                    model="controller",
                    status="success",
                    latency_ms=20,
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                ),
                elapsed_ms=30,
            )],
            usage_status="exact",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            elapsed_ms=30,
        )
    return QAContextBundle(
        planner_mode="bounded_agent_v2",
        planner_summary="test",
        sources=[
            QAContextSource(
                kind="selected_page",
                source_file="Lecture01.pdf",
                page_number=1,
            ),
            QAContextSource(
                kind="retrieved_page",
                source_file="Lecture01.pdf",
                page_number=5,
            ),
            QAContextSource(
                kind="retrieved_page",
                source_file="Lecture01.pdf",
                page_number=6,
            ),
        ],
        tool_traces=[ToolTrace(
            call_id="call-1",
            name="search_slides",
            status="executed",
        )],
        agent_summary=summary,
    )


def test_case_result_separates_exact_acceptable_and_grounded_evidence():
    case = _case(
        acceptable_context_page_sets=[[5, 6]],
        acceptable_citation_pages=[5, 6],
    )
    started = time.perf_counter()
    result = agent_eval_v2._case_result(
        case,
        bundle=_bundle(),
        answer="The expert rules are described here [Lecture01.pdf, page 5].",
        known_project_pages={("lecture01.pdf", page) for page in (1, 5, 6, 7)},
        metrics=[LLMCallMetrics(
            operation="stream",
            model="answer",
            status="success",
            latency_ms=20,
            prompt_tokens=30,
            completion_tokens=10,
            total_tokens=40,
            first_token_latency_ms=5,
            streamed=True,
        )],
        started=started,
        stream_started=started + 0.01,
    )

    assert result.exact_context_hit is False
    assert result.acceptable_context_hit is True
    assert result.grounded_evidence_pass is True
    assert result.citation_task_pass is True
    assert result.task_success is True
    assert result.citations[0].exact_gold_page is False
    assert result.citations[0].acceptable_gold_page is True
    assert result.gold_review_status == "reviewed_versioned"


def test_case_without_reviewed_alternative_keeps_strict_exact_semantics():
    result = agent_eval_v2._case_result(
        _case(),
        bundle=_bundle(),
        answer="The expert rules are here [Lecture01.pdf, page 5].",
        known_project_pages={("lecture01.pdf", page) for page in (1, 5, 6, 7)},
        metrics=[],
        started=time.perf_counter(),
        stream_started=None,
    )

    assert result.exact_context_hit is False
    assert result.acceptable_context_hit is False
    assert result.task_success is False
    assert result.gold_review_status == "not_needed"


def test_run_case_calls_production_pipeline_with_agent_enabled(monkeypatch):
    calls = []

    def fake_pipeline(request, cancel_signal, emit, **kwargs):
        calls.append((request.action_type, kwargs["agent_enabled"]))
        kwargs["stream_start_callback"](time.perf_counter())
        kwargs["metrics_callback"](LLMCallMetrics(
            operation="stream",
            model="answer",
            status="success",
            latency_ms=1,
            first_token_latency_ms=1,
            streamed=True,
        ))
        return QAJobResult(
            answer_markdown="Grounded [Lecture01.pdf, page 7].",
            context_bundle=QAContextBundle(
                planner_mode="bounded_agent_v2",
                sources=[QAContextSource(
                    kind="retrieved_page",
                    source_file="Lecture01.pdf",
                    page_number=7,
                )],
                agent_summary=AgentRunSummary(
                    mode="bounded_agent_v2",
                    rounds=1,
                    executed_tool_calls=1,
                    baseline_source_count=1,
                    stop_reason="finish_requested",
                    usage_status="unavailable",
                    elapsed_ms=1,
                ),
                tool_traces=[ToolTrace(
                    call_id="call-1",
                    name="search_slides",
                    status="executed",
                )],
            ),
        )

    monkeypatch.setattr(agent_eval_v2, "run_qa_pipeline", fake_pipeline)
    project = ProjectManifest(
        project_id="project-1",
        name="project",
        created_at="2026-07-17T00:00:00Z",
        updated_at="2026-07-17T00:00:00Z",
        files=[ProjectFile(
            file_id="file-1",
            original_name="Lecture01.pdf",
            safe_name="Lecture01.pdf",
            sha256="a",
            size_bytes=1,
            page_count=7,
            created_at="2026-07-17T00:00:00Z",
            status="ready",
        )],
    )
    slide = SlidePage(
        slide_id="slide-1",
        module_id="project-1",
        project_id="project-1",
        file_id="file-1",
        source_file="Lecture01.pdf",
        source_type="pdf",
        page_number=1,
        global_index=1,
        text="Page 1",
        text_quality="ok",
    )
    result = agent_eval_v2._run_case(
        _case(required_context_pages=[7]),
        project=project,
        file_id="file-1",
        slides=[slide],
        known_project_pages={("lecture01.pdf", 7)},
    )

    assert calls == [("ask", True)]
    assert result.task_success is True
    assert result.answer_first_token_latency_ms == 1
    assert result.user_ttft_ms is not None


def test_report_contract_does_not_store_full_answer():
    result = agent_eval_v2._case_result(
        _case(required_context_pages=[5]),
        bundle=_bundle(),
        answer="A very long answer [Lecture01.pdf, page 5]." * 100,
        known_project_pages={("lecture01.pdf", page) for page in (1, 5, 6)},
        metrics=[],
        started=time.perf_counter(),
        stream_started=None,
    )
    payload = result.model_dump(mode="json")
    assert "answer_markdown" not in payload
    assert len(payload["answer_excerpt"]) <= 600

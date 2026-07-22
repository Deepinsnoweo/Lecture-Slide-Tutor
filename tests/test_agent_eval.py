from __future__ import annotations

from pathlib import Path

import pytest

from slide_tutor import agent_eval
from slide_tutor.agent_eval import (
    DEFAULT_AGENT_EVAL_FIXTURE,
    AgentEvalCase,
    AgentEvalCaseResult,
    AgentEvalReport,
    AgentEvalSuite,
    ParsedCitation,
    _assess_case,
    _summary,
    _validate_suite_references,
    load_agent_eval_suite,
    parse_answer_citations,
    rescore_agent_eval_report,
)
from slide_tutor.models import (
    LLMCallMetrics,
    ProjectFile,
    ProjectManifest,
    QAContextBundle,
    QAContextPlan,
    QAContextPlanningResult,
    QAContextSource,
    SlidePage,
    ToolTrace,
)


def _case() -> AgentEvalCase:
    return AgentEvalCase(
        case_id="case-1",
        category="explicit_page",
        source_file="Lecture01.pdf",
        current_page=3,
        question="Read page 17",
        expected_tools=["load_pages"],
        required_context_pages=[17],
        citation_gold_pages=[17],
    )


def _slide(page: int) -> SlidePage:
    return SlidePage(
        slide_id=f"slide-{page}",
        module_id="project-1",
        project_id="project-1",
        file_id="file-1",
        source_file="Lecture01.pdf",
        source_type="pdf",
        page_number=page,
        global_index=page,
        text=f"Page {page}",
        text_quality="ok",
    )


def test_main_fixture_has_40_cases_and_supplement_is_separate():
    supplement_path = Path("Docs/eval/agent-e2e-prospects-supplement-v1.json")
    if not DEFAULT_AGENT_EVAL_FIXTURE.exists() or not supplement_path.exists():
        pytest.skip("Internal evaluation fixtures are excluded from the published snapshot")
    suite = load_agent_eval_suite(DEFAULT_AGENT_EVAL_FIXTURE, min_cases=40)
    supplement = load_agent_eval_suite(supplement_path)

    assert len(suite.cases) == 40
    assert suite.project_name == "Machine Learning Test"
    assert len(supplement.cases) == 8
    assert supplement.project_name == "情报学展望"


def test_parse_answer_citations_supports_required_prompt_syntax():
    assert parse_answer_citations(
        "Claim [Lecture01.pdf, page 17]. "
        "中文 [Lecture02.pdf, 第4页]. "
        "Numbered [1, Lecture01.pdf, page 5]. "
        "Prose (Lecture02.pdf, page 6)."
    ) == [
        ("Lecture01.pdf", 17, True),
        ("Lecture02.pdf", 4, False),
        ("Lecture01.pdf", 5, False),
        ("Lecture02.pdf", 6, False),
    ]


def test_parse_answer_citations_strips_prose_from_prefix():
    assert parse_answer_citations(
        "Contrast (from Lecture01.pdf, page 21)."
    ) == [("Lecture01.pdf", 21, False)]


def test_assess_case_separates_task_success_from_tool_accuracy():
    planning = QAContextPlanningResult(
        plan=QAContextPlan(page_numbers=[17]),
        mode="function_calling",
        tool_traces=[ToolTrace(
            call_id="call-1",
            name="search_slides",
            status="executed",
        )],
    )
    bundle = QAContextBundle(sources=[QAContextSource(
        kind="explicit_page",
        source_file="Lecture01.pdf",
        page_number=17,
    )])
    metrics = [LLMCallMetrics(
        operation="text",
        model="answer",
        status="success",
        latency_ms=10,
        prompt_tokens=20,
        completion_tokens=5,
        total_tokens=25,
    )]

    result = _assess_case(
        _case(),
        planning=planning,
        bundle=bundle,
        answer="A grounded claim [Lecture01.pdf, page 17].",
        known_project_pages={("lecture01.pdf", 17)},
        observations=metrics,
        latency_ms=20,
    )

    assert result.task_success is True
    assert result.tool_selection_correct is False
    assert result.parameter_legal is True
    assert result.citation_task_pass is True
    assert result.reported_total_tokens == 25
    assert result.token_usage_complete is True


def test_assess_case_rejects_citations_not_in_actual_context():
    planning = QAContextPlanningResult(
        plan=QAContextPlan(page_numbers=[17]),
        mode="function_calling",
        tool_traces=[ToolTrace(
            call_id="call-1",
            name="load_pages",
            status="executed",
        )],
    )
    result = _assess_case(
        _case(),
        planning=planning,
        bundle=QAContextBundle(sources=[QAContextSource(
            kind="selected_page",
            source_file="Lecture01.pdf",
            page_number=3,
        )]),
        answer="Unsupported [Lecture01.pdf, page 17].",
        known_project_pages={("lecture01.pdf", 3), ("lecture01.pdf", 17)},
        observations=[],
        latency_ms=5,
    )

    assert result.task_success is False
    assert result.citations[0].project_valid is True
    assert result.citations[0].grounded_in_context is False
    assert "context_miss" in result.failure_types
    assert "citation_invalid" in result.failure_types


def test_summary_uses_applicable_denominator_for_parameter_legality():
    base = dict(
        category="explicit_page",
        question="q",
        source_file="Lecture01.pdf",
        current_page=1,
        latency_ms=10,
    )
    results = [
        AgentEvalCaseResult(
            **base,
            case_id="case-1",
            planning_mode="function_calling",
            task_success=True,
            tool_selection_correct=True,
            parameter_legal=True,
            context_hit=True,
            citation_task_pass=True,
            citations=[ParsedCitation(
                source_file="Lecture01.pdf",
                page_number=1,
                project_valid=True,
                grounded_in_context=True,
                gold_page=True,
                format_compliant=True,
            )],
        ),
        AgentEvalCaseResult(
            **base,
            case_id="case-2",
            planning_mode="function_calling_fallback",
            parameter_legal=None,
        ),
    ]

    summary = _summary(results)

    assert summary.parameter_applicable_cases == 1
    assert summary.parameter_legal_rate == 1.0
    assert summary.fallback_rate == 0.5
    assert summary.task_success_rate == 0.5


def test_reference_validation_fails_before_live_calls():
    suite = AgentEvalSuite(
        fixture_version="v1",
        name="suite",
        project_name="project",
        cases=[_case()],
    )

    with pytest.raises(ValueError, match="missing required context"):
        _validate_suite_references(
            suite,
            {"Lecture01.pdf": [_slide(3)]},
        )


def test_rescore_updates_citation_metrics_without_live_calls(monkeypatch):
    result = AgentEvalCaseResult(
        case_id="case-1",
        category="explicit_page",
        question="Read page 17",
        source_file="Lecture01.pdf",
        current_page=3,
        planning_mode="function_calling",
        expected_tools=["load_pages"],
        actual_tools=["load_pages"],
        tool_selection_correct=True,
        parameter_legal=True,
        required_context_pages=[17],
        actual_context_pages=[3, 17],
        context_hit=True,
        answer_nonempty=True,
        task_success=False,
        latency_ms=10,
        answer_markdown="Answer (from Lecture01.pdf, page 17).",
    )
    report = AgentEvalReport(
        fixture_version="v1",
        generated_at="2026-07-16T00:00:00Z",
        execution_command="live",
        machine="test",
        project={"project_id": "project-1"},
        model_contract={},
        summary=_summary([result]),
        cases=[result],
    )
    project = ProjectManifest(
        project_id="project-1",
        name="project",
        created_at="2026-07-16T00:00:00Z",
        updated_at="2026-07-16T00:00:00Z",
        files=[ProjectFile(
            file_id="file-1",
            original_name="Lecture01.pdf",
            safe_name="Lecture01.pdf",
            sha256="a",
            size_bytes=1,
            page_count=20,
            created_at="2026-07-16T00:00:00Z",
            status="ready",
        )],
    )
    monkeypatch.setattr(agent_eval, "_resolve_project", lambda *args: project)
    monkeypatch.setattr(
        agent_eval.projects,
        "load_file_slides",
        lambda *args: [_slide(3), _slide(17)],
    )

    rescored = rescore_agent_eval_report(
        report,
        AgentEvalSuite(
            fixture_version="v1",
            name="suite",
            project_name="project",
            cases=[_case()],
        ),
    )

    assert rescored.summary.task_success_rate == 1.0
    assert rescored.cases[0].citations[0].source_file == "Lecture01.pdf"
    assert rescored.model_contract["rescored_without_llm_calls"] is True

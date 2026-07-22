"""Formal evaluator for the production bounded Agent v2 Q&A path.

This module is deliberately separate from :mod:`slide_tutor.agent_eval`, which
continues to evaluate the v1 single-turn planner.  The v2 evaluator runs the
production ``run_qa_pipeline`` with the Agent flag forced on for the case,
collects backend streaming metrics, and reports strict versus reviewed
acceptable evidence separately.
"""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Literal

from pydantic import BaseModel, Field

from slide_tutor import config, projects
from slide_tutor.agent_eval import (
    DEFAULT_AGENT_EVAL_FIXTURE,
    AgentEvalCase,
    AgentEvalSuite,
    _normalized_file_name,
    _percentile,
    _resolve_project,
    _validate_suite_references,
    load_agent_eval_suite,
    parse_answer_citations,
)
from slide_tutor.models import LLMCallMetrics, ProjectManifest, QAContextBundle, SlidePage
from slide_tutor.qa_jobs import QAJobRequest
from slide_tutor.qa_pipelines import run_qa_pipeline


FailureOrigin = Literal[
    "product_agent",
    "provider_environment",
    "ui_harness",
    "gold_definition",
]


class AgentV2Citation(BaseModel):
    source_file: str
    page_number: int = Field(ge=1)
    project_valid: bool
    grounded_in_context: bool
    exact_gold_page: bool
    acceptable_gold_page: bool
    format_compliant: bool

    @property
    def grounded(self) -> bool:
        return self.project_valid and self.grounded_in_context


class AgentV2CaseResult(BaseModel):
    case_id: str
    category: str
    question: str
    source_file: str
    current_page: int
    planning_mode: str
    rounds: int = Field(ge=0)
    stop_reason: str
    fallback_requested: bool = False
    actual_tools: list[str] = Field(default_factory=list)
    expected_tools_present: bool = False
    parameter_legal: bool | None = None
    required_context_pages: list[int] = Field(default_factory=list)
    acceptable_context_page_sets: list[list[int]] = Field(default_factory=list)
    actual_context_pages: list[int] = Field(default_factory=list)
    exact_context_hit: bool = False
    acceptable_context_hit: bool = False
    grounded_evidence_pass: bool = False
    citations: list[AgentV2Citation] = Field(default_factory=list)
    citation_task_pass: bool = False
    citation_grounded_count: int = Field(default=0, ge=0)
    citation_format_compliant: bool | None = None
    answer_nonempty: bool = False
    task_success: bool = False
    usage_status: Literal["exact", "partial", "unavailable"]
    controller_tokens: int | None = Field(default=None, ge=0)
    answer_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    agent_latency_ms: float = Field(default=0.0, ge=0)
    controller_latency_ms: float = Field(default=0.0, ge=0)
    tool_orchestration_latency_ms: float = Field(default=0.0, ge=0)
    answer_first_token_latency_ms: float | None = Field(default=None, ge=0)
    user_ttft_ms: float | None = Field(default=None, ge=0)
    answer_stream_latency_ms: float | None = Field(default=None, ge=0)
    total_latency_ms: float = Field(ge=0)
    llm_calls: list[LLMCallMetrics] = Field(default_factory=list)
    failure_types: list[str] = Field(default_factory=list)
    failure_origin: FailureOrigin | None = None
    gold_review_status: Literal[
        "not_needed", "pending_review", "reviewed_versioned"
    ] = "not_needed"
    answer_excerpt: str = ""
    error: str | None = None


class AgentV2MetricSummary(BaseModel):
    case_count: int = Field(ge=0)
    task_success_rate: float = Field(ge=0, le=1)
    exact_context_hit_rate: float = Field(ge=0, le=1)
    acceptable_context_hit_rate: float = Field(ge=0, le=1)
    grounded_evidence_rate: float = Field(ge=0, le=1)
    expected_tool_presence_rate: float = Field(ge=0, le=1)
    parameter_legal_rate: float | None = Field(default=None, ge=0, le=1)
    citation_coverage_rate: float = Field(ge=0, le=1)
    citation_grounded_precision: float | None = Field(default=None, ge=0, le=1)
    citation_format_compliance_rate: float | None = Field(default=None, ge=0, le=1)
    mean_rounds: float = Field(ge=0)
    max_rounds_rate: float = Field(ge=0, le=1)
    fallback_rate: float = Field(ge=0, le=1)
    agent_latency_mean_ms: float = Field(ge=0)
    controller_latency_mean_ms: float = Field(ge=0)
    answer_first_token_latency_mean_ms: float | None = Field(default=None, ge=0)
    user_ttft_mean_ms: float | None = Field(default=None, ge=0)
    user_ttft_p50_ms: float | None = Field(default=None, ge=0)
    user_ttft_p95_ms: float | None = Field(default=None, ge=0)
    total_latency_mean_ms: float = Field(ge=0)
    total_latency_p50_ms: float = Field(ge=0)
    total_latency_p95_ms: float = Field(ge=0)
    provider_token_coverage_rate: float = Field(ge=0, le=1)
    controller_tokens: int | None = Field(default=None, ge=0)
    answer_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class AgentV2Report(BaseModel):
    report_version: str = "agent-e2e-v2-report-v1"
    fixture_version: str
    generated_at: str
    execution_command: str
    machine: str
    project: dict[str, Any]
    model_contract: dict[str, Any]
    summary: AgentV2MetricSummary
    by_category: dict[str, AgentV2MetricSummary] = Field(default_factory=dict)
    failure_origin_distribution: dict[str, int] = Field(default_factory=dict)
    cases: list[AgentV2CaseResult]


def _context_refs(bundle: QAContextBundle) -> set[tuple[str, int]]:
    return {
        (_normalized_file_name(source.source_file), source.page_number)
        for source in bundle.sources
        if source.kind != "memory"
        and source.source_file
        and source.page_number is not None
    }


def _actual_tools(bundle: QAContextBundle) -> list[str]:
    names: list[str] = []
    for trace in bundle.tool_traces:
        if trace.name not in names:
            names.append(trace.name)
    return names


def _controller_metrics(bundle: QAContextBundle) -> list[LLMCallMetrics]:
    if bundle.agent_summary is None:
        return []
    return [
        trace.controller_metrics
        for trace in bundle.agent_summary.round_traces
        if trace.controller_metrics is not None
    ]


def _sum_exact(metrics: list[LLMCallMetrics]) -> int | None:
    if not metrics or any(item.total_tokens is None for item in metrics):
        return None
    return sum(item.total_tokens or 0 for item in metrics)


def _usage_status(metrics: list[LLMCallMetrics]) -> Literal[
    "exact", "partial", "unavailable"
]:
    if not metrics or not any(
        value is not None
        for item in metrics
        for value in (item.prompt_tokens, item.completion_tokens, item.total_tokens)
    ):
        return "unavailable"
    if all(
        value is not None
        for item in metrics
        for value in (item.prompt_tokens, item.completion_tokens, item.total_tokens)
    ):
        return "exact"
    return "partial"


def _parameter_legal(bundle: QAContextBundle) -> bool | None:
    if not bundle.tool_traces:
        return None
    return all(trace.status != "rejected" for trace in bundle.tool_traces)


def _accepted_context(case: AgentEvalCase, actual: set[int]) -> tuple[bool, str]:
    exact = set(case.required_context_pages).issubset(actual)
    if exact:
        return True, "not_needed"
    sets = case.acceptable_context_page_sets
    if not sets:
        return False, "not_needed"
    accepted = any(set(page_set).issubset(actual) for page_set in sets)
    return accepted, "reviewed_versioned" if accepted else "pending_review"


def _failure_origin(
    *,
    error: str | None,
    metrics: list[LLMCallMetrics],
    acceptable_context_hit: bool,
    grounded_evidence_pass: bool,
) -> FailureOrigin | None:
    if error is None and acceptable_context_hit and grounded_evidence_pass:
        return None
    if any(metric.status == "error" for metric in metrics):
        return "provider_environment"
    if error and "gold" in error.casefold():
        return "gold_definition"
    return "product_agent"


def _case_result(
    case: AgentEvalCase,
    *,
    bundle: QAContextBundle,
    answer: str,
    known_project_pages: set[tuple[str, int]],
    metrics: list[LLMCallMetrics],
    started: float,
    stream_started: float | None,
    error: str | None = None,
) -> AgentV2CaseResult:
    context_refs = _context_refs(bundle)
    case_file = _normalized_file_name(case.source_file)
    actual_context_pages = sorted(
        page for source_file, page in context_refs if source_file == case_file
    )
    exact_context_hit = set(case.required_context_pages).issubset(
        actual_context_pages
    )
    acceptable_context_hit, gold_review_status = _accepted_context(
        case, set(actual_context_pages),
    )
    accepted_citation_pages = set(case.citation_gold_pages) | set(
        case.acceptable_citation_pages
    )
    citations: list[AgentV2Citation] = []
    for source_file, page_number, format_compliant in parse_answer_citations(answer):
        key = (_normalized_file_name(source_file), page_number)
        citations.append(AgentV2Citation(
            source_file=source_file,
            page_number=page_number,
            project_valid=key in known_project_pages,
            grounded_in_context=key in context_refs,
            exact_gold_page=(
                _normalized_file_name(source_file) == case_file
                and page_number in case.citation_gold_pages
            ),
            acceptable_gold_page=(
                _normalized_file_name(source_file) == case_file
                and page_number in accepted_citation_pages
            ),
            format_compliant=format_compliant,
        ))
    grounded_evidence_pass = bool(citations) and all(
        citation.grounded for citation in citations
    )
    citation_task_pass = grounded_evidence_pass and any(
        citation.acceptable_gold_page for citation in citations
    )
    answer_nonempty = bool(answer.strip())
    summary = bundle.agent_summary
    controller = _controller_metrics(bundle)
    stream_metrics = [metric for metric in metrics if metric.operation == "stream"]
    stream_metric = stream_metrics[-1] if stream_metrics else None
    controller_tokens = _sum_exact(controller)
    answer_tokens = (
        stream_metric.total_tokens
        if stream_metric is not None and stream_metric.total_tokens is not None
        else None
    )
    all_metrics = [*controller, *stream_metrics]
    usage_status = _usage_status(all_metrics)
    agent_latency = summary.elapsed_ms if summary is not None else 0.0
    controller_latency = sum(metric.latency_ms for metric in controller)
    answer_stream_latency = stream_metric.latency_ms if stream_metric else None
    answer_first_token = (
        stream_metric.first_token_latency_ms if stream_metric else None
    )
    user_ttft = None
    if stream_started is not None and answer_first_token is not None:
        user_ttft = (stream_started - started) * 1000 + answer_first_token
    total_latency = (time.perf_counter() - started) * 1000
    actual_tools = _actual_tools(bundle)
    expected_tools_present = set(case.expected_tools).issubset(actual_tools)
    parameter_legal = _parameter_legal(bundle)
    fallback_requested = bool(
        summary is not None and summary.mode == "bounded_agent_v2_fallback"
    )
    failures: list[str] = []
    if not exact_context_hit:
        failures.append("exact_context_miss")
    if not acceptable_context_hit:
        failures.append("acceptable_context_miss")
    if not expected_tools_present:
        failures.append("expected_tool_missing")
    if parameter_legal is False:
        failures.append("invalid_parameters")
    if not citations:
        failures.append("citation_missing")
    elif not grounded_evidence_pass:
        failures.append("citation_ungrounded")
    elif not citation_task_pass:
        failures.append("citation_gold_miss")
    if citations and not all(citation.format_compliant for citation in citations):
        failures.append("citation_format_noncompliant")
    if fallback_requested:
        failures.append("agent_fallback")
    if error:
        failures.append("runtime_error")
    task_success = (
        error is None
        and answer_nonempty
        and acceptable_context_hit
        and citation_task_pass
    )
    origin = _failure_origin(
        error=error,
        metrics=all_metrics,
        acceptable_context_hit=acceptable_context_hit,
        grounded_evidence_pass=grounded_evidence_pass,
    )
    if origin == "gold_definition":
        failures.append("gold_definition")
    return AgentV2CaseResult(
        case_id=case.case_id,
        category=case.category,
        question=case.question,
        source_file=case.source_file,
        current_page=case.current_page,
        planning_mode=bundle.planner_mode or "unknown",
        rounds=summary.rounds if summary is not None else 0,
        stop_reason=summary.stop_reason if summary is not None else "unknown",
        fallback_requested=fallback_requested,
        actual_tools=actual_tools,
        expected_tools_present=expected_tools_present,
        parameter_legal=parameter_legal,
        required_context_pages=list(case.required_context_pages),
        acceptable_context_page_sets=[list(item) for item in case.acceptable_context_page_sets],
        actual_context_pages=actual_context_pages,
        exact_context_hit=exact_context_hit,
        acceptable_context_hit=acceptable_context_hit,
        grounded_evidence_pass=grounded_evidence_pass,
        citations=citations,
        citation_task_pass=citation_task_pass,
        citation_grounded_count=sum(citation.grounded for citation in citations),
        citation_format_compliant=(
            all(citation.format_compliant for citation in citations)
            if citations else None
        ),
        answer_nonempty=answer_nonempty,
        task_success=task_success,
        usage_status=usage_status,
        controller_tokens=controller_tokens,
        answer_tokens=answer_tokens,
        total_tokens=(
            controller_tokens + answer_tokens
            if controller_tokens is not None and answer_tokens is not None
            else None
        ),
        agent_latency_ms=agent_latency,
        controller_latency_ms=controller_latency,
        tool_orchestration_latency_ms=max(agent_latency - controller_latency, 0.0),
        answer_first_token_latency_ms=answer_first_token,
        user_ttft_ms=user_ttft,
        answer_stream_latency_ms=answer_stream_latency,
        total_latency_ms=total_latency,
        llm_calls=all_metrics,
        failure_types=list(dict.fromkeys(failures)),
        failure_origin=origin,
        gold_review_status=gold_review_status,
        answer_excerpt=answer.strip()[:600],
        error=error,
    )


def _empty_bundle() -> QAContextBundle:
    return QAContextBundle()


def _run_case(
    case: AgentEvalCase,
    *,
    project: ProjectManifest,
    file_id: str,
    slides: list[SlidePage],
    known_project_pages: set[tuple[str, int]],
) -> AgentV2CaseResult:
    by_page = {slide.page_number: slide for slide in slides}
    current_slide = by_page.get(case.current_page)
    if current_slide is None:
        raise ValueError(
            f"{case.case_id}: {case.source_file} has no page {case.current_page}"
        )
    request = QAJobRequest(
        task_id=f"agent-v2-eval-{case.case_id}",
        session_id=f"agent-v2-eval-{case.case_id}",
        project_id=project.project_id,
        file_id=file_id,
        source_file=case.source_file,
        page_number=case.current_page,
        action_type="ask",
        language=case.language,
        question=case.question,
        current_slide=current_slide,
        file_page_count=max(by_page),
        memory=case.memory,
    )
    metrics: list[LLMCallMetrics] = []
    stream_started: float | None = None
    started = time.perf_counter()

    def mark_stream_start(value: float) -> None:
        nonlocal stream_started
        stream_started = value

    try:
        result = run_qa_pipeline(
            request,
            None,
            lambda _delta: None,
            metrics_callback=metrics.append,
            stream_start_callback=mark_stream_start,
            agent_enabled=True,
        )
        return _case_result(
            case,
            bundle=result.context_bundle or _empty_bundle(),
            answer=result.answer_markdown,
            known_project_pages=known_project_pages,
            metrics=metrics,
            started=started,
            stream_started=stream_started,
        )
    except Exception as exc:
        return _case_result(
            case,
            bundle=_empty_bundle(),
            answer="",
            known_project_pages=known_project_pages,
            metrics=metrics,
            started=started,
            stream_started=stream_started,
            error=f"{type(exc).__name__}: {str(exc)[:240]}",
        )


def _summary(results: list[AgentV2CaseResult]) -> AgentV2MetricSummary:
    if not results:
        return AgentV2MetricSummary(
            case_count=0,
            task_success_rate=0.0,
            exact_context_hit_rate=0.0,
            acceptable_context_hit_rate=0.0,
            grounded_evidence_rate=0.0,
            expected_tool_presence_rate=0.0,
            citation_coverage_rate=0.0,
            mean_rounds=0.0,
            max_rounds_rate=0.0,
            fallback_rate=0.0,
            agent_latency_mean_ms=0.0,
            controller_latency_mean_ms=0.0,
            total_latency_mean_ms=0.0,
            total_latency_p50_ms=0.0,
            total_latency_p95_ms=0.0,
            provider_token_coverage_rate=0.0,
        )
    parameter = [item.parameter_legal for item in results if item.parameter_legal is not None]
    citations = [citation for item in results for citation in item.citations]
    ttft = [item.user_ttft_ms for item in results if item.user_ttft_ms is not None]
    total = [item.total_latency_ms for item in results]
    controller_tokens = [item.controller_tokens for item in results if item.controller_tokens is not None]
    answer_tokens = [item.answer_tokens for item in results if item.answer_tokens is not None]
    exact_total = [item.total_tokens for item in results if item.total_tokens is not None]
    return AgentV2MetricSummary(
        case_count=len(results),
        task_success_rate=mean(item.task_success for item in results),
        exact_context_hit_rate=mean(item.exact_context_hit for item in results),
        acceptable_context_hit_rate=mean(item.acceptable_context_hit for item in results),
        grounded_evidence_rate=mean(item.grounded_evidence_pass for item in results),
        expected_tool_presence_rate=mean(item.expected_tools_present for item in results),
        parameter_legal_rate=mean(parameter) if parameter else None,
        citation_coverage_rate=mean(bool(item.citations) for item in results),
        citation_grounded_precision=(
            mean(item.grounded for item in citations) if citations else None
        ),
        citation_format_compliance_rate=(
            mean(item.format_compliant for item in citations) if citations else None
        ),
        mean_rounds=mean(item.rounds for item in results),
        max_rounds_rate=mean(item.stop_reason == "max_rounds" for item in results),
        fallback_rate=mean(item.fallback_requested for item in results),
        agent_latency_mean_ms=mean(item.agent_latency_ms for item in results),
        controller_latency_mean_ms=mean(item.controller_latency_ms for item in results),
        answer_first_token_latency_mean_ms=(
            mean(item.answer_first_token_latency_ms for item in results if item.answer_first_token_latency_ms is not None)
            if any(item.answer_first_token_latency_ms is not None for item in results) else None
        ),
        user_ttft_mean_ms=mean(ttft) if ttft else None,
        user_ttft_p50_ms=median(ttft) if ttft else None,
        user_ttft_p95_ms=_percentile(ttft, 0.95) if ttft else None,
        total_latency_mean_ms=mean(total),
        total_latency_p50_ms=median(total),
        total_latency_p95_ms=_percentile(total, 0.95),
        provider_token_coverage_rate=mean(item.usage_status == "exact" for item in results),
        controller_tokens=sum(controller_tokens) if controller_tokens else None,
        answer_tokens=sum(answer_tokens) if answer_tokens else None,
        total_tokens=sum(exact_total) if exact_total else None,
    )


def run_agent_eval_v2(
    suite: AgentEvalSuite,
    *,
    project_id: str | None = None,
    limit: int | None = None,
    execution_command: str = "",
) -> AgentV2Report:
    project = _resolve_project(suite, project_id)
    files_by_name = {
        file.original_name: file
        for file in project.files
        if file.status == "ready"
    }
    required_files = {case.source_file for case in suite.cases}
    missing_files = required_files - set(files_by_name)
    if missing_files:
        raise FileNotFoundError(
            f"benchmark project is missing files: {sorted(missing_files)}"
        )
    slides_by_file: dict[str, list[SlidePage]] = {}
    known_project_pages: set[tuple[str, int]] = set()
    for file_name, project_file in files_by_name.items():
        slides = projects.load_file_slides(project.project_id, project_file.file_id)
        slides_by_file[file_name] = slides
        known_project_pages.update(
            (_normalized_file_name(file_name), slide.page_number) for slide in slides
        )
    _validate_suite_references(suite, slides_by_file)
    selected_cases = suite.cases[:limit] if limit else suite.cases
    results: list[AgentV2CaseResult] = []
    for position, case in enumerate(selected_cases, start=1):
        print(f"[{position}/{len(selected_cases)}] {case.case_id}", flush=True)
        project_file = files_by_name[case.source_file]
        results.append(_run_case(
            case,
            project=project,
            file_id=project_file.file_id,
            slides=slides_by_file[case.source_file],
            known_project_pages=known_project_pages,
        ))
    grouped: dict[str, list[AgentV2CaseResult]] = defaultdict(list)
    for result in results:
        grouped[result.category].append(result)
    origins = Counter(
        result.failure_origin
        for result in results
        if result.failure_origin is not None
    )
    return AgentV2Report(
        fixture_version=suite.fixture_version,
        generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        execution_command=execution_command,
        machine=platform.platform(),
        project={
            "project_id": project.project_id,
            "name": project.name,
            "file_count": len(files_by_name),
            "page_count": sum(file.page_count for file in files_by_name.values()),
        },
        model_contract={
            "provider": config.LLM_PROVIDER,
            "base_url": config.LLM_BASE_URL,
            "agent_model": config.QA_AGENT_MODEL,
            "answer_model": config.LLM_MODEL,
            "agent_enabled_for_eval": True,
            "evaluator": "production_run_qa_pipeline",
            "evidence_contract": "exact+acceptable+grounded-v1",
            "latency_contract": "backend-stream-v1",
        },
        summary=_summary(results),
        by_category={
            category: _summary(category_results)
            for category, category_results in sorted(grouped.items())
        },
        failure_origin_distribution=dict(sorted(origins.items())),
        cases=results,
    )


def render_markdown(report: AgentV2Report) -> str:
    summary = report.summary
    lines = [
        "# Agentic Q&A v2 Evaluation",
        "",
        f"- Report version: `{report.report_version}`",
        f"- Fixture: `{report.fixture_version}`",
        f"- Generated: `{report.generated_at}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Cases | {summary.case_count} |",
        f"| Accepted task success | {summary.task_success_rate:.1%} |",
        f"| Exact context hit | {summary.exact_context_hit_rate:.1%} |",
        f"| Acceptable context hit | {summary.acceptable_context_hit_rate:.1%} |",
        f"| Grounded evidence | {summary.grounded_evidence_rate:.1%} |",
        f"| Mean rounds | {summary.mean_rounds:.3f} |",
        f"| User TTFT p50 | {summary.user_ttft_p50_ms if summary.user_ttft_p50_ms is not None else 'unavailable'} ms |",
        f"| Total latency p95 | {summary.total_latency_p95_ms:.1f} ms |",
        f"| Exact token coverage | {summary.provider_token_coverage_rate:.1%} |",
        "",
        "## Cases",
        "",
        "| Case | Success | Exact | Acceptable | Grounded | Rounds | TTFT ms | Origin |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for item in report.cases:
        lines.append(
            f"| {item.case_id} | {'yes' if item.task_success else 'no'} | "
            f"{'yes' if item.exact_context_hit else 'no'} | "
            f"{'yes' if item.acceptable_context_hit else 'no'} | "
            f"{'yes' if item.grounded_evidence_pass else 'no'} | {item.rounds} | "
            f"{item.user_ttft_ms if item.user_ttft_ms is not None else '—'} | "
            f"{item.failure_origin or '—'} |"
        )
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_AGENT_EVAL_FIXTURE)
    parser.add_argument("--project-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    command = " ".join(shlex.quote(item) for item in sys.argv)
    suite = load_agent_eval_suite(args.fixture)
    report = run_agent_eval_v2(
        suite,
        project_id=args.project_id,
        limit=args.limit,
        execution_command=command,
    )
    payload = report.model_dump(mode="json")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AgentV2CaseResult",
    "AgentV2Citation",
    "AgentV2MetricSummary",
    "AgentV2Report",
    "run_agent_eval_v2",
    "render_markdown",
]

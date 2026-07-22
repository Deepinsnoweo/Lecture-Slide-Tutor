"""End-to-end evaluation for the bounded Q&A tool-using workflow.

The evaluator runs the production planner, tool runtime, context builder,
retrieval, answer prompt, and citation parser against a versioned task suite.
It is intentionally opt-in because a live run calls the configured LLM twice
per normal case (planner plus final answer, with an extra JSON planner call
when Function Calling falls back).
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shlex
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from slide_tutor import config, index_store, projects
from slide_tutor.llm import complete_text
from slide_tutor.models import (
    LLMCallMetrics,
    ProjectManifest,
    QAContextBundle,
    QAContextPlan,
    QAContextPlanningResult,
    SlidePage,
)
from slide_tutor.prompts import PROMPT_FREE_QA, build_qa_system_prompt
from slide_tutor.qa_context import (
    build_qa_context_bundle,
    plan_qa_context_with_tools,
)


DEFAULT_AGENT_EVAL_FIXTURE = (
    Path(__file__).resolve().parents[1] / "Docs" / "eval" / "agent-e2e-gold-v1.json"
)
ToolName = Literal["load_pages", "load_neighbor_pages", "search_slides"]
_STRICT_CITATION_PATTERN = re.compile(
    r"\[\s*[^,\[\]]+?\.pdf\s*,\s*page\s+\d+\s*\]",
    re.IGNORECASE,
)
_CITATION_PATTERN = re.compile(
    r"(?P<raw>"
    r"[\[\(（]\s*"
    r"(?:\d+\s*[,，]\s*)?"
    r"(?P<file>[^,\]\)）]+?\.pdf)\s*[,，]\s*"
    r"(?:(?:page|p\.?)\s*(?P<page_en>\d+)|第\s*(?P<page_cjk>\d+)\s*页)"
    r"\s*[\]\)）]"
    r")",
    re.IGNORECASE,
)


class AgentEvalCase(BaseModel):
    case_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    current_page: int = Field(ge=1)
    question: str = Field(min_length=1)
    language: str = Field(default="简体中文", min_length=1)
    expected_tools: list[ToolName] = Field(min_length=1)
    required_context_pages: list[int] = Field(min_length=1)
    acceptable_context_page_sets: list[list[int]] = Field(default_factory=list)
    citation_gold_pages: list[int] = Field(min_length=1)
    acceptable_citation_pages: list[int] = Field(default_factory=list)
    memory: list[dict[str, Any]] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def validate_unique_expectations(self) -> "AgentEvalCase":
        if len(set(self.expected_tools)) != len(self.expected_tools):
            raise ValueError(f"{self.case_id}: expected_tools must be unique")
        if len(set(self.required_context_pages)) != len(self.required_context_pages):
            raise ValueError(
                f"{self.case_id}: required_context_pages must be unique"
            )
        for page_set in self.acceptable_context_page_sets:
            if not page_set:
                raise ValueError(
                    f"{self.case_id}: acceptable context page sets cannot be empty"
                )
            if len(set(page_set)) != len(page_set):
                raise ValueError(
                    f"{self.case_id}: acceptable context pages must be unique"
                )
        if len(self.acceptable_context_page_sets) != len({
            tuple(sorted(page_set)) for page_set in self.acceptable_context_page_sets
        }):
            raise ValueError(
                f"{self.case_id}: acceptable context page sets must be unique"
            )
        if len(set(self.citation_gold_pages)) != len(self.citation_gold_pages):
            raise ValueError(f"{self.case_id}: citation_gold_pages must be unique")
        if len(set(self.acceptable_citation_pages)) != len(self.acceptable_citation_pages):
            raise ValueError(
                f"{self.case_id}: acceptable_citation_pages must be unique"
            )
        return self


class AgentEvalSuite(BaseModel):
    fixture_version: str = Field(min_length=1)
    name: str = Field(min_length=1)
    project_name: str = Field(min_length=1)
    cases: list[AgentEvalCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_case_ids(self) -> "AgentEvalSuite":
        ids = [case.case_id for case in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError("agent eval case_id values must be unique")
        return self


class ParsedCitation(BaseModel):
    source_file: str
    page_number: int = Field(ge=1)
    project_valid: bool
    grounded_in_context: bool
    gold_page: bool
    format_compliant: bool

    @property
    def correct(self) -> bool:
        return self.project_valid and self.grounded_in_context


class AgentEvalCaseResult(BaseModel):
    case_id: str
    category: str
    question: str
    source_file: str
    current_page: int
    planning_mode: str
    fallback_reason: str | None = None
    expected_tools: list[str] = Field(default_factory=list)
    actual_tools: list[str] = Field(default_factory=list)
    tool_selection_correct: bool = False
    parameter_legal: bool | None = None
    required_context_pages: list[int] = Field(default_factory=list)
    actual_context_pages: list[int] = Field(default_factory=list)
    context_hit: bool = False
    citations: list[ParsedCitation] = Field(default_factory=list)
    citation_task_pass: bool = False
    citation_grounded_count: int = Field(default=0, ge=0)
    citation_format_compliant: bool | None = None
    answer_nonempty: bool = False
    task_success: bool = False
    latency_ms: float = Field(ge=0)
    llm_calls: list[LLMCallMetrics] = Field(default_factory=list)
    reported_prompt_tokens: int = Field(default=0, ge=0)
    reported_completion_tokens: int = Field(default=0, ge=0)
    reported_total_tokens: int = Field(default=0, ge=0)
    token_usage_complete: bool = False
    failure_types: list[str] = Field(default_factory=list)
    answer_markdown: str = ""
    answer_excerpt: str = ""
    error: str | None = None


class AgentEvalMetricSummary(BaseModel):
    case_count: int = Field(ge=0)
    task_success_rate: float = Field(ge=0, le=1)
    tool_selection_accuracy: float = Field(ge=0, le=1)
    parameter_legal_rate: float | None = Field(default=None, ge=0, le=1)
    parameter_applicable_cases: int = Field(ge=0)
    citation_correctness_rate: float = Field(ge=0, le=1)
    citation_grounded_precision: float | None = Field(default=None, ge=0, le=1)
    citation_coverage_rate: float = Field(ge=0, le=1)
    citation_format_compliance_rate: float | None = Field(default=None, ge=0, le=1)
    context_hit_rate: float = Field(ge=0, le=1)
    fallback_rate: float = Field(ge=0, le=1)
    latency_mean_ms: float = Field(ge=0)
    latency_p50_ms: float = Field(ge=0)
    latency_p95_ms: float = Field(ge=0)
    provider_token_coverage_rate: float = Field(ge=0, le=1)
    reported_total_tokens: int = Field(ge=0)
    complete_task_token_mean: float | None = Field(default=None, ge=0)
    complete_task_token_p50: float | None = Field(default=None, ge=0)
    complete_task_token_p95: float | None = Field(default=None, ge=0)


class AgentEvalReport(BaseModel):
    report_version: str = "agent-e2e-report-v1"
    fixture_version: str
    generated_at: str
    execution_command: str
    machine: str
    project: dict[str, Any]
    model_contract: dict[str, Any]
    summary: AgentEvalMetricSummary
    by_category: dict[str, AgentEvalMetricSummary] = Field(default_factory=dict)
    failure_type_distribution: dict[str, int] = Field(default_factory=dict)
    cases: list[AgentEvalCaseResult]


def load_agent_eval_suite(
    path: str | Path = DEFAULT_AGENT_EVAL_FIXTURE,
    *,
    min_cases: int = 1,
) -> AgentEvalSuite:
    fixture_path = Path(path)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    suite = AgentEvalSuite.model_validate(payload)
    if len(suite.cases) < min_cases:
        raise ValueError(
            f"{fixture_path} contains {len(suite.cases)} cases; "
            f"at least {min_cases} are required"
        )
    return suite


def parse_answer_citations(answer: str) -> list[tuple[str, int, bool]]:
    """Extract supported citation variants and mark the exact prompt format."""
    citations: list[tuple[str, int, bool]] = []
    for match in _CITATION_PATTERN.finditer(answer):
        source_file = re.sub(
            r"^(?:translated\s+)?from\s+",
            "",
            match.group("file").strip(),
            flags=re.IGNORECASE,
        )
        page_number = int(match.group("page_en") or match.group("page_cjk"))
        citations.append((
            source_file,
            page_number,
            bool(_STRICT_CITATION_PATTERN.fullmatch(match.group("raw"))),
        ))
    return citations


def _normalized_file_name(value: str) -> str:
    return Path(value.strip()).name.casefold()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _planning_parameter_legality(
    planning: QAContextPlanningResult,
) -> bool | None:
    if planning.tool_traces:
        return all(trace.status != "rejected" for trace in planning.tool_traces)
    reason = (planning.fallback_reason or "").casefold()
    invalid_markers = (
        "invalid argument",
        "malformed",
        "unknown tool",
        "exceeds limit",
        "qatoolselectionerror",
        "toolruntimeerror",
    )
    if any(marker in reason for marker in invalid_markers):
        return False
    return None


def _effective_tool_names(planning: QAContextPlanningResult) -> list[str]:
    names: list[str] = []
    for trace in planning.tool_traces:
        if trace.name not in names:
            names.append(trace.name)
    return names


def _context_refs(bundle: QAContextBundle) -> set[tuple[str, int]]:
    refs: set[tuple[str, int]] = set()
    for source in bundle.sources:
        if (
            source.kind != "memory"
            and source.source_file
            and source.page_number is not None
        ):
            refs.add((_normalized_file_name(source.source_file), source.page_number))
    return refs


def _reported_tokens(
    observations: list[LLMCallMetrics],
) -> tuple[int, int, int, bool]:
    prompt = sum(item.prompt_tokens or 0 for item in observations)
    completion = sum(item.completion_tokens or 0 for item in observations)
    total = sum(item.total_tokens or 0 for item in observations)
    complete = bool(observations) and all(
        item.total_tokens is not None for item in observations
    )
    return prompt, completion, total, complete


def _failure_types(
    *,
    planning: QAContextPlanningResult,
    tool_selection_correct: bool,
    parameter_legal: bool | None,
    context_hit: bool,
    citations: list[ParsedCitation],
    citation_task_pass: bool,
    answer_nonempty: bool,
    error: str | None,
) -> list[str]:
    failures: list[str] = []
    if planning.mode == "function_calling_fallback":
        failures.append("planner_fallback")
    if not tool_selection_correct:
        failures.append("tool_selection")
    if parameter_legal is False:
        failures.append("invalid_parameters")
    if not context_hit:
        failures.append("context_miss")
    if not citations:
        failures.append("citation_missing")
    elif any(not citation.correct for citation in citations):
        failures.append("citation_invalid")
    if citations and not any(citation.gold_page for citation in citations):
        failures.append("citation_gold_miss")
    if citations and any(not citation.format_compliant for citation in citations):
        failures.append("citation_format_noncompliant")
    if not citation_task_pass and "citation_missing" not in failures:
        failures.append("citation_task_failure")
    if not answer_nonempty:
        failures.append("answer_empty")
    if error:
        failures.append("runtime_error")
    return failures


def _assess_case(
    case: AgentEvalCase,
    *,
    planning: QAContextPlanningResult,
    bundle: QAContextBundle,
    answer: str,
    known_project_pages: set[tuple[str, int]],
    observations: list[LLMCallMetrics],
    latency_ms: float,
    error: str | None = None,
) -> AgentEvalCaseResult:
    actual_tools = _effective_tool_names(planning)
    tool_selection_correct = set(actual_tools) == set(case.expected_tools)
    parameter_legal = _planning_parameter_legality(planning)
    context_refs = _context_refs(bundle)
    case_file = _normalized_file_name(case.source_file)
    actual_context_pages = sorted(
        page for source_file, page in context_refs if source_file == case_file
    )
    context_hit = all(
        (case_file, page) in context_refs for page in case.required_context_pages
    )

    citations = [
        ParsedCitation(
            source_file=source_file,
            page_number=page_number,
            project_valid=(
                _normalized_file_name(source_file), page_number
            ) in known_project_pages,
            grounded_in_context=(
                _normalized_file_name(source_file), page_number
            ) in context_refs,
            gold_page=(
                _normalized_file_name(source_file) == case_file
                and page_number in case.citation_gold_pages
            ),
            format_compliant=format_compliant,
        )
        for source_file, page_number, format_compliant in parse_answer_citations(answer)
    ]
    citation_task_pass = bool(citations) and all(
        citation.correct for citation in citations
    ) and any(citation.gold_page for citation in citations)
    answer_nonempty = bool(answer.strip())
    task_success = (
        error is None
        and answer_nonempty
        and context_hit
        and citation_task_pass
    )
    prompt_tokens, completion_tokens, total_tokens, token_complete = (
        _reported_tokens(observations)
    )
    failures = _failure_types(
        planning=planning,
        tool_selection_correct=tool_selection_correct,
        parameter_legal=parameter_legal,
        context_hit=context_hit,
        citations=citations,
        citation_task_pass=citation_task_pass,
        answer_nonempty=answer_nonempty,
        error=error,
    )
    return AgentEvalCaseResult(
        case_id=case.case_id,
        category=case.category,
        question=case.question,
        source_file=case.source_file,
        current_page=case.current_page,
        planning_mode=planning.mode,
        fallback_reason=planning.fallback_reason,
        expected_tools=list(case.expected_tools),
        actual_tools=actual_tools,
        tool_selection_correct=tool_selection_correct,
        parameter_legal=parameter_legal,
        required_context_pages=list(case.required_context_pages),
        actual_context_pages=actual_context_pages,
        context_hit=context_hit,
        citations=citations,
        citation_task_pass=citation_task_pass,
        citation_grounded_count=sum(
            1 for citation in citations if citation.correct
        ),
        citation_format_compliant=(
            all(citation.format_compliant for citation in citations)
            if citations else None
        ),
        answer_nonempty=answer_nonempty,
        task_success=task_success,
        latency_ms=latency_ms,
        llm_calls=observations,
        reported_prompt_tokens=prompt_tokens,
        reported_completion_tokens=completion_tokens,
        reported_total_tokens=total_tokens,
        token_usage_complete=token_complete,
        failure_types=failures,
        answer_markdown=answer.strip(),
        answer_excerpt=answer.strip()[:600],
        error=error,
    )


def _empty_planning(error: Exception) -> QAContextPlanningResult:
    from slide_tutor.models import QAContextPlan

    return QAContextPlanningResult(
        plan=QAContextPlan(),
        mode="function_calling_fallback",
        fallback_reason=f"{type(error).__name__}: {str(error)[:160]}",
    )


def _run_case(
    case: AgentEvalCase,
    *,
    project: ProjectManifest,
    file_id: str,
    slides: list[SlidePage],
    index,
    known_project_pages: set[tuple[str, int]],
) -> AgentEvalCaseResult:
    by_page = {slide.page_number: slide for slide in slides}
    current_slide = by_page.get(case.current_page)
    if current_slide is None:
        raise ValueError(
            f"{case.case_id}: {case.source_file} has no page {case.current_page}"
        )
    observations: list[LLMCallMetrics] = []
    started = time.perf_counter()
    planning: QAContextPlanningResult | None = None
    bundle = QAContextBundle()
    answer = ""
    try:
        planning = plan_qa_context_with_tools(
            case.question,
            current_slide,
            max(by_page),
            case.memory,
            metrics_callback=observations.append,
            tool_calling_enabled=True,
        )
        bundle = build_qa_context_bundle(
            planning.plan,
            project.project_id,
            current_slide,
            file_id,
            max(by_page),
            case.memory,
            index,
            original_question=case.question,
        )
        ambiguity_note = ""
        if planning.plan.ambiguity_note:
            ambiguity_note = (
                f"\nTopic ambiguity: {planning.plan.ambiguity_note}\n"
                "Address the ambiguity explicitly and stay grounded in the "
                "supplied context."
            )
        user_prompt = PROMPT_FREE_QA.format(
            target_language=case.language,
            context=bundle.prompt_context,
            question=case.question,
            chat_history="",
            ambiguity_note=ambiguity_note,
        )
        answer = complete_text(
            build_qa_system_prompt(case.language),
            user_prompt,
            metrics_callback=observations.append,
        )
        return _assess_case(
            case,
            planning=planning,
            bundle=bundle,
            answer=answer,
            known_project_pages=known_project_pages,
            observations=observations,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    except Exception as exc:
        planning = planning or _empty_planning(exc)
        return _assess_case(
            case,
            planning=planning,
            bundle=bundle,
            answer=answer,
            known_project_pages=known_project_pages,
            observations=observations,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=f"{type(exc).__name__}: {str(exc)[:240]}",
        )


def _resolve_project(
    suite: AgentEvalSuite,
    project_id: str | None,
) -> ProjectManifest:
    if project_id:
        project = projects.load_project(project_id)
        if project.name != suite.project_name:
            raise ValueError(
                f"project {project_id!r} is named {project.name!r}; "
                f"fixture requires {suite.project_name!r}"
            )
        return project
    matches = [
        project for project in projects.list_projects()
        if project.name == suite.project_name
    ]
    if not matches:
        raise FileNotFoundError(
            f"no local project named {suite.project_name!r}; "
            "import the benchmark project or pass --project-id"
        )
    if len(matches) > 1:
        ids = ", ".join(project.project_id for project in matches)
        raise ValueError(
            f"multiple projects named {suite.project_name!r}: {ids}; "
            "pass --project-id"
        )
    return matches[0]


def _validate_suite_references(
    suite: AgentEvalSuite,
    slides_by_file: dict[str, list[SlidePage]],
) -> None:
    errors: list[str] = []
    pages_by_file = {
        source_file: {slide.page_number for slide in slides}
        for source_file, slides in slides_by_file.items()
    }
    for case in suite.cases:
        known_pages = pages_by_file.get(case.source_file, set())
        if case.current_page not in known_pages:
            errors.append(
                f"{case.case_id}: missing current page "
                f"{case.source_file} page {case.current_page}"
            )
        missing_context = set(case.required_context_pages) - known_pages
        if missing_context:
            errors.append(
                f"{case.case_id}: missing required context pages "
                f"{sorted(missing_context)} in {case.source_file}"
            )
        missing_citations = set(case.citation_gold_pages) - known_pages
        if missing_citations:
            errors.append(
                f"{case.case_id}: missing citation gold pages "
                f"{sorted(missing_citations)} in {case.source_file}"
            )
    if errors:
        raise ValueError(
            "Invalid Agent evaluation references:\n" + "\n".join(errors)
        )


def _summary(results: list[AgentEvalCaseResult]) -> AgentEvalMetricSummary:
    if not results:
        return AgentEvalMetricSummary(
            case_count=0,
            task_success_rate=0.0,
            tool_selection_accuracy=0.0,
            parameter_applicable_cases=0,
            citation_correctness_rate=0.0,
            citation_coverage_rate=0.0,
            context_hit_rate=0.0,
            fallback_rate=0.0,
            latency_mean_ms=0.0,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            provider_token_coverage_rate=0.0,
            reported_total_tokens=0,
        )
    parameter_results = [
        result.parameter_legal
        for result in results
        if result.parameter_legal is not None
    ]
    citations = [
        citation
        for result in results
        for citation in result.citations
    ]
    complete_tokens = [
        float(result.reported_total_tokens)
        for result in results
        if result.token_usage_complete
    ]
    latencies = [result.latency_ms for result in results]
    return AgentEvalMetricSummary(
        case_count=len(results),
        task_success_rate=mean(result.task_success for result in results),
        tool_selection_accuracy=mean(
            result.tool_selection_correct for result in results
        ),
        parameter_legal_rate=(
            mean(parameter_results) if parameter_results else None
        ),
        parameter_applicable_cases=len(parameter_results),
        citation_correctness_rate=mean(
            result.citation_task_pass for result in results
        ),
        citation_grounded_precision=(
            mean(citation.correct for citation in citations)
            if citations else None
        ),
        citation_coverage_rate=mean(bool(result.citations) for result in results),
        citation_format_compliance_rate=(
            mean(citation.format_compliant for citation in citations)
            if citations else None
        ),
        context_hit_rate=mean(result.context_hit for result in results),
        fallback_rate=mean(
            result.planning_mode == "function_calling_fallback"
            for result in results
        ),
        latency_mean_ms=mean(latencies),
        latency_p50_ms=median(latencies),
        latency_p95_ms=_percentile(latencies, 0.95),
        provider_token_coverage_rate=mean(
            result.token_usage_complete for result in results
        ),
        reported_total_tokens=sum(
            result.reported_total_tokens for result in results
        ),
        complete_task_token_mean=(
            mean(complete_tokens) if complete_tokens else None
        ),
        complete_task_token_p50=(
            median(complete_tokens) if complete_tokens else None
        ),
        complete_task_token_p95=(
            _percentile(complete_tokens, 0.95) if complete_tokens else None
        ),
    )


def run_agent_eval(
    suite: AgentEvalSuite,
    *,
    project_id: str | None = None,
    limit: int | None = None,
    execution_command: str = "",
) -> AgentEvalReport:
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
    index = index_store.load_index(project.project_id)
    if index is None:
        raise RuntimeError(
            "benchmark requires a current prebuilt project index; "
            "run Pre-index project before the evaluation"
        )

    slides_by_file: dict[str, list[SlidePage]] = {}
    known_project_pages: set[tuple[str, int]] = set()
    for file_name, project_file in files_by_name.items():
        slides = projects.load_file_slides(project.project_id, project_file.file_id)
        slides_by_file[file_name] = slides
        for slide in slides:
            known_project_pages.add(
                (_normalized_file_name(file_name), slide.page_number)
            )
    _validate_suite_references(suite, slides_by_file)

    selected_cases = suite.cases[:limit] if limit else suite.cases
    results: list[AgentEvalCaseResult] = []
    for position, case in enumerate(selected_cases, start=1):
        project_file = files_by_name[case.source_file]
        print(
            f"[{position}/{len(selected_cases)}] {case.case_id}",
            flush=True,
        )
        results.append(_run_case(
            case,
            project=project,
            file_id=project_file.file_id,
            slides=slides_by_file[case.source_file],
            index=index,
            known_project_pages=known_project_pages,
        ))

    grouped: dict[str, list[AgentEvalCaseResult]] = defaultdict(list)
    for result in results:
        grouped[result.category].append(result)
    failures = Counter(
        failure
        for result in results
        for failure in result.failure_types
    )
    return AgentEvalReport(
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
            "planner_model": config.QA_PLANNER_MODEL,
            "answer_model": config.LLM_MODEL,
            "tool_calling_forced_for_eval": True,
            "retrieval_backend": str(index[1].__class__.__name__),
            "citation_parser": "citation-v2",
        },
        summary=_summary(results),
        by_category={
            category: _summary(category_results)
            for category, category_results in sorted(grouped.items())
        },
        failure_type_distribution=dict(sorted(failures.items())),
        cases=results,
    )


def rescore_agent_eval_report(
    report: AgentEvalReport,
    suite: AgentEvalSuite,
    *,
    project_id: str | None = None,
) -> AgentEvalReport:
    """Recompute deterministic citation/task metrics without new LLM calls."""
    project = _resolve_project(suite, project_id)
    cases_by_id = {case.case_id: case for case in suite.cases}
    known_project_pages: set[tuple[str, int]] = set()
    for project_file in project.files:
        if project_file.status != "ready":
            continue
        for slide in projects.load_file_slides(
            project.project_id, project_file.file_id,
        ):
            known_project_pages.add((
                _normalized_file_name(project_file.original_name),
                slide.page_number,
            ))

    rescored: list[AgentEvalCaseResult] = []
    for result in report.cases:
        case = cases_by_id.get(result.case_id)
        if case is None:
            raise ValueError(f"report contains unknown case: {result.case_id}")
        case_file = _normalized_file_name(case.source_file)
        context_refs = {
            (case_file, page_number)
            for page_number in result.actual_context_pages
        }
        citations = [
            ParsedCitation(
                source_file=source_file,
                page_number=page_number,
                project_valid=(
                    _normalized_file_name(source_file), page_number
                ) in known_project_pages,
                grounded_in_context=(
                    _normalized_file_name(source_file), page_number
                ) in context_refs,
                gold_page=(
                    _normalized_file_name(source_file) == case_file
                    and page_number in case.citation_gold_pages
                ),
                format_compliant=format_compliant,
            )
            for source_file, page_number, format_compliant
            in parse_answer_citations(result.answer_markdown)
        ]
        citation_task_pass = bool(citations) and all(
            citation.correct for citation in citations
        ) and any(citation.gold_page for citation in citations)
        planning = QAContextPlanningResult(
            plan=QAContextPlan(),
            mode=result.planning_mode,
            fallback_reason=result.fallback_reason,
        )
        failures = _failure_types(
            planning=planning,
            tool_selection_correct=result.tool_selection_correct,
            parameter_legal=result.parameter_legal,
            context_hit=result.context_hit,
            citations=citations,
            citation_task_pass=citation_task_pass,
            answer_nonempty=result.answer_nonempty,
            error=result.error,
        )
        rescored.append(result.model_copy(update={
            "citations": citations,
            "citation_task_pass": citation_task_pass,
            "citation_grounded_count": sum(
                1 for citation in citations if citation.correct
            ),
            "citation_format_compliant": (
                all(citation.format_compliant for citation in citations)
                if citations else None
            ),
            "task_success": (
                result.error is None
                and result.answer_nonempty
                and result.context_hit
                and citation_task_pass
            ),
            "failure_types": failures,
        }))

    grouped: dict[str, list[AgentEvalCaseResult]] = defaultdict(list)
    for result in rescored:
        grouped[result.category].append(result)
    failures = Counter(
        failure
        for result in rescored
        for failure in result.failure_types
    )
    contract = dict(report.model_contract)
    contract["citation_parser"] = "citation-v2"
    contract["rescored_without_llm_calls"] = True
    return report.model_copy(update={
        "model_contract": contract,
        "summary": _summary(rescored),
        "by_category": {
            category: _summary(category_results)
            for category, category_results in sorted(grouped.items())
        },
        "failure_type_distribution": dict(sorted(failures.items())),
        "cases": rescored,
    })


def _rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def render_markdown_report(report: AgentEvalReport) -> str:
    summary = report.summary
    lines = [
        "# Agent Q&A 端到端评测报告",
        "",
        f"- 生成时间：{report.generated_at}",
        f"- 项目：{report.project['name']}（{report.project['page_count']} 页）",
        f"- Planner：`{report.model_contract['planner_model']}`",
        f"- Answer model：`{report.model_contract['answer_model']}`",
        f"- 任务数：{summary.case_count}",
        "",
        "## 核心指标",
        "",
        "| 指标 | 结果 |",
        "| --- | ---: |",
        f"| 任务成功率 | {_rate(summary.task_success_rate)} |",
        f"| 工具选择准确率（case exact match） | {_rate(summary.tool_selection_accuracy)} |",
        (
            "| 参数合法率（有参数证据的 case） | "
            f"{_rate(summary.parameter_legal_rate)} "
            f"（n={summary.parameter_applicable_cases}） |"
        ),
        f"| 引用正确率（严格 case 口径） | {_rate(summary.citation_correctness_rate)} |",
        f"| 引用 grounded precision（citation 口径） | {_rate(summary.citation_grounded_precision)} |",
        f"| 引用覆盖率 | {_rate(summary.citation_coverage_rate)} |",
        (
            "| 引用格式合规率（精确 `[file, page N]`） | "
            f"{_rate(summary.citation_format_compliance_rate)} |"
        ),
        f"| 必需证据页命中率 | {_rate(summary.context_hit_rate)} |",
        f"| Fallback 率 | {_rate(summary.fallback_rate)} |",
        "",
        "## 延迟与 Token",
        "",
        (
            f"- 端到端延迟：mean {summary.latency_mean_ms:.0f} ms，"
            f"p50 {summary.latency_p50_ms:.0f} ms，"
            f"p95 {summary.latency_p95_ms:.0f} ms。"
        ),
        (
            f"- Provider Token 覆盖率："
            f"{_rate(summary.provider_token_coverage_rate)}；"
            f"已报告 Token 合计 {summary.reported_total_tokens}。"
        ),
    ]
    if summary.complete_task_token_mean is not None:
        lines.append(
            "- usage 完整任务的 Token："
            f"mean {summary.complete_task_token_mean:.0f}，"
            f"p50 {summary.complete_task_token_p50:.0f}，"
            f"p95 {summary.complete_task_token_p95:.0f}。"
        )
    else:
        lines.append("- Provider 未返回完整 usage，无法给出精确的单任务 Token 分布。")

    lines.extend([
        "",
        "## 按任务类型",
        "",
        "| 类型 | n | 成功率 | 工具准确率 | 引用正确率 | Fallback | p50 延迟 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for category, category_summary in report.by_category.items():
        lines.append(
            f"| {category} | {category_summary.case_count} | "
            f"{_rate(category_summary.task_success_rate)} | "
            f"{_rate(category_summary.tool_selection_accuracy)} | "
            f"{_rate(category_summary.citation_correctness_rate)} | "
            f"{_rate(category_summary.fallback_rate)} | "
            f"{category_summary.latency_p50_ms:.0f} ms |"
        )

    lines.extend([
        "",
        "## 失败类型分布",
        "",
        "失败类型为多标签统计，同一任务可计入多个类型。",
        "",
        "| 类型 | 次数 |",
        "| --- | ---: |",
    ])
    if report.failure_type_distribution:
        for failure, count in report.failure_type_distribution.items():
            lines.append(f"| {failure} | {count} |")
    else:
        lines.append("| 无 | 0 |")

    notable = [
        result for result in report.cases
        if result.failure_types
    ]
    lines.extend([
        "",
        "## 异常与失败任务",
        "",
        "| Case | 类型 | Planner | 期望工具 | 实际工具 | 失败标签 |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    if notable:
        for result in notable:
            lines.append(
                f"| {result.case_id} | {result.category} | "
                f"{result.planning_mode} | "
                f"{', '.join(result.expected_tools)} | "
                f"{', '.join(result.actual_tools) or '—'} | "
                f"{', '.join(result.failure_types)} |"
            )
    else:
        lines.append("| 无 | — | — | — | — | — |")
    return "\n".join(lines) + "\n"


def save_agent_eval_report(
    json_path: str | Path,
    markdown_path: str | Path,
    report: AgentEvalReport,
) -> None:
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_output.write_text(
        render_markdown_report(report),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the live bounded-Agent Q&A end-to-end evaluation.",
    )
    parser.add_argument(
        "--fixture",
        default=str(DEFAULT_AGENT_EVAL_FIXTURE),
        help="Versioned Agent task suite JSON.",
    )
    parser.add_argument(
        "--project-id",
        help="Disambiguate the local project when names are duplicated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N cases for a smoke test.",
    )
    parser.add_argument(
        "--json-output",
        default="Docs/eval/agent-e2e-machine-learning-live.json",
    )
    parser.add_argument(
        "--markdown-output",
        default="Docs/eval/agent-e2e-machine-learning-live.md",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    command = shlex.join([sys.executable, "-m", "slide_tutor.agent_eval", *sys.argv[1:]])
    suite = load_agent_eval_suite(args.fixture)
    report = run_agent_eval(
        suite,
        project_id=args.project_id,
        limit=args.limit,
        execution_command=command,
    )
    save_agent_eval_report(args.json_output, args.markdown_output, report)
    print(render_markdown_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

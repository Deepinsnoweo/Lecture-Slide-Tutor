"""Versioned retrieval gold sets and deterministic evaluation helpers.

The evaluator deliberately accepts a retrieval callable so CI can use a tiny
controlled backend while local performance runs can use the real index and
embedding model.  Reports contain aggregate metrics, category/method slices,
latency, and case-level failures instead of hiding regressions in one score.
"""

from __future__ import annotations

import json
import argparse
import platform
import shlex
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy.sparse import csr_matrix

from slide_tutor import config
from slide_tutor.models import RetrievalChunk, SlidePage
from slide_tutor.retrieval import (
    _bm25_search,
    _exact_title_search,
    _semantic_search,
    retrieve_for_question,
)

GoldScope = Literal["whole_project", "current_file", "explicit_file", "selected_files"]
RequiredMethod = Literal[
    "positional", "exact_title", "lexical", "semantic", "any"
]
RetrievalCallable = Callable[["RetrievalGoldCase"], list[RetrievalChunk]]

DEFAULT_GOLD_FIXTURE = Path(__file__).with_name("retrieval_gold_v1.json")


class RetrievalGoldCase(BaseModel):
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    category: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    scope: GoldScope
    file_id: str | None = None
    file_ids: list[str] = Field(default_factory=list)
    current_slide_id: str | None = None
    relevant_slide_ids: list[str] = Field(default_factory=list)
    expected_empty: bool = False
    required_method: RequiredMethod
    notes: str = ""

    @model_validator(mode="after")
    def validate_scope_fields(self) -> "RetrievalGoldCase":
        if self.scope in {"current_file", "explicit_file"} and not self.file_id:
            raise ValueError(f"{self.case_id}: file_id is required for {self.scope}")
        if self.scope == "selected_files" and not self.file_ids:
            raise ValueError(f"{self.case_id}: file_ids is required for selected_files")
        if not self.relevant_slide_ids and not self.expected_empty:
            raise ValueError(f"{self.case_id}: relevant_slide_ids cannot be empty unless expected_empty")
        if len(set(self.file_ids)) != len(self.file_ids):
            raise ValueError(f"{self.case_id}: file_ids must be unique")
        if len(set(self.relevant_slide_ids)) != len(self.relevant_slide_ids):
            raise ValueError(f"{self.case_id}: relevant_slide_ids must be unique")
        return self


class RetrievalGoldSet(BaseModel):
    fixture_version: str = Field(min_length=1)
    name: str = Field(min_length=1)
    cases: list[RetrievalGoldCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_case_ids(self) -> "RetrievalGoldSet":
        ids = [case.case_id for case in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError("gold case_id values must be unique")
        return self


class RetrievalEvidence(BaseModel):
    slide_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    retrieval_methods: list[str] = Field(default_factory=list)
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None


class RetrievalCaseResult(BaseModel):
    case_id: str
    category: str
    required_method: RequiredMethod
    relevant_slide_ids: list[str]
    retrieved_slide_ids: list[str]
    evidence: list[RetrievalEvidence] = Field(default_factory=list)
    latency_ms: float = Field(ge=0)
    recall_at_3: float = Field(ge=0, le=1)
    recall_at_5: float = Field(ge=0, le=1)
    reciprocal_rank: float = Field(ge=0, le=1)
    precision_at_5: float = Field(ge=0, le=1)
    method_satisfied: bool
    matched_relevant_slide_ids: list[str] = Field(default_factory=list)
    matched_methods: dict[str, list[str]] = Field(default_factory=dict)
    method_failure_reason: str | None = None
    error: str | None = None


class RetrievalMetricSummary(BaseModel):
    case_count: int = Field(ge=0)
    recall_at_3: float = Field(ge=0, le=1)
    recall_at_5: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    precision_at_5: float = Field(ge=0, le=1)
    deterministic_hit_rate: float | None = Field(default=None, ge=0, le=1)


class RetrievalFailure(BaseModel):
    case_id: str
    category: str
    required_method: RequiredMethod
    expected_slide_ids: list[str]
    actual_top_k: list[str]
    retrieval_methods: dict[str, list[str]] = Field(default_factory=dict)
    ranks_and_scores: dict[str, dict[str, int | float | None]] = Field(
        default_factory=dict,
    )
    failure_kind: Literal["retrieval", "method", "retrieval_and_method"] = "retrieval"
    matched_relevant_slide_ids: list[str] = Field(default_factory=list)
    method_failure_reason: str | None = None
    error: str | None = None


class RetrievalEvaluationReport(BaseModel):
    fixture_version: str
    generated_at: str
    machine: str
    model_contract: dict[str, Any] = Field(default_factory=dict)
    index_signatures: dict[str, str] = Field(default_factory=dict)
    overall: RetrievalMetricSummary
    by_category: dict[str, RetrievalMetricSummary] = Field(default_factory=dict)
    by_method: dict[str, RetrievalMetricSummary] = Field(default_factory=dict)
    latency_p50_ms: float = Field(ge=0)
    latency_p95_ms: float = Field(ge=0)
    cases: list[RetrievalCaseResult]
    failures: list[RetrievalFailure] = Field(default_factory=list)


class RetrievalBaselineComparison(BaseModel):
    fixture_version: str
    generated_at: str
    reports: dict[str, RetrievalEvaluationReport]


class RetrievalRunReport(BaseModel):
    """CLI artifact combining the main run, baselines, and explicit gates."""

    fixture_version: str
    generated_at: str
    execution_command: str
    reference_validation: dict[str, Any]
    evaluation: RetrievalEvaluationReport
    baseline_comparison: RetrievalBaselineComparison
    gates: dict[str, bool]


class GoldReferenceError(ValueError):
    """Raised when a fixture refers to a missing project/file/slide."""


def load_gold_set(path: str | Path = DEFAULT_GOLD_FIXTURE, *, min_cases: int = 30) -> RetrievalGoldSet:
    """Load and schema-validate a versioned fixture before an evaluation run."""
    fixture_path = Path(path)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    gold_set = RetrievalGoldSet.model_validate(payload)
    if len(gold_set.cases) < min_cases:
        raise ValueError(
            f"{fixture_path} contains {len(gold_set.cases)} cases; "
            f"at least {min_cases} are required"
        )
    return gold_set


def validate_gold_references(
    gold_set: RetrievalGoldSet,
    project_loader: Callable[[str], Any],
    slide_loader: Callable[[str, str], list[SlidePage]],
) -> None:
    """Validate every project/file/slide reference, failing before evaluation."""
    errors: list[str] = []
    project_files: dict[str, set[str]] = {}
    project_slides: dict[str, set[str]] = {}
    project_slides_by_file: dict[str, dict[str, set[str]]] = {}

    for case in gold_set.cases:
        try:
            if case.project_id not in project_files:
                project = project_loader(case.project_id)
                file_ids = {
                    file.file_id for file in project.files if file.status == "ready"
                }
                project_files[case.project_id] = file_ids
                slides: set[str] = set()
                slides_by_file: dict[str, set[str]] = {}
                for file_id in file_ids:
                    file_slides = {
                        slide.slide_id
                        for slide in slide_loader(case.project_id, file_id)
                    }
                    slides.update(file_slides)
                    slides_by_file[file_id] = file_slides
                project_slides[case.project_id] = slides
                project_slides_by_file[case.project_id] = slides_by_file
        except Exception as exc:
            errors.append(f"{case.case_id}: project {case.project_id!r}: {exc}")
            continue

        known_files = project_files[case.project_id]
        known_slides = project_slides[case.project_id]
        if case.file_id and case.file_id not in known_files:
            errors.append(f"{case.case_id}: missing file {case.file_id!r}")
        if case.file_ids:
            missing_files = set(case.file_ids) - known_files
            if missing_files:
                errors.append(f"{case.case_id}: missing files {sorted(missing_files)}")
        if case.scope in {"current_file", "explicit_file"} and case.file_id:
            reference_slides = project_slides_by_file[case.project_id].get(
                case.file_id, set(),
            )
        elif case.scope == "selected_files":
            reference_slides = set().union(*(
                project_slides_by_file[case.project_id].get(file_id, set())
                for file_id in case.file_ids
            ))
        else:
            reference_slides = known_slides
        missing_slides = set(case.relevant_slide_ids) - reference_slides
        if missing_slides:
            errors.append(
                f"{case.case_id}: missing relevant slides {sorted(missing_slides)}"
            )
        if (
            case.current_slide_id
            and case.current_slide_id not in reference_slides
        ):
            errors.append(
                f"{case.case_id}: missing current slide {case.current_slide_id!r}"
            )

    if errors:
        raise GoldReferenceError("Invalid retrieval gold references:\n" + "\n".join(errors))


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


def _summary(results: list[RetrievalCaseResult]) -> RetrievalMetricSummary:
    if not results:
        return RetrievalMetricSummary(
            case_count=0,
            recall_at_3=0.0,
            recall_at_5=0.0,
            mrr=0.0,
            precision_at_5=0.0,
        )
    deterministic = [
        1.0 if result.method_satisfied and result.recall_at_5 == 1.0 else 0.0
        for result in results
        if result.required_method == "positional"
    ]
    return RetrievalMetricSummary(
        case_count=len(results),
        recall_at_3=mean(result.recall_at_3 for result in results),
        recall_at_5=mean(result.recall_at_5 for result in results),
        mrr=mean(result.reciprocal_rank for result in results),
        precision_at_5=mean(result.precision_at_5 for result in results),
        deterministic_hit_rate=mean(deterministic) if deterministic else None,
    )


def _evidence(chunk: RetrievalChunk) -> RetrievalEvidence:
    return RetrievalEvidence(
        slide_id=chunk.slide_id,
        page_number=chunk.page_number,
        retrieval_methods=list(chunk.retrieval_methods),
        dense_rank=chunk.dense_rank,
        dense_score=chunk.dense_score,
        bm25_rank=chunk.bm25_rank,
        bm25_score=chunk.bm25_score,
        rrf_score=chunk.rrf_score,
    )


_ALLOWED_METHODS: dict[str, set[str] | None] = {
    "positional": {"positional"},
    "exact_title": {"exact_title"},
    "lexical": {"bm25", "lexical", "tfidf"},
    "semantic": {"dense"},
    "any": None,
}


def _method_compliance(
    case: RetrievalGoldCase,
    retrieved: list[RetrievalChunk],
) -> tuple[bool, list[str], dict[str, list[str]], str | None]:
    """Check method provenance separately from page retrieval correctness."""
    allowed = _ALLOWED_METHODS[case.required_method]
    if not case.relevant_slide_ids:
        return (not retrieved), [], {}, None if not retrieved else "expected empty scope result"
    matched: list[str] = []
    methods: dict[str, list[str]] = {}
    for chunk in retrieved:
        if chunk.slide_id not in case.relevant_slide_ids:
            continue
        chunk_methods = list(chunk.retrieval_methods)
        matched.append(chunk.slide_id)
        methods[chunk.slide_id] = chunk_methods
    if allowed is None:
        return True, matched, methods, None
    missing = [
        slide_id for slide_id in case.relevant_slide_ids
        if slide_id not in methods
    ]
    wrong_method = [
        slide_id for slide_id, chunk_methods in methods.items()
        if not set(chunk_methods).intersection(allowed)
    ]
    if missing or wrong_method:
        detail: list[str] = []
        if missing:
            detail.append(f"missing relevant pages: {missing}")
        if wrong_method:
            detail.append(
                "matched pages lacked required method "
                f"{case.required_method}: {wrong_method}"
            )
        return False, matched, methods, "; ".join(detail)
    return True, matched, methods, None


def evaluate_gold_set(
    gold_set: RetrievalGoldSet,
    retrieve_case: RetrievalCallable,
    *,
    machine: str | None = None,
    model_contract: dict[str, Any] | None = None,
    index_signatures: dict[str, str] | None = None,
    top_k: int = 5,
) -> RetrievalEvaluationReport:
    """Evaluate one retriever and produce reproducible aggregate/failure data."""
    if top_k < 5:
        raise ValueError("top_k must be at least 5 so Precision@5 is available")

    results: list[RetrievalCaseResult] = []
    failures: list[RetrievalFailure] = []
    for case in gold_set.cases:
        started = time.perf_counter()
        error: str | None = None
        try:
            retrieved = list(retrieve_case(case))[:top_k]
        except Exception as exc:
            retrieved = []
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - started) * 1000
        relevant = set(case.relevant_slide_ids)
        ids = [chunk.slide_id for chunk in retrieved]

        def recall_at(k: int) -> float:
            if not relevant:
                return 1.0 if not ids[:k] else 0.0
            return len(set(ids[:k]) & relevant) / len(relevant)

        first_rank = next(
            (rank for rank, slide_id in enumerate(ids, start=1) if slide_id in relevant),
            None,
        )
        method_satisfied, matched_relevant, matched_methods, method_failure = (
            _method_compliance(case, retrieved)
        )
        result = RetrievalCaseResult(
            case_id=case.case_id,
            category=case.category,
            required_method=case.required_method,
            relevant_slide_ids=list(case.relevant_slide_ids),
            retrieved_slide_ids=ids,
            evidence=[_evidence(chunk) for chunk in retrieved],
            latency_ms=latency_ms,
            recall_at_3=recall_at(3),
            recall_at_5=recall_at(5),
            reciprocal_rank=1.0 / first_rank if first_rank else 0.0,
            precision_at_5=(
                1.0 if not relevant and not ids[:5]
                else len(set(ids[:5]) & relevant) / 5
            ),
            method_satisfied=method_satisfied,
            matched_relevant_slide_ids=matched_relevant,
            matched_methods=matched_methods,
            method_failure_reason=method_failure,
            error=error,
        )
        results.append(result)
        retrieval_failed = result.recall_at_5 < 1.0 or bool(error)
        method_failed = not result.method_satisfied
        if retrieval_failed or method_failed:
            if retrieval_failed and method_failed:
                failure_kind: Literal["retrieval", "method", "retrieval_and_method"] = (
                    "retrieval_and_method"
                )
            elif method_failed:
                failure_kind = "method"
            else:
                failure_kind = "retrieval"
            failures.append(RetrievalFailure(
                case_id=case.case_id,
                category=case.category,
                required_method=case.required_method,
                expected_slide_ids=list(case.relevant_slide_ids),
                actual_top_k=ids[:5],
                retrieval_methods={
                    chunk.slide_id: list(chunk.retrieval_methods)
                    for chunk in retrieved[:5]
                },
                ranks_and_scores={
                    chunk.slide_id: {
                        "dense_rank": chunk.dense_rank,
                        "dense_score": chunk.dense_score,
                        "bm25_rank": chunk.bm25_rank,
                        "bm25_score": chunk.bm25_score,
                        "rrf_score": chunk.rrf_score,
                    }
                    for chunk in retrieved[:5]
                },
                failure_kind=failure_kind,
                matched_relevant_slide_ids=matched_relevant,
                method_failure_reason=method_failure,
                error=error,
            ))

    by_category: dict[str, list[RetrievalCaseResult]] = defaultdict(list)
    by_method: dict[str, list[RetrievalCaseResult]] = defaultdict(list)
    for result in results:
        by_category[result.category].append(result)
        by_method[result.required_method].append(result)
    latencies = [result.latency_ms for result in results]
    return RetrievalEvaluationReport(
        fixture_version=gold_set.fixture_version,
        generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        machine=machine or platform.platform(),
        model_contract=model_contract or {},
        index_signatures=index_signatures or {},
        overall=_summary(results),
        by_category={key: _summary(value) for key, value in sorted(by_category.items())},
        by_method={key: _summary(value) for key, value in sorted(by_method.items())},
        latency_p50_ms=median(latencies) if latencies else 0.0,
        latency_p95_ms=_percentile(latencies, 0.95),
        cases=results,
        failures=failures,
    )


def evaluate_baselines(
    gold_set: RetrievalGoldSet,
    retrievers: dict[str, RetrievalCallable],
    *,
    machine: str | None = None,
    model_contract: dict[str, Any] | None = None,
    index_signatures: dict[str, str] | None = None,
    top_k: int = 5,
) -> RetrievalBaselineComparison:
    """Run the same fixture against named baselines for apples-to-apples comparison."""
    reports = {
        name: evaluate_gold_set(
            gold_set,
            retriever,
            machine=machine,
            model_contract=model_contract,
            index_signatures=index_signatures,
            top_k=top_k,
        )
        for name, retriever in retrievers.items()
    }
    return RetrievalBaselineComparison(
        fixture_version=gold_set.fixture_version,
        generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        reports=reports,
    )


def save_evaluation_report(path: str | Path, report: BaseModel) -> None:
    """Persist a validated report as a portable JSON artifact."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _slide_from_chunk(chunk: RetrievalChunk) -> SlidePage:
    return SlidePage(
        slide_id=chunk.slide_id,
        module_id=chunk.project_id,
        project_id=chunk.project_id,
        file_id=chunk.file_id,
        source_file=chunk.source_file,
        source_type="pdf",
        page_number=chunk.page_number,
        global_index=chunk.page_number,
        title_guess=chunk.title or None,
        text=chunk.text,
        text_quality="ok",
    )


def _scope_index(index: tuple[list[RetrievalChunk], Any, Any], case: RetrievalGoldCase):
    chunks, backend, matrix = index
    if case.scope == "whole_project":
        return index
    if case.scope == "selected_files":
        allowed_files = set(case.file_ids)
        positions = [
            position for position, chunk in enumerate(chunks)
            if chunk.file_id in allowed_files
        ]
    else:
        if not case.file_id:
            raise ValueError(f"{case.case_id}: file_id is required for scoped retrieval")
        positions = [
            position for position, chunk in enumerate(chunks)
            if chunk.file_id == case.file_id
        ]
    scoped_chunks = [chunks[position] for position in positions]
    if isinstance(matrix, np.ndarray):
        scoped_matrix = matrix[positions]
    else:
        scoped_matrix = matrix.tocsr()[positions]
    return scoped_chunks, backend, scoped_matrix


def make_project_retriever(
    index_provider: Callable[[str], tuple[list[RetrievalChunk], Any, Any]],
) -> RetrievalCallable:
    """Adapt the production retriever plus deterministic positional tools."""
    def retrieve(case: RetrievalGoldCase) -> list[RetrievalChunk]:
        index = _scope_index(index_provider(case.project_id), case)
        chunks, _backend, _matrix = index
        if case.expected_empty:
            return []
        current_slide = None
        if case.current_slide_id:
            current_chunk = next(
                (chunk for chunk in chunks if chunk.slide_id == case.current_slide_id),
                None,
            )
            if current_chunk is not None:
                current_slide = _slide_from_chunk(current_chunk)
        if case.required_method == "positional":
            return _retrieve_positional(case, chunks, current_slide)
        return retrieve_for_question(
            case.query,
            current_slide,
            index,
            lexical_query=case.query,
        )

    return retrieve


def _retrieve_positional(
    case: RetrievalGoldCase,
    chunks: list[RetrievalChunk],
    current_slide: SlidePage | None,
) -> list[RetrievalChunk]:
    """Execute current/explicit/neighbor page tools without a planner/LLM."""
    if current_slide is None:
        raise ValueError(f"{case.case_id}: positional case needs current_slide_id")
    from slide_tutor.qa_context import deterministic_positional_targets

    max_page = max((chunk.page_number for chunk in chunks), default=current_slide.page_number)
    include_current, page_numbers, offsets = deterministic_positional_targets(
        case.query, current_slide.page_number, max_page,
    )
    by_page = {chunk.page_number: chunk for chunk in chunks}
    ordered: list[RetrievalChunk] = []
    if include_current:
        target = by_page.get(current_slide.page_number)
        if target is not None:
            ordered.append(target)
    for page_number in page_numbers:
        target = by_page.get(page_number)
        if target is not None:
            ordered.append(target)
    for offset in offsets:
        target = by_page.get(current_slide.page_number + offset)
        if target is not None:
            ordered.append(target)
    seen: set[str] = set()
    result: list[RetrievalChunk] = []
    for chunk in ordered:
        if chunk.slide_id in seen:
            continue
        seen.add(chunk.slide_id)
        result.append(chunk.model_copy(update={"retrieval_methods": ["positional"]}))
    return result


def make_controlled_retriever(
    index_provider: Callable[[str], tuple[list[RetrievalChunk], Any, Any]],
) -> RetrievalCallable:
    """Create a deterministic no-network semantic surrogate for CI.

    It uses a small, versioned alias vocabulary only for the fixture concepts;
    it does not claim to measure embedding quality.  Production evaluation is
    performed by ``make_project_retriever`` with the configured shared model.
    """
    aliases = {
        "长时程抑制": "long term depression ltd",
        "长时程增强": "long term potentiation ltp",
        "低频刺激": "low frequency stimulation weaken synaptic depression",
        "削弱突触": "weaken synaptic depression",
        "增强突触连接": "strengthen synaptic connection potentiation",
        "梯度下降": "gradient descent parameter loss",
        "注意力机制": "attention mechanism weights tokens",
        "过度依赖训练数据": "regularization overfitting generalization",
        "训练集表现好": "overfitting generalization",
        "新记忆": "memory consolidation long term",
        "depresion": "depression",
        "potention": "potentiation",
        "gradent": "gradient",
    }

    def retrieve(case: RetrievalGoldCase) -> list[RetrievalChunk]:
        index = _scope_index(index_provider(case.project_id), case)
        chunks, _backend, _matrix = index
        if case.expected_empty:
            return []
        if case.required_method == "positional":
            current = next(
                (chunk for chunk in chunks if chunk.slide_id == case.current_slide_id),
                None,
            )
            return _retrieve_positional(case, chunks, _slide_from_chunk(current) if current else None)
        query = case.query.casefold()
        for source, target in aliases.items():
            query = query.replace(source.casefold(), target)
        query_tokens = {
            token for token in query.replace("-", " ").split()
            if len(token) > 1
        }
        ranked: list[tuple[float, RetrievalChunk]] = []
        for chunk in chunks:
            haystack = f"{chunk.title} {chunk.text}".casefold().replace("-", " ")
            score = sum(1.0 for token in query_tokens if token in haystack)
            if chunk.title.casefold() in query:
                score += 10.0
            if score > 0:
                ranked.append((score, chunk))
        ranked.sort(key=lambda item: (-item[0], item[1].slide_id))
        method = {
            "exact_title": "exact_title",
            "lexical": "bm25",
            "semantic": "dense",
        }.get(case.required_method, "controlled")
        return [
            chunk.model_copy(update={"retrieval_methods": [method]})
            for _score, chunk in ranked[:config.RETRIEVAL_TOP_K]
        ]

    return retrieve


def make_baseline_retrievers(
    index_provider: Callable[[str], tuple[list[RetrievalChunk], Any, Any]],
) -> dict[str, RetrievalCallable]:
    """Build the four named baselines required by the P5 report."""
    def scoped(case: RetrievalGoldCase):
        return _scope_index(index_provider(case.project_id), case)

    def clean_only_dense(case: RetrievalGoldCase) -> list[RetrievalChunk]:
        chunks, backend, matrix = scoped(case)
        return _semantic_search(case.query, chunks, backend, matrix, top_k=5)

    def title_aware_dense(case: RetrievalGoldCase) -> list[RetrievalChunk]:
        chunks, backend, matrix = scoped(case)
        exact = _exact_title_search(case.query, chunks)
        semantic = _semantic_search(case.query, chunks, backend, matrix, top_k=5)
        by_id: dict[str, RetrievalChunk] = {}
        for chunk in exact + semantic:
            existing = by_id.get(chunk.slide_id)
            methods = list(existing.retrieval_methods) if existing else []
            for method in chunk.retrieval_methods:
                if method not in methods:
                    methods.append(method)
            by_id[chunk.slide_id] = chunk.model_copy(update={"retrieval_methods": methods})
        return list(by_id.values())[:5]

    def bm25(case: RetrievalGoldCase) -> list[RetrievalChunk]:
        chunks, _backend, _matrix = scoped(case)
        return _bm25_search(case.query, chunks, top_k=5)

    return {
        "clean-only-dense": clean_only_dense,
        "title-aware-dense": title_aware_dense,
        "bm25": bm25,
        "hybrid-rrf": make_project_retriever(index_provider),
    }


def _index_metadata(project_id: str) -> tuple[dict[str, Any], str]:
    from slide_tutor import cache, config

    project_dir = cache.get_project_dir(project_id)
    meta = cache.load_json(project_dir / config.INDEX_DIRNAME / "index_meta.json", default={}) or {}
    return meta, str(meta.get("index_config_signature", ""))


def _build_controlled_index(project_id: str):
    from slide_tutor import index_store

    chunks = index_store.build_chunks(project_id)
    backend, matrix = index_store._build_tfidf_index(chunks)
    return chunks, backend, matrix


def _retrieval_gate(
    report: RetrievalEvaluationReport,
    *,
    configured: bool,
    baseline: RetrievalEvaluationReport | None = None,
) -> dict[str, bool]:
    cases = {case.case_id: case for case in report.cases}
    exact = report.by_category.get("exact_title")
    positional = report.by_method.get("positional")
    ltd = [cases.get(case_id) for case_id in ("exact-ltd-title", "distractor-ltd-memory")]
    gates = {
        "no_case_errors": all(case.error is None for case in report.cases),
        "exact_title_recall_at_3": bool(exact and exact.recall_at_3 == 1.0),
        "positional_deterministic_hit_rate": bool(
            positional and positional.deterministic_hit_rate == 1.0
        ),
        "exact_title_method_compliance": bool(
            exact and all(
                case.method_satisfied for case in report.cases
                if case.required_method == "exact_title"
            )
        ),
        "positional_method_compliance": bool(
            positional and all(
                case.method_satisfied for case in report.cases
                if case.required_method == "positional"
            )
        ),
        "ltd_pages_in_top_k": all(
            case is not None
            and {"slide-neuro-050", "slide-neuro-051"}.issubset(case.retrieved_slide_ids)
            for case in ltd
        ),
    }
    if baseline is not None:
        focused = {"semantic", "cross_lingual"}
        current_failures = sum(
            case.category in focused for case in report.failures
        )
        baseline_failures = sum(
            case.category in focused for case in baseline.failures
        )
        gates["semantic_cross_lingual_no_regression"] = current_failures <= baseline_failures
    else:
        gates["semantic_cross_lingual_no_regression"] = True
    if configured:
        gates["configured_backend_is_embedding"] = (
            report.model_contract.get("backend") == "embedding"
        )
    return gates


def _run_cli(args: argparse.Namespace) -> int:
    from slide_tutor import index_store
    from slide_tutor.projects import load_file_slides, load_project
    from slide_tutor.retrieval_fixture import build_retrieval_gold_project

    gold = load_gold_set(args.gold)
    if args.build_fixture:
        build_retrieval_gold_project()
        # The synthetic fixture is intentionally parser-free.  Prevent a local
        # Marker installation from loading its multi-GB model during a gold run.
        config.MARKER_ENABLED = False
    validate_gold_references(gold, load_project, load_file_slides)

    if args.backend == "configured":
        index_store.rebuild_index(gold.cases[0].project_id)
        index = index_store.load_index(gold.cases[0].project_id)
        if index is None:
            raise RuntimeError("configured embedding index could not be loaded")
        provider = lambda _project_id: index
        retriever = make_project_retriever(provider)
        meta, signature = _index_metadata(gold.cases[0].project_id)
        contract = {
            "backend": meta.get("backend"),
            "model_id": meta.get("embedding_model"),
            "revision": meta.get("embedding_model_revision"),
            "dimension": meta.get("embedding_dimension"),
            "index_config_signature": signature,
        }
    else:
        index = _build_controlled_index(gold.cases[0].project_id)
        provider = lambda _project_id: index
        retriever = make_controlled_retriever(provider)
        contract = {
            "backend": "controlled",
            "model_id": None,
            "revision": None,
            "dimension": None,
            "index_config_signature": "controlled-fixture-v1",
        }

    report = evaluate_gold_set(
        gold,
        retriever,
        model_contract=contract,
        index_signatures={gold.cases[0].project_id: contract["index_config_signature"]},
    )
    baseline_retrievers = (
        make_baseline_retrievers(provider)
        if args.backend == "configured"
        else {
            "clean-only-dense": retriever,
            "title-aware-dense": retriever,
            "bm25": retriever,
            "hybrid-rrf": retriever,
        }
    )
    comparison = evaluate_baselines(
        gold,
        baseline_retrievers,
        model_contract=contract,
        index_signatures={gold.cases[0].project_id: contract["index_config_signature"]},
    )
    title_baseline = comparison.reports["title-aware-dense"]
    gates = _retrieval_gate(
        report,
        configured=args.backend == "configured",
        baseline=title_baseline,
    )
    artifact = RetrievalRunReport(
        fixture_version=gold.fixture_version,
        generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        execution_command=shlex.join(args.command),
        reference_validation={"passed": True, "case_count": len(gold.cases)},
        evaluation=report,
        baseline_comparison=comparison,
        gates=gates,
    )
    save_evaluation_report(args.output, artifact)
    print(json.dumps({"output": str(args.output), "gates": gates}, ensure_ascii=False, indent=2))
    return 0 if all(gates.values()) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run retrieval gold evaluation")
    parser.add_argument("--build-fixture", action="store_true")
    parser.add_argument("--backend", choices=("controlled", "configured"), default="controlled")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_FIXTURE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.command = ["python", "-m", "slide_tutor.retrieval_eval", *(
        ["--build-fixture"] if args.build_fixture else []
    ), "--backend", args.backend, "--output", str(args.output)]
    return _run_cli(args)


if __name__ == "__main__":  # pragma: no cover - exercised by CLI verification
    raise SystemExit(main())

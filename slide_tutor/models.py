from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, StrictBool, field_validator, model_validator

from slide_tutor.config import SCHEMA_VERSION


_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?:^|[\s:=\"'])/(?:Users|private|tmp|var|home|opt|etc|Volumes)(?:/|$)",
)


def _contains_absolute_path(value: str) -> bool:
    return bool(_ABSOLUTE_PATH_PATTERN.search(value))


IndexProgressStage = Literal[
    "queued",
    "build page content",
    "load shared model",
    "build chunks",
    "encode passages",
    "persist index",
    "activate index",
]


@dataclass(frozen=True)
class IndexProgress:
    """Shared page/index progress event containing observed runtime facts."""

    stage: IndexProgressStage
    status: Literal["started", "progress", "completed", "failed", "cancelled"]
    completed: int = 0
    total: int = 0
    reused: int = 0
    elapsed_seconds: float = 0.0
    backend: str | None = None
    dimension: int | None = None
    fallback_reason: str = ""
    detail: str = ""


# ── Legacy Phase 1.0 models (module-based). Retained for old cache compatibility.
# New code should use ProjectManifest / ProjectFile / FileSlidesPayload below.


class SourceFile(BaseModel):
    file_name: str = Field(min_length=1)
    safe_name: str = Field(min_length=1)
    sha256: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    page_count: int | None = Field(default=None, ge=0)


class ProjectFile(BaseModel):
    file_id: str = Field(min_length=1)
    original_name: str = Field(min_length=1)
    safe_name: str = Field(min_length=1)
    sha256: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    page_count: int = Field(ge=0)
    created_at: str = Field(min_length=1)
    status: Literal["processing", "ready", "failed"]
    error_message: str | None = None
    source_type: str = "pdf"  # "pdf" | "pptx" — default "pdf" for old projects


class ProjectManifest(BaseModel):
    schema_version: str = SCHEMA_VERSION
    project_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    updated_at: str = Field(min_length=1)
    file_order: list[str] = Field(default_factory=list)
    files: list[ProjectFile] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Project name cannot be empty")
        return normalized


# Deprecated: ModuleManifest is the legacy Phase 1.0 structure.
# New code should use ProjectManifest from the project-first API.


class ModuleManifest(BaseModel):
    schema_version: str = SCHEMA_VERSION
    module_id: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    source_files: list[SourceFile] = Field(default_factory=list)
    slide_count: int = Field(ge=0)
    embedding_model: str | None = None
    vectorstore_backend: str | None = None
    status: Literal["processing", "ready", "failed"]
    error_message: str | None = None


class SlidePage(BaseModel):
    slide_id: str = Field(min_length=1)
    module_id: str = Field(min_length=1)
    project_id: str | None = None
    file_id: str | None = None
    source_file: str = Field(min_length=1)
    source_type: Literal["pdf", "pptx"]
    page_number: int = Field(ge=1)
    global_index: int = Field(ge=1)
    title_guess: str | None = None
    text: str = ""
    image_path: str | None = None
    thumbnail_path: str | None = None
    text_quality: Literal["ok", "empty", "low"]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("title_guess")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


# Deprecated: SlidesPayload flattens all PDF pages into one global list.
# New code should use FileSlidesPayload (per-file slides) via slide_tutor.projects.


class SlidesPayload(BaseModel):
    schema_version: str = SCHEMA_VERSION
    module_id: str = Field(min_length=1)
    slides: list[SlidePage] = Field(default_factory=list)


class FileSlidesPayload(BaseModel):
    schema_version: str = SCHEMA_VERSION
    project_id: str = Field(min_length=1)
    file_id: str = Field(min_length=1)
    slides: list[SlidePage] = Field(default_factory=list)


# ── Phase 2 models ──


class Citation(BaseModel):
    """A reference to a source slide, included in LLM answers so students can
    trace claims back to the original course material."""
    source_file: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    snippet_reference: str = Field(min_length=1)
    slide_id: str | None = None
    # Reviewed questions retain the stable evidence identity used to create
    # this display citation; legacy citations may leave the fields empty.
    project_id: str | None = None
    file_id: str | None = None
    evidence_id: str | None = None
    page_quality: PageQuality | None = None
    evidence_eligibility: EvidenceEligibility | None = None
    retrieval_provenance: list[str] = Field(default_factory=list)


PageRouteAction = Literal[
    "select", "supplement", "sparse_valid", "needs_vlm", "empty",
]
PageQuality = Literal[
    "ok", "degraded", "needs_vlm", "empty", "sparse_valid", "fallback", "failed",
]
EvidenceEligibility = Literal[
    "eligible", "degraded", "needs_vlm_only", "ineligible",
]
GenerationStatus = Literal["current", "stale", "missing", "building", "failed"]
GenerationLifecycleStatus = Literal[
    "baseline_building",
    "baseline_current",
    "enhancing",
    "enhanced_current",
    "enhancement_failed",
    "stale",
]
GenerationLayerStatus = Literal["missing", "building", "current", "stale", "failed"]
GenerationLayer = Literal["baseline", "enhanced"]


class PageGenerationMetadata(BaseModel):
    """Active canonical-content generation contract."""

    generation_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    source_signature: str = ""
    page_content_contract_signature: str = Field(min_length=1)
    # Missing on legacy generations. An empty identity remains readable but
    # is never considered current, preventing cross-version cache promotion.
    parser_identity: dict[str, str] = Field(default_factory=dict)
    page_count: int = Field(ge=0)
    page_manifest_signature: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    storage_root: str = ""
    status: Literal["current", "failed"] = "current"
    error: str | None = None


class PageGenerationIntegrity(BaseModel):
    """Read-only validation result for an active canonical page generation."""

    status: Literal["missing", "current", "stale", "corrupt"]
    reason: str = ""
    expected_count: int = Field(ge=0)
    valid_count: int = Field(ge=0)


class GenerationLifecycle(BaseModel):
    """Persisted baseline/enhanced state for one coherent project snapshot."""

    schema_version: str = "generation-lifecycle-v1"
    project_id: str = Field(min_length=1)
    source_signature: str = ""
    lifecycle_status: GenerationLifecycleStatus = "stale"
    active_layer: GenerationLayer = "baseline"
    baseline_status: GenerationLayerStatus = "missing"
    enhanced_status: GenerationLayerStatus = "missing"
    baseline_page_generation_id: str | None = None
    baseline_index_generation_id: str | None = None
    baseline_source_signature: str = ""
    enhanced_page_generation_id: str | None = None
    enhanced_index_generation_id: str | None = None
    enhanced_source_signature: str = ""
    started_at: str | None = None
    updated_at: str = ""
    error: str | None = None


PageTextSourceState = Literal[
    "current_generation", "legacy_cache", "ephemeral_degraded", "slide_fallback",
]


class PageTextResolution(BaseModel):
    """Canonical prompt text together with an honest cache/source state."""

    text: str = Field(min_length=1)
    quality: PageQuality
    source_state: PageTextSourceState
    warning: str = ""


class GenerationState(BaseModel):
    """Validated summary used by UI/status helpers for both cache layers."""

    page_status: GenerationStatus = "missing"
    index_status: GenerationStatus = "missing"
    active_generation_id: str | None = None
    active_index_generation_id: str | None = None
    source_signature: str = ""
    page_content_contract_signature: str = ""
    retrieval_contract_signature: str = ""
    detail: str = ""
    lifecycle_status: GenerationLifecycleStatus = "stale"
    active_layer: GenerationLayer = "baseline"
    baseline_status: GenerationLayerStatus = "missing"
    enhanced_status: GenerationLayerStatus = "missing"
    baseline_page_generation_id: str | None = None
    baseline_index_generation_id: str | None = None
    enhanced_page_generation_id: str | None = None
    enhanced_index_generation_id: str | None = None
    enhancement_error: str | None = None
PAGE_CONTENT_CONTRACT_VERSION = "page-content-v3"
PAGE_ROUTER_CONTRACT_VERSION = "page-router-v1"


class RetrievalChunk(BaseModel):
    """A text chunk for vector retrieval. One chunk per slide page (pages are
    short enough that page-level granularity works well for search)."""
    chunk_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    file_id: str = Field(min_length=1)
    slide_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    title: str = ""
    normalized_title: str = ""
    text: str = Field(min_length=1)
    snippet_reference: str = ""  # first ~150 chars of text for citations
    # PIO-9 stable content identities.  Empty values remain valid for legacy
    # persisted chunks and are deterministically backfilled when rebuilding.
    chunk_key: str = ""
    embedding_key: str = ""
    retrieval_methods: list[str] = Field(default_factory=list)
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    page_quality: PageQuality = "ok"
    route_action: PageRouteAction = "select"
    selected_provider: str = "pymupdf"
    evidence_eligibility: EvidenceEligibility = "eligible"
    page_content_contract_version: str = PAGE_CONTENT_CONTRACT_VERSION
    quality_warnings: list[str] = Field(default_factory=list)
    generation_id: str | None = None


# ── Phase 2 Advance models ──


class PageQualitySignals(BaseModel):
    """Deterministic, unit-bearing signals used by the page-content router."""

    pymupdf_char_count: int = Field(default=0, ge=0)
    marker_char_count: int = Field(default=0, ge=0)
    pymupdf_word_count: int = Field(default=0, ge=0)
    marker_word_count: int = Field(default=0, ge=0)
    pymupdf_nonempty_lines: int = Field(default=0, ge=0)
    marker_nonempty_lines: int = Field(default=0, ge=0)
    marker_to_pymupdf_ratio: float = Field(default=0.0, ge=0)
    marker_loss_ratio: float = Field(default=0.0, ge=0, le=1)
    title_retained_by_pymupdf: bool = False
    title_retained_by_marker: bool = False
    pymupdf_formula_count: int = Field(default=0, ge=0)
    marker_formula_count: int = Field(default=0, ge=0)
    pymupdf_table_line_count: int = Field(default=0, ge=0)
    marker_table_line_count: int = Field(default=0, ge=0)
    pymupdf_list_count: int = Field(default=0, ge=0)
    marker_list_count: int = Field(default=0, ge=0)
    pymupdf_span_count: int = Field(default=0, ge=0)
    visual_ratio: float = Field(default=0.0, ge=0, le=1)
    repeated_noise_ratio: float = Field(default=0.0, ge=0, le=1)
    orphan_fragment_ratio: float = Field(default=0.0, ge=0, le=1)


class PageUnderstanding(BaseModel):
    """Versioned candidates and canonical content for one slide page.

    ``clean_markdown``/``raw_text``/``parser_provider`` remain as a compatibility
    view for old consumers.  New consumers should use ``canonical_markdown``
    and the explicit candidate/provenance fields.
    """
    schema_version: str = SCHEMA_VERSION
    project_id: str = Field(min_length=1)
    file_id: str = Field(min_length=1)
    slide_id: str = Field(min_length=1)
    clean_markdown: str = ""          # Marker output or cleaned PyMuPDF text
    raw_text: str = ""                # original PyMuPDF text for traceability
    parser_provider: str = "pymupdf"  # "marker" or "pymupdf"
    parser_version: str | None = None
    quality: PageQuality = "ok"
    warnings: list[str] = Field(default_factory=list)
    created_at: str = ""
    canonical_markdown: str = ""
    selected_provider: Literal["marker", "pymupdf", "vlm", "none"] = "pymupdf"
    pymupdf_text: str = ""
    marker_markdown: str = ""
    quality_signals: PageQualitySignals = Field(default_factory=PageQualitySignals)
    route_action: PageRouteAction = "select"
    page_content_contract_version: str = PAGE_CONTENT_CONTRACT_VERSION
    router_contract_version: str = PAGE_ROUTER_CONTRACT_VERSION
    generation_id: str | None = None

    @model_validator(mode="after")
    def migrate_legacy_content_fields(self) -> "PageUnderstanding":
        """Deterministically hydrate the new view when reading old cache JSON."""
        if not self.canonical_markdown:
            self.canonical_markdown = self.clean_markdown
        if not self.pymupdf_text:
            self.pymupdf_text = self.raw_text
        if self.selected_provider == "pymupdf" and self.parser_provider in {
            "marker", "vlm", "none",
        }:
            self.selected_provider = self.parser_provider  # type: ignore[assignment]
        if self.quality == "fallback" and self.route_action == "select":
            self.route_action = "select"
        return self


# ── Phase 4 Advance: Q&A memory v2 (Stage A) ──


class QAMemoryTurn(BaseModel):
    """One Q&A round in the short-term conversation memory.  Stored as a
    question+answer pair so each round is one entry (not user+assistant
    separately).  Keeps the last 5 rounds; cleared on project switch but
    preserved across page changes within the same project."""
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    file_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    source_file: str = Field(min_length=1)
    language: str = Field(min_length=1)


QAActionType = Literal["explain", "translate", "ask"]
QATaskStatus = Literal["pending", "running", "completed", "cancelled", "failed"]


class QAContextPlan(BaseModel):
    """Bounded context planner output — the planner decides which context
    sources to load before the answer model runs.  It does NOT produce facts;
    it only selects context tools from a fixed whitelist.

    All fields are optional with safe defaults so a planner failure → empty
    plan still produces a valid fallback via _default_plan()."""
    include_current_page: bool = True
    page_numbers: list[int] = Field(default_factory=list)
    page_offsets: list[int] = Field(default_factory=list)
    semantic_query: str = ""
    use_memory: bool = True
    ambiguity_note: str | None = None


class ToolCall(BaseModel):
    """Provider-neutral normalized tool call; SDK response objects never cross this boundary."""

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMCallMetrics(BaseModel):
    """Observed transport facts for one LLM request.

    Streaming fields are optional so old persisted metrics and non-streaming
    callers remain backward compatible.
    """

    operation: Literal["text", "tool_calls", "agent_turn", "json", "stream"]
    model: str = Field(min_length=1)
    attempt: int = Field(default=1, ge=1)
    status: Literal["success", "error", "cancelled"]
    latency_ms: float = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    first_token_latency_ms: float | None = Field(default=None, ge=0)
    streamed: bool = False
    error_type: str | None = None


class ToolTrace(BaseModel):
    """Safe audit summary for one accepted/rejected tool selection."""

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    status: Literal["accepted", "executed", "rejected", "deduplicated"]
    arguments: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    error: str | None = None


class QAContextPlanningResult(BaseModel):
    """Planner decision plus safe provenance before context assembly."""

    plan: QAContextPlan
    mode: Literal[
        "json_planner",
        "function_calling",
        "function_calling_fallback",
    ]
    tool_traces: list[ToolTrace] = Field(default_factory=list)
    fallback_reason: str | None = None


class QAContextSource(BaseModel):
    """One context source used to answer the current question.  Displayed in
    the 'Context used' expander so the student can trace the answer back to
    specific slides and conversation turns."""
    kind: Literal[
        "selected_page",
        "explicit_page",
        "nearby_page",
        "retrieved_page",
        "memory",
    ]
    source_file: str | None = None
    page_number: int | None = None
    title: str | None = None
    reason: str = ""
    snippet: str = ""
    retrieval_methods: list[str] = Field(default_factory=list)


class ToolSourceRef(BaseModel):
    """Provider-neutral source reference; Q&A converts applicable slide refs later."""

    kind: str = Field(min_length=1)
    ref_id: str | None = None
    source_file: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentMessage(BaseModel):
    """Provider-neutral message kept in memory during one bounded run."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "AgentMessage":
        if self.role == "tool":
            if not self.tool_call_id:
                raise ValueError("tool message requires tool_call_id")
            if self.tool_calls:
                raise ValueError("tool message cannot carry tool_calls")
            if _contains_absolute_path(self.content):
                raise ValueError("tool message cannot contain an absolute path")
        elif self.role in {"system", "user"}:
            if self.tool_calls:
                raise ValueError(f"{self.role} message cannot carry tool_calls")
            if self.tool_call_id:
                raise ValueError(f"{self.role} message cannot carry tool_call_id")
        elif self.tool_call_id:
            raise ValueError("assistant message cannot carry tool_call_id")
        return self


class AgentModelTurn(BaseModel):
    """Normalized response from one Agent controller request."""

    tool_calls: list[ToolCall] = Field(default_factory=list)
    assistant_text: str = ""
    provider_finish_reason: str | None = None

    @model_validator(mode="after")
    def validate_actions(self) -> "AgentModelTurn":
        finish_calls = [call for call in self.tool_calls if call.name == "finish_answer"]
        if len(finish_calls) > 1:
            raise ValueError("Agent turn cannot contain multiple finish_answer calls")
        if finish_calls and len(self.tool_calls) != 1:
            raise ValueError("finish_answer cannot be mixed with read tool calls")
        return self


class AgentObservation(BaseModel):
    """Bounded safe observation returned for exactly one Agent call."""

    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    status: Literal["executed", "deduplicated", "error", "no_new_context"]
    content_for_model: str = ""
    sources: list[ToolSourceRef] = Field(default_factory=list)
    truncated: bool = False
    error: str | None = None

    @model_validator(mode="after")
    def validate_status(self) -> "AgentObservation":
        if self.status == "error" and not (self.error or "").strip():
            raise ValueError("error observation requires error")
        if self.status != "error" and self.error:
            raise ValueError("non-error observation cannot carry error")
        if self.tool_name == "finish_answer":
            raise ValueError("finish_answer does not produce an observation")
        return self


class AgentRoundTrace(BaseModel):
    """Sanitized trace summary; never stores prompts or observation prose."""

    round_number: int = Field(ge=1)
    tool_traces: list[ToolTrace] = Field(default_factory=list)
    observation_source_count: int = Field(default=0, ge=0)
    observation_chars: int = Field(default=0, ge=0)
    controller_metrics: LLMCallMetrics | None = None
    elapsed_ms: float = Field(ge=0)


AgentStopReason = Literal[
    "finish_requested",
    "max_rounds",
    "max_total_tool_calls",
    "token_budget",
    "time_budget",
    "provider_error_after_progress",
    "protocol_error_after_progress",
    "tool_exhausted",
    "fallback_v1",
]


class AgentRunSummary(BaseModel):
    """Sanitized run-level provenance for UI and future answer records."""

    mode: Literal["bounded_agent_v2", "bounded_agent_v2_fallback"]
    rounds: int = Field(ge=0)
    executed_tool_calls: int = Field(ge=0)
    baseline_source_count: int = Field(ge=0)
    stop_reason: AgentStopReason
    round_traces: list[AgentRoundTrace] = Field(default_factory=list)
    usage_status: Literal["exact", "partial", "unavailable"]
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    elapsed_ms: float = Field(ge=0)
    fallback_reason: str | None = None

    @model_validator(mode="after")
    def validate_usage_shape(self) -> "AgentRunSummary":
        token_values = (
            self.prompt_tokens,
            self.completion_tokens,
            self.total_tokens,
        )
        populated = sum(value is not None for value in token_values)
        if self.usage_status == "exact" and populated != 3:
            raise ValueError("exact usage requires all token fields")
        if self.usage_status == "unavailable" and populated:
            raise ValueError("unavailable usage cannot carry token fields")
        if self.usage_status == "partial" and populated in {0, 3}:
            raise ValueError("partial usage requires some, but not all, token fields")
        return self


class QAContextBundle(BaseModel):
    """Assembled context for the answer model — sectioned prompt text plus
    transparent source metadata for the UI."""
    prompt_context: str = ""
    sources: list[QAContextSource] = Field(default_factory=list)
    memory_turns: list[QAMemoryTurn] = Field(default_factory=list)
    planner_summary: str | None = None
    planner_mode: Literal[
        "deterministic",
        "json_planner",
        "function_calling",
        "function_calling_fallback",
        "bounded_agent_v2",
        "bounded_agent_v2_fallback",
    ] | None = None
    tool_traces: list[ToolTrace] = Field(default_factory=list)
    agent_summary: AgentRunSummary | None = None


class QAAnswerRecord(BaseModel):
    """One visible answer; deliberately separate from short-term memory."""

    record_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    action_type: QAActionType
    status: QATaskStatus
    project_id: str = Field(min_length=1)
    file_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    language: str = Field(min_length=1)
    question: str = ""
    answer_markdown: str = ""
    context_bundle: QAContextBundle | None = None
    created_at: str = Field(min_length=1)
    completed_at: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_status_invariants(self) -> "QAAnswerRecord":
        if self.status == "completed" and not self.answer_markdown.strip():
            raise ValueError("completed QAAnswerRecord requires answer_markdown")
        if self.status == "failed" and not (self.error or "").strip():
            raise ValueError("failed QAAnswerRecord requires error")
        if self.status == "cancelled" and self.error:
            raise ValueError("cancelled QAAnswerRecord cannot carry provider error")
        return self


class QAAnswerHistoryBundle(BaseModel):
    """Project-first durable history; current output and memory stay session-only."""

    schema_version: str = "qa-answer-history-v1"
    project_id: str = Field(min_length=1)
    records: list[QAAnswerRecord] = Field(default_factory=list)
    updated_at: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_project_scope(self) -> "QAAnswerHistoryBundle":
        if any(record.project_id != self.project_id for record in self.records):
            raise ValueError("QA answer history cannot contain another project's records")
        ids = [record.record_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("QA answer history record ids must be unique")
        return self


class QACurrentResponse(BaseModel):
    """Project-scoped live/terminal output, independent from answer history."""

    task_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    file_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    action_type: QAActionType
    language: str = Field(min_length=1)
    question: str = ""
    status: QATaskStatus
    answer_markdown: str = ""
    context_bundle: QAContextBundle | None = None
    quality_warning: str | None = None
    error: str | None = None
    created_at: str = Field(min_length=1)
    completed_at: str | None = None


AnswerHistoryScope = Literal["current_page", "current_file", "whole_project"]


class QAAnswerPageGroup(BaseModel):
    """User-facing projection for one complete page identity."""

    project_id: str = Field(min_length=1)
    file_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    latest_activity_at: str = Field(min_length=1)
    explain: QAAnswerRecord | None = None
    translations: list[QAAnswerRecord] = Field(default_factory=list)
    asks: list[QAAnswerRecord] = Field(default_factory=list)
    previous_versions: list[QAAnswerRecord] = Field(default_factory=list)
    recent_tasks: list[QAAnswerRecord] = Field(default_factory=list)


class QAAnswerProjection(BaseModel):
    """Read-only display projection; raw audit records remain unchanged."""

    scope: AnswerHistoryScope
    groups: list[QAAnswerPageGroup] = Field(default_factory=list)
    project_record_count: int = Field(ge=0)
    visible_record_count: int = Field(ge=0)


RetrievalScopeKind = Literal[
    "current_page", "current_file", "selected_files", "whole_project", "topic",
]


class RetrievalScope(BaseModel):
    """Explicit retrieval boundary shared by Q&A and future Stage C callers."""

    kind: RetrievalScopeKind
    project_id: str = Field(min_length=1)
    slide_id: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    file_id: str | None = None
    file_ids: list[str] = Field(default_factory=list)
    topic_query: str | None = None

    @model_validator(mode="after")
    def validate_scope(self) -> "RetrievalScope":
        if self.kind == "current_page" and (not self.slide_id or self.page_number is None):
            raise ValueError("current_page scope requires slide_id and page_number")
        if self.kind == "current_file" and not self.file_id:
            raise ValueError("current_file scope requires file_id")
        if self.kind == "selected_files":
            if not self.file_ids or len(set(self.file_ids)) != len(self.file_ids):
                raise ValueError("selected_files scope requires unique non-empty file_ids")
        if self.kind == "whole_project" and (self.file_id or self.file_ids or self.slide_id):
            raise ValueError("whole_project scope cannot carry file/page filters")
        if self.kind == "topic" and not (self.topic_query or "").strip():
            raise ValueError("topic scope requires topic_query")
        if self.file_id and self.file_ids:
            raise ValueError("scope cannot carry both file_id and file_ids")
        return self

    @property
    def summary(self) -> str:
        if self.kind == "current_page":
            return f"current_page:{self.file_id or '*'}:{self.page_number}:{self.slide_id}"
        if self.kind == "current_file":
            return f"current_file:{self.file_id}"
        if self.kind == "selected_files":
            return f"selected_files:{','.join(self.file_ids)}"
        if self.kind == "topic":
            files = ",".join(self.file_ids) if self.file_ids else "*"
            return f"topic:{files}:{self.topic_query}"
        return "whole_project"


class ToolExecutionContext(BaseModel):
    """Trusted execution scope injected by the application, never by the model."""

    project_id: str = Field(min_length=1)
    current_file_id: str | None = None
    current_page_number: int | None = Field(default=None, ge=1)
    max_page: int | None = Field(default=None, ge=1)
    retrieval_scope: RetrievalScope | None = None
    allowed_file_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_scope_identity(self) -> "ToolExecutionContext":
        if (
            self.current_file_id
            and self.allowed_file_ids
            and self.current_file_id not in self.allowed_file_ids
        ):
            raise ValueError("current_file_id must be in allowed_file_ids")
        if self.retrieval_scope and self.retrieval_scope.project_id != self.project_id:
            raise ValueError("retrieval_scope project must match execution project")
        return self


class ToolResult(BaseModel):
    """Generic handler result reusable by Q&A, learning-state tools, and agents."""

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    status: Literal["success", "error"]
    content_for_model: str = ""
    structured_data: dict[str, Any] = Field(default_factory=dict)
    sources: list[ToolSourceRef] = Field(default_factory=list)
    error: str | None = None
    truncated: bool = False

    @model_validator(mode="after")
    def validate_error_state(self) -> "ToolResult":
        if self.status == "error" and not (self.error or "").strip():
            raise ValueError("error ToolResult requires error")
        if self.status == "success" and self.error:
            raise ValueError("success ToolResult cannot carry error")
        return self


QuestionType = Literal["mcq", "true_false", "fill_blank", "short_answer"]
QuestionDifficulty = Literal["basic", "applied"]


class EvidenceRef(BaseModel):
    """A page-addressable evidence pointer; no free-form citation-only target."""

    project_id: str = Field(min_length=1)
    file_id: str = Field(min_length=1)
    slide_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    snippet: str = ""
    retrieval_methods: list[str] = Field(default_factory=list)
    page_quality: PageQuality = "ok"
    evidence_eligibility: EvidenceEligibility = "eligible"

    @property
    def evidence_id(self) -> str:
        """Expose a deterministic page identity for prompt/model hand-off."""
        return f"{self.file_id}:{self.slide_id}"


class EvidenceQualitySummary(BaseModel):
    status: EvidenceEligibility
    warnings: list[str] = Field(default_factory=list)
    blocking: bool = False

    @model_validator(mode="after")
    def validate_blocking(self) -> "EvidenceQualitySummary":
        if self.status == "needs_vlm_only" and not self.blocking:
            raise ValueError("needs_vlm_only evidence must be blocking")
        if self.status == "ineligible" and not self.blocking:
            raise ValueError("ineligible evidence must be blocking")
        return self


class PlannedQuestionSlot(BaseModel):
    """Stage C planning boundary; deliberately contains no page evidence."""

    slot_id: str = Field(min_length=1)
    file_ids: list[str] = Field(default_factory=list)
    topic_hint: str = Field(default="topic", min_length=1)
    retrieval_query: str = Field(min_length=1)
    difficulty: QuestionDifficulty
    question_type: QuestionType
    coverage_role: Literal["single_file", "cross_file"] = "single_file"

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_scope(cls, value: Any) -> Any:
        """Read the pre-Stage-C single-file cache without preserving its shape."""
        if isinstance(value, dict):
            payload = dict(value)
            if not payload.get("file_ids") and payload.get("file_id"):
                payload["file_ids"] = [payload["file_id"]]
            if not payload.get("topic_hint") and payload.get("topic"):
                payload["topic_hint"] = payload["topic"]
            return payload
        return value

    @model_validator(mode="after")
    def validate_scope(self) -> "PlannedQuestionSlot":
        """Make the planner scope explicit before any evidence is retrieved."""
        if not self.file_ids or len(set(self.file_ids)) != len(self.file_ids):
            raise ValueError("PlannedQuestionSlot requires unique non-empty file_ids")
        if self.coverage_role == "single_file" and len(self.file_ids) != 1:
            raise ValueError("single_file slot must name exactly one file")
        if self.coverage_role == "cross_file" and len(self.file_ids) < 2:
            raise ValueError("cross_file slot must name at least two files")
        return self

    @property
    def file_id(self) -> str | None:
        """Compatibility view for old callers; serialization uses file_ids."""
        return self.file_ids[0] if len(self.file_ids) == 1 else None

    @property
    def topic(self) -> str:
        """Compatibility view for old callers; new code uses topic_hint."""
        return self.topic_hint


class BoundQuestionSlot(BaseModel):
    """Planned slot after evidence binding, without implementing binding itself."""

    slot_id: str = Field(min_length=1)
    file_ids: list[str] = Field(default_factory=list)
    topic_hint: str = Field(default="topic", min_length=1)
    retrieval_query: str = Field(min_length=1)
    difficulty: QuestionDifficulty
    question_type: QuestionType
    coverage_role: Literal["single_file", "cross_file"] = "single_file"
    evidence_refs: list[EvidenceRef] = Field(min_length=1)
    evidence_summary: str = Field(min_length=1)
    evidence_quality: EvidenceQualitySummary

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_scope(cls, value: Any) -> Any:
        """Keep old Bound cache readable while enforcing the new scope contract."""
        if isinstance(value, dict):
            payload = dict(value)
            if not payload.get("file_ids") and payload.get("file_id"):
                payload["file_ids"] = [payload["file_id"]]
            if not payload.get("topic_hint") and payload.get("topic"):
                payload["topic_hint"] = payload["topic"]
            return payload
        return value

    @model_validator(mode="after")
    def validate_scope(self) -> "BoundQuestionSlot":
        """Ensure binding cannot silently widen or collapse planner scope."""
        if not self.file_ids or len(set(self.file_ids)) != len(self.file_ids):
            raise ValueError("BoundQuestionSlot requires unique non-empty file_ids")
        if self.coverage_role == "single_file" and len(self.file_ids) != 1:
            raise ValueError("single_file slot must name exactly one file")
        if self.coverage_role == "cross_file" and len(self.file_ids) < 2:
            raise ValueError("cross_file slot must name at least two files")
        return self

    @model_validator(mode="after")
    def validate_bound_evidence(self) -> "BoundQuestionSlot":
        if self.evidence_quality.blocking:
            raise ValueError("blocking evidence cannot produce a valid BoundQuestionSlot")
        if not self.evidence_refs:
            raise ValueError("BoundQuestionSlot requires evidence_refs")
        return self

    @property
    def file_id(self) -> str | None:
        """Compatibility view for the legacy single-file contract."""
        return self.file_ids[0] if len(self.file_ids) == 1 else None

    @property
    def topic(self) -> str:
        """Compatibility view for code that still renders a topic label."""
        return self.topic_hint


class ExamBlueprint(BaseModel):
    """Stage C planned blueprint schema only; no planner or LLM execution."""

    blueprint_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    target_count: int = Field(ge=1)
    language: str = "中文"
    question_type_mix: dict[QuestionType, int] = Field(default_factory=dict)
    slots: list[PlannedQuestionSlot] = Field(min_length=1)
    blueprint_schema_version: str = "planned-v1"
    source_signature: str = ""
    planning_contract_signature: str = ""
    page_content_contract_signature: str = ""
    retrieval_contract_signature: str = ""

    @model_validator(mode="after")
    def validate_slots(self) -> "ExamBlueprint":
        ids = [slot.slot_id for slot in self.slots]
        if len(ids) != len(set(ids)):
            raise ValueError("ExamBlueprint slot_id values must be unique")
        if len(self.slots) != self.target_count:
            raise ValueError("ExamBlueprint slot count must equal target_count")
        if self.question_type_mix and sum(self.question_type_mix.values()) != self.target_count:
            raise ValueError("question_type_mix must sum to target_count")
        return self


class BoundExamBlueprint(BaseModel):
    blueprint_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    target_count: int = Field(ge=1)
    language: str = "中文"
    slots: list[BoundQuestionSlot] = Field(min_length=1)
    blueprint_schema_version: str = "bound-v1"
    planned_blueprint_id: str = ""
    source_signature: str = ""
    planning_contract_signature: str = ""
    page_content_contract_signature: str = ""
    retrieval_contract_signature: str = ""

    @model_validator(mode="after")
    def validate_bound_slots(self) -> "BoundExamBlueprint":
        ids = [slot.slot_id for slot in self.slots]
        if len(ids) != len(set(ids)):
            raise ValueError("BoundExamBlueprint slot_id values must be unique")
        if len(self.slots) != self.target_count:
            raise ValueError("BoundExamBlueprint slot count must equal target_count")
        return self


class BindingIssue(BaseModel):
    """A truthful reason why one planned slot cannot become bound evidence."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    slot_id: str = Field(min_length=1)
    severity: Literal["blocking", "warning"] = "blocking"
    retryable: bool = False


class BindingOutcome(BaseModel):
    """Binding either returns a complete blueprint or explicit slot issues."""

    planned_blueprint_id: str = Field(min_length=1)
    bound_blueprint: BoundExamBlueprint | None = None
    issues: list[BindingIssue] = Field(default_factory=list)
    replanned_once: bool = False

    @model_validator(mode="after")
    def validate_result_shape(self) -> "BindingOutcome":
        if self.bound_blueprint is not None and self.issues:
            raise ValueError("A successful BindingOutcome cannot carry binding issues")
        if self.bound_blueprint is None and not self.issues:
            raise ValueError("An unsuccessful BindingOutcome requires binding issues")
        return self


class GenerationSnapshot(BaseModel):
    """Immutable page/index identity captured for one Reviewed request."""

    source_signature: str = ""
    active_layer: GenerationLayer = "baseline"
    page_generation_id: str | None = None
    index_generation_id: str | None = None
    page_content_contract_signature: str = ""
    retrieval_contract_signature: str = ""


class ReviewedExamMetadata(BaseModel):
    """Auditable Reviewed exam facts without prompts, hidden reasoning, or raw responses."""

    schema_version: str = "reviewed-exam-metadata-v1"
    quality_mode: Literal["fast", "reviewed"] = "reviewed"
    blueprint_schema_version: str = "planned-v2"
    planner_prompt_version: str = ""
    planner_model: str = ""
    generation_prompt_version: str = ""
    generation_model: str = ""
    review_prompt_version: str = ""
    review_model: str = ""
    source_signature: str = ""
    page_generation_id: str | None = None
    index_generation_id: str | None = None
    active_layer: GenerationLayer = "baseline"
    planned_blueprint: ExamBlueprint | None = None
    bound_blueprint: BoundExamBlueprint | None = None
    validation_summary: dict[str, Any] = Field(default_factory=dict)
    review_findings_summary: dict[str, Any] = Field(default_factory=dict)
    repair_summary: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""
    completed_at: str = ""


# ── Pre-index status (Phase 4 Advance) ──

PreindexStatus = Literal[
    "not_ready", "not_indexed", "lazy_current", "current", "stale",
]


# ── Phase 3 models ──

# Prompt version — bump when prompt templates change so cached QuizSets
# generated with old prompts are automatically invalidated.
QUIZ_PROMPT_VERSION = "v5"


def _require_option_keys(options: dict[str, str], expected: set[str], label: str) -> None:
    if set(options.keys()) != expected:
        raise ValueError(f"{label} options must have exactly keys {sorted(expected)}")
    if any(not value.strip() for value in options.values()):
        raise ValueError(f"{label} option text cannot be empty")


def _looks_like_true_false_pair(options: dict[str, str]) -> bool:
    true_values = {"true", "t", "yes", "correct", "对", "正确", "是", "真"}
    false_values = {"false", "f", "no", "incorrect", "错", "错误", "否", "假"}
    normalized = {str(value).strip().lower() for value in options.values()}
    return bool(normalized & true_values) and bool(normalized & false_values)


class QuizQuestion(BaseModel):
    """A quiz question — supported types: mcq, true_false, fill_blank, short_answer.
    Fields are validated conditionally based on question_type."""
    question_id: str = Field(min_length=1)
    question_origin_id: str = ""
    reused_from_attempt_id: str | None = None
    slot_id: str | None = None  # internal Reviewed generation alignment
    question_type: Literal["mcq", "true_false", "fill_blank", "short_answer"]
    topic_tag: str = Field(min_length=1)
    topic_key: str = ""
    topic_names: dict[str, str] = Field(default_factory=dict)
    difficulty: Literal["basic", "applied"]
    question: str = Field(min_length=1)
    # MCQ / True-False
    options: dict[str, str] = Field(default_factory=dict)
    correct_option: str = ""  # "A"/"B"/"C"/"D" for MCQ/TF
    # Fill-blank
    blank_answer: str = ""
    # Short-answer
    model_answer: str = ""
    scoring_rubric: str = ""
    # Common
    explanation: str = ""  # why the correct answer is correct
    distractor_rationales: dict[str, str] = Field(default_factory=dict)
    citations: list[Citation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_by_question_type(self) -> "QuizQuestion":
        """Strict Quiz v3 contract. Generated questions must declare a type."""
        if not self.citations:
            raise ValueError("citations cannot be empty")

        if self.question_type == "mcq":
            _require_option_keys(self.options, {"A", "B", "C", "D"}, "MCQ")
            if self.correct_option not in self.options:
                raise ValueError("MCQ correct_option must be one of A/B/C/D")
            return self

        if self.question_type == "true_false":
            _require_option_keys(self.options, {"A", "B"}, "True/False")
            if self.correct_option not in {"A", "B"}:
                raise ValueError("True/False correct_option must be A or B")
            if not _looks_like_true_false_pair(self.options):
                raise ValueError("True/False options must represent true/false values")
            return self

        if self.question_type == "fill_blank":
            if self.options:
                raise ValueError("Fill-blank options must be empty")
            if self.correct_option:
                raise ValueError("Fill-blank correct_option must be empty")
            if not self.blank_answer.strip():
                raise ValueError("Fill-blank blank_answer cannot be empty")
            return self

        if self.question_type == "short_answer":
            if self.options:
                raise ValueError("Short-answer options must be empty")
            if self.correct_option:
                raise ValueError("Short-answer correct_option must be empty")
            if not self.model_answer.strip():
                raise ValueError("Short-answer model_answer cannot be empty")
            if not self.scoring_rubric.strip():
                raise ValueError("Short-answer scoring_rubric cannot be empty")
            return self

        return self


class FillBlankGrade(BaseModel):
    """One AI-only binary fill-blank grading result."""

    question_id: str = Field(min_length=1)
    is_correct: StrictBool
    feedback: str = ""


class FillBlankGradeBatch(BaseModel):
    """Strict response contract for one batched fill-blank grading call."""

    grades: list[FillBlankGrade] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_question_ids(self) -> "FillBlankGradeBatch":
        ids = [grade.question_id for grade in self.grades]
        if len(ids) != len(set(ids)):
            raise ValueError("FillBlankGrade question_id values must be unique")
        return self


class ShortAnswerGrade(BaseModel):
    question_id: str = Field(min_length=1)
    points_awarded: float = Field(strict=True, ge=0, allow_inf_nan=False)
    grading_feedback: str = ""


class ShortAnswerGradeBatch(BaseModel):
    grades: list[ShortAnswerGrade]


class TopicCatalogEntry(BaseModel):
    """Bounded planner input: titles and short summaries, never full page text."""

    file_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    page_numbers: list[int] = Field(min_length=1)
    summary: str = Field(min_length=1, max_length=240)


class TopicRegistryEntry(BaseModel):
    """Stable project topic identity with localized display names."""

    topic_key: str = Field(min_length=1)
    names: dict[str, str] = Field(default_factory=dict)
    aliases: dict[str, list[str]] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    provisional: bool = False


class TopicRegistry(BaseModel):
    schema_version: str = "topic-registry-v1"
    project_id: str = Field(min_length=1)
    entries: list[TopicRegistryEntry] = Field(default_factory=list)


class QuizReuseSummary(BaseModel):
    reused_count: int = Field(default=0, ge=0)
    generated_count: int = Field(default=0, ge=0)
    source_attempt_ids: list[str] = Field(default_factory=list)


class QuizSet(BaseModel):
    """A complete quiz — pure questions, cacheable, no user data attached."""
    schema_version: str = SCHEMA_VERSION
    quiz_set_id: str = ""             # stable cache key (D10)
    project_id: str = Field(min_length=1)
    file_ids: list[str] = Field(default_factory=list)  # PDFs covered
    total_count: int = Field(ge=0)
    difficulty_mix: str = ""         # e.g. "4 basic + 2 applied"
    question_type_mix: str = Field(min_length=1)
    questions: list[QuizQuestion] = Field(default_factory=list)
    created_at: str = ""
    model: str = ""                  # LLM used for generation
    prompt_version: str = QUIZ_PROMPT_VERSION
    language: str = "中文"            # answer language at generation time
    quality_mode: Literal["fast", "reviewed"] = "fast"
    reviewed_metadata: ReviewedExamMetadata | None = None
    generation_request_signature: str = ""
    generation_snapshot: GenerationSnapshot | None = None
    reuse_summary: QuizReuseSummary = Field(default_factory=QuizReuseSummary)


class ReviewedExamBundle(BaseModel):
    """Single atomic activation unit for a Reviewed exam and its audit data."""

    schema_version: str = "reviewed-exam-bundle-v1"
    quiz_set: QuizSet
    metadata: ReviewedExamMetadata


class QuizAttempt(BaseModel):
    """One user attempt at a quiz — stored separately from the cached QuizSet."""
    attempt_id: str = Field(min_length=1)
    quiz_set_id: str = ""            # back-reference to the QuizSet
    quiz_type: Literal["per_pdf", "final_exam"] = "per_pdf"
    project_id: str = Field(min_length=1)
    file_ids: list[str] = Field(default_factory=list)
    total_count: int = Field(ge=0)
    answers: dict[str, str | None] = Field(default_factory=dict)
    submitted_at: str = ""


class QuizResultDetail(BaseModel):
    """Per-question result with optional wrong-reason for incorrect answers.
    Quiz v3 adds partial-credit fields for short-answer grading (4.7a)."""
    question_id: str = Field(min_length=1)
    question_type: Literal["mcq", "true_false", "fill_blank", "short_answer"] = "mcq"
    topic_tag: str = ""
    topic_key: str = ""
    topic_names: dict[str, str] = Field(default_factory=dict)
    question: str = ""
    user_answer: str | None = None        # "A"/"B"/"C"/"D"/None, or free text
    correct_option: str = ""
    is_correct: bool = False
    explanation: str = ""                 # generic explanation from the QuizQuestion
    wrong_reason: str | None = None       # targeted at user's specific wrong answer
    citations: list[Citation] = Field(default_factory=list)
    # Phase 4 Quiz v3: partial credit support
    points_awarded: float | None = None   # 1.0, 2.5 etc; None = not graded yet
    max_points: float = 1.0
    grading_feedback: str | None = None   # AI 批语（简答题）


class QuizResult(BaseModel):
    """Aggregated result of one quiz attempt, independent of the QuizSet.
    Quiz v3: score is float for partial-credit questions; max_score is optional."""
    attempt_id: str = ""
    score: float = Field(ge=0)
    total: int = Field(ge=0)            # question count (legacy compat)
    max_score: float | None = None       # sum(max_points), available after Quiz v3 grading
    details: list[QuizResultDetail] = Field(default_factory=list)


class WeaknessItem(BaseModel):
    """A knowledge weakness or strength identified from the quiz result."""
    topic: str = ""
    error_count: int = 0
    question_ids: list[str] = Field(default_factory=list)
    diagnosis: str = ""              # natural-language explanation
    review_pages: list[Citation] = Field(default_factory=list)


class QuizDiagnosisFact(BaseModel):
    """Immutable scored fact exposed to diagnosis generation and review."""

    question_id: str = Field(min_length=1)
    question_type: QuestionType
    topic: str = ""
    is_correct: bool
    points_awarded: float
    max_points: float = Field(gt=0)
    citations: list[Citation] = Field(default_factory=list)


class WeaknessReport(BaseModel):
    """Post-quiz knowledge diagnosis.  LLM-generated preferred, rule-based fallback."""
    weaknesses: list[WeaknessItem] = Field(default_factory=list)
    strengths: list[WeaknessItem] = Field(default_factory=list)
    generated_by: Literal["llm", "rule"] = "llm"
    report_kind: Literal["weakness", "mastery"] = "weakness"
    quality_mode: Literal["standard", "reviewed"] = "standard"
    support_reviewed: bool = False
    quality_metadata: dict[str, Any] = Field(default_factory=dict)

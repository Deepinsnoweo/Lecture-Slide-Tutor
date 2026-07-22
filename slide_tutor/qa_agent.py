"""Free Ask Agent context adapter for the bounded v2 controller.

This module owns no Provider loop.  It converts the existing selection tools
into real, bounded observations by reusing ``execute_tool_calls`` and the
single ``build_qa_context_bundle`` implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from slide_tutor.agent_policy import AgentExecutionPolicy
from slide_tutor.agent_runtime import AgentRuntimeResult, run_bounded_agent
from slide_tutor.llm import complete_agent_turn
from slide_tutor.models import (
    AgentObservation,
    QAContextBundle,
    QAContextPlan,
    QAContextSource,
    SlidePage,
    ToolCall,
    ToolExecutionContext,
    ToolResult,
    ToolSourceRef,
    ToolTrace,
)
from slide_tutor.qa_context import (
    build_qa_context_bundle,
    build_qa_tool_execution_context,
    deterministic_positional_targets,
)
from slide_tutor.qa_tools import QA_TOOL_REGISTRY, tool_results_to_qa_context_plan
from slide_tutor.tool_runtime import (
    ToolRuntimeError,
    build_openai_tool_schemas,
    execute_tool_calls,
    normalize_tool_calls,
)


def _call_key(call: ToolCall) -> str:
    return f"{call.name}:{json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)}"


def _source_key(source: QAContextSource) -> tuple[str, str | None, int | None, str]:
    """Deduplicate page sources by complete file/page identity."""
    if source.kind == "memory":
        return ("memory", source.source_file, source.page_number, source.reason)
    return ("page", source.source_file, source.page_number, "")


def _source_ref(source: QAContextSource) -> ToolSourceRef:
    return ToolSourceRef(
        kind=source.kind,
        source_file=source.source_file,
        page_number=source.page_number,
        metadata={"reason": source.reason[:160]},
    )


def _truncate_observation(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _observation_digest(
    call: ToolCall,
    result: ToolResult,
    trace: ToolTrace,
    fresh_sources: list[QAContextSource],
    observed_bundle: QAContextBundle,
) -> str:
    """Build a compact controller-facing summary without copying page text.

    The final answerer still receives the complete merged bundle.  The
    controller only needs enough provenance and a short snippet to decide
    whether another read is justified, so keeping this representation
    deterministic prevents large observations from dominating later turns.
    """
    lines = [
        f"tool={call.name} status={trace.status}",
        (
            f"new_sources={len(fresh_sources)} "
            f"loaded_prompt_chars={len(observed_bundle.prompt_context)}"
        ),
    ]
    if result.structured_data:
        selection = json.dumps(
            result.structured_data,
            sort_keys=True,
            ensure_ascii=False,
        )
        lines.append(f"selection={selection[:240]}")
    for source in fresh_sources:
        identity = source.source_file or "unnamed source"
        if source.page_number is not None:
            identity += f", page {source.page_number}"
        if source.title:
            identity += f", title={source.title}"
        snippet = " ".join((source.snippet or source.reason).split())[:180]
        lines.append(f"source={identity}; snippet={snippet or '(no snippet)'}")
    if not fresh_sources:
        lines.append("result=No new context source was added; do not repeat the same call.")
    return "\n".join(lines)


def build_agent_baseline(
    question: str,
    current_slide: SlidePage,
    max_page: int,
    memory: list[dict],
    index: tuple[list[Any], object, object],
) -> tuple[QAContextPlan, QAContextBundle]:
    """Build current/positional/memory context before the first Agent turn."""
    force_current, page_numbers, page_offsets = deterministic_positional_targets(
        question,
        current_slide.page_number,
        max_page,
    )
    plan = QAContextPlan(
        include_current_page=True,
        page_numbers=page_numbers,
        page_offsets=page_offsets,
        semantic_query="",
        use_memory=True,
    )
    bundle = build_qa_context_bundle(
        plan,
        current_slide.project_id or "",
        current_slide,
        current_slide.file_id or "",
        max_page,
        memory,
        index,
        original_question=question,
    )
    return plan, bundle


def _selection_plan(result: ToolResult) -> QAContextPlan:
    plan = tool_results_to_qa_context_plan([result])
    return plan.model_copy(update={
        "include_current_page": False,
        "use_memory": False,
    })


@dataclass
class QAAgentContextAccumulator:
    """Mutable in-memory context state shared by bounded Agent rounds."""

    question: str
    current_slide: SlidePage
    max_page: int
    memory: list[dict]
    index: tuple[list[Any], object, object]
    policy: AgentExecutionPolicy
    plan: QAContextPlan
    bundle: QAContextBundle
    seen_call_keys: set[str] = field(default_factory=set)
    executed_call_count: int = 0
    observation_chars_used: int = 0

    @property
    def baseline_source_count(self) -> int:
        return len(self.bundle.sources)

    def _merge_bundle(self, observed_bundle: QAContextBundle) -> list[QAContextSource]:
        existing = {_source_key(source) for source in self.bundle.sources}
        fresh: list[QAContextSource] = []
        for source in observed_bundle.sources:
            key = _source_key(source)
            if key in existing:
                continue
            existing.add(key)
            self.bundle.sources.append(source)
            fresh.append(source)
        if fresh and observed_bundle.prompt_context.strip():
            self.bundle.prompt_context = "\n".join(filter(
                None,
                [self.bundle.prompt_context.strip(), observed_bundle.prompt_context.strip()],
            ))
        if fresh:
            self.bundle.planner_summary = "; ".join(filter(
                None,
                [self.bundle.planner_summary, observed_bundle.planner_summary],
            ))
        return fresh

    def observe_result(
        self,
        call: ToolCall,
        result: ToolResult | None,
        trace: ToolTrace,
    ) -> AgentObservation:
        """Turn one validated selection result into a real bounded observation."""
        key = _call_key(call)
        if trace.status == "deduplicated" or key in self.seen_call_keys:
            return AgentObservation(
                call_id=call.call_id,
                tool_name=call.name,
                status="deduplicated",
                content_for_model="Duplicate selection; no new context was loaded.",
            )
        self.seen_call_keys.add(key)
        self.executed_call_count += 1
        if result is None or result.status == "error":
            return AgentObservation(
                call_id=call.call_id,
                tool_name=call.name,
                status="error",
                error=(result.error if result is not None else "tool returned no result")
                or "tool failed",
            )

        observation_plan = _selection_plan(result)
        observed_bundle = build_qa_context_bundle(
            observation_plan,
            self.current_slide.project_id or "",
            self.current_slide,
            self.current_slide.file_id or "",
            self.max_page,
            [],
            self.index,
            original_question=self.question,
        )
        fresh_sources = self._merge_bundle(observed_bundle)
        self.plan = self.plan.model_copy(update={
            "page_numbers": list(dict.fromkeys(
                [*self.plan.page_numbers, *observation_plan.page_numbers],
            )),
            "page_offsets": list(dict.fromkeys(
                [*self.plan.page_offsets, *observation_plan.page_offsets],
            )),
            "semantic_query": self.plan.semantic_query or observation_plan.semantic_query,
        })
        remaining_budget = max(
            self.policy.max_total_observation_chars - self.observation_chars_used,
            0,
        )
        digest = _observation_digest(
            call,
            result,
            trace,
            fresh_sources,
            observed_bundle,
        )
        content, truncated = _truncate_observation(
            digest,
            min(self.policy.max_observation_chars_per_call, remaining_budget),
        )
        self.observation_chars_used += len(content)
        if not fresh_sources:
            content = "Selection completed; no new context sources were added."
            truncated = False
        return AgentObservation(
            call_id=call.call_id,
            tool_name=call.name,
            status="executed" if fresh_sources else "no_new_context",
            content_for_model=content,
            sources=[_source_ref(source) for source in fresh_sources],
            truncated=truncated,
        )


def create_agent_accumulator(
    question: str,
    current_slide: SlidePage,
    max_page: int,
    memory: list[dict],
    index: tuple[list[Any], object, object],
    policy: AgentExecutionPolicy,
) -> QAAgentContextAccumulator:
    plan, bundle = build_agent_baseline(
        question, current_slide, max_page, memory, index,
    )
    return QAAgentContextAccumulator(
        question=question,
        current_slide=current_slide,
        max_page=max_page,
        memory=memory,
        index=index,
        policy=policy,
        plan=plan,
        bundle=bundle,
    )


def build_agent_controller_messages(
    question: str,
    baseline_bundle: QAContextBundle,
) -> list[Any]:
    """Create the in-memory controller prompt with trusted baseline context."""
    from slide_tutor.models import AgentMessage

    system = (
        "You are a bounded context controller for a course-slide tutor. "
        "Do not answer the student and do not write explanations. Choose only "
        "the supplied read tools or call finish_answer. The application injects "
        "the current PDF scope; never invent project IDs, file IDs, paths, or "
        "tools. Use the previous tool observations to decide whether another "
        "read is needed. For search_slides, write a short retrieval query of "
        "roughly 3-10 high-signal terms, preserving technical terms and useful "
        "multilingual words; never paste the whole question or add IDs. Examples: "
        "world model two types -> 'world model 两种 世界モデル'; "
        "express listener reactions -> 'express listener reactions assessment "
        "backchannels'; regression classification -> 'regression classification "
        "target output'. If the baseline or a new observation already contains "
        "direct evidence, call finish_answer. After a no-new-context or duplicate "
        "observation, finish unless the user question still has an explicit gap."
    )
    user = (
        f"Student question:\n{question}\n\n"
        "Trusted baseline context loaded before the first round:\n"
        f"{baseline_bundle.prompt_context or '(no extractable baseline text)'}\n\n"
        "Use a read tool only for a concrete missing evidence gap. Call "
        "finish_answer as soon as the accumulated sources directly support the "
        "question; do not request a confirmation round."
    )
    return [
        AgentMessage(role="system", content=system),
        AgentMessage(role="user", content=user),
    ]


def agent_tool_schemas() -> list[dict[str, Any]]:
    """Return v2 read schemas plus its separate finish control action."""
    return [
        *build_openai_tool_schemas(QA_TOOL_REGISTRY),
        {
            "type": "function",
            "function": {
                "name": "finish_answer",
                "description": "Finish context gathering and send accumulated context to the answerer.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        },
    ]


def run_free_ask_agent(
    *,
    question: str,
    current_slide: SlidePage,
    max_page: int,
    memory: list[dict],
    index: tuple[list[Any], object, object],
    policy: AgentExecutionPolicy,
    cancel_signal=None,
    controller=None,
) -> AgentRuntimeResult:
    """Run v2 over a trusted baseline and return accumulated final context."""
    accumulator = create_agent_accumulator(
        question,
        current_slide,
        max_page,
        memory,
        index,
        policy,
    )
    return run_bounded_agent(
        initial_messages=build_agent_controller_messages(
            question,
            accumulator.bundle,
        ),
        tools=agent_tool_schemas(),
        context=accumulator,
        execute_calls=lambda calls: execute_qa_selection_calls(
            calls,
            accumulator=accumulator,
        ),
        validate_calls=lambda calls: validate_qa_selection_calls(
            calls,
            accumulator=accumulator,
        ),
        policy=policy,
        controller=controller or complete_agent_turn,
        cancel_signal=cancel_signal,
    )


def execute_qa_selection_calls(
    calls: list[ToolCall],
    *,
    accumulator: QAAgentContextAccumulator,
    context: ToolExecutionContext | None = None,
) -> tuple[list[AgentObservation], list[ToolTrace]]:
    """Execute one read-only batch and return one observation per call id."""
    normalized = normalize_tool_calls(calls)
    if len(normalized) > accumulator.policy.max_calls_per_round:
        raise ToolRuntimeError("tool call batch exceeds Agent per-round limit")
    call_ids = [call.call_id for call in normalized]
    if len(set(call_ids)) != len(call_ids):
        raise ToolRuntimeError("duplicate tool call id in Agent turn")
    execution_context = context or build_qa_tool_execution_context(
        accumulator.current_slide,
        accumulator.max_page,
    )
    fresh_calls = [
        call for call in normalized
        if _call_key(call) not in accumulator.seen_call_keys
    ]
    results, executed_traces = execute_tool_calls(
        fresh_calls,
        registry=QA_TOOL_REGISTRY,
        context=execution_context,
        max_calls=accumulator.policy.max_calls_per_round,
    )
    result_by_id = {result.call_id: result for result in results}
    trace_by_id = {trace.call_id: trace for trace in executed_traces}
    observations: list[AgentObservation] = []
    traces: list[ToolTrace] = []
    for call in normalized:
        key = _call_key(call)
        if key in accumulator.seen_call_keys and call.call_id not in trace_by_id:
            trace = ToolTrace(
                call_id=call.call_id,
                name=call.name,
                status="deduplicated",
                arguments=call.arguments,
                summary="duplicate call skipped",
            )
            traces.append(trace)
            observations.append(accumulator.observe_result(call, None, trace))
            continue
        trace = trace_by_id.get(call.call_id)
        if trace is None:
            trace = ToolTrace(
                call_id=call.call_id,
                name=call.name,
                status="rejected",
                arguments=call.arguments,
                error="tool runtime returned no trace",
            )
        traces.append(trace)
        observations.append(
            accumulator.observe_result(call, result_by_id.get(call.call_id), trace)
        )
    return observations, traces


def validate_qa_selection_calls(
    calls: list[ToolCall],
    *,
    accumulator: QAAgentContextAccumulator,
) -> list[ToolCall]:
    """Validate an entire read batch before any handler is executed."""
    normalized = normalize_tool_calls(calls)
    if len(normalized) > accumulator.policy.max_calls_per_round:
        raise ToolRuntimeError("tool call batch exceeds Agent per-round limit")
    call_ids = [call.call_id for call in normalized]
    if len(set(call_ids)) != len(call_ids):
        raise ToolRuntimeError("duplicate tool call id in Agent turn")
    for call in normalized:
        if call.name not in accumulator.policy.allowed_tools:
            raise ToolRuntimeError("Agent selected a tool outside the allowlist")
        definition = QA_TOOL_REGISTRY.get(call.name)
        if definition is None:
            raise ToolRuntimeError(f"unknown tool: {call.name}")
        try:
            arguments = definition.arguments_model.model_validate(call.arguments)
        except Exception as exc:
            raise ToolRuntimeError(f"invalid arguments for tool: {call.name}") from exc
    return normalized


__all__ = [
    "QAAgentContextAccumulator",
    "build_agent_baseline",
    "create_agent_accumulator",
    "execute_qa_selection_calls",
    "validate_qa_selection_calls",
]

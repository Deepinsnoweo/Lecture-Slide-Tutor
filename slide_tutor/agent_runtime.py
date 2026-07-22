"""Provider-neutral bounded Agent state machine for Q&A v2."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event
from typing import Any, Protocol

from slide_tutor.agent_policy import AgentExecutionPolicy
from slide_tutor.llm import (
    LLMAgentTurnProtocolError,
    LLMAgentTurnUnavailable,
    LLMCancelled,
    complete_agent_turn,
)
from slide_tutor.models import (
    AgentMessage,
    AgentModelTurn,
    AgentObservation,
    AgentRoundTrace,
    AgentRunSummary,
    LLMCallMetrics,
    ToolCall,
    ToolTrace,
)
from slide_tutor.tool_runtime import ToolRuntimeError


class AgentCancelled(LLMCancelled):
    """Cooperative cancellation; callers must not fallback or finalize."""


class AgentRuntimeContext(Protocol):
    """Minimal adapter surface required by this provider-neutral runtime."""

    bundle: Any
    baseline_source_count: int
    executed_call_count: int
    seen_call_keys: set[str]


Controller = Callable[..., AgentModelTurn]
ExecuteCalls = Callable[
    [list[ToolCall]],
    tuple[list[AgentObservation], list[ToolTrace]],
]
ValidateCalls = Callable[[list[ToolCall]], list[ToolCall]]


@dataclass(frozen=True)
class AgentRuntimeResult:
    """Final accumulated context plus sanitized run summary."""

    bundle: Any
    summary: AgentRunSummary
    fallback_requested: bool = False


def _safe_failure(exc: Exception) -> str:
    return f"{type(exc).__name__}: Agent controller/tool protocol failure"


def _usage_summary(metrics: list[LLMCallMetrics]) -> tuple[
    str, int | None, int | None, int | None
]:
    if not metrics:
        return "unavailable", None, None, None
    fields = [
        [item.prompt_tokens, item.completion_tokens, item.total_tokens]
        for item in metrics
    ]
    populated = [value is not None for row in fields for value in row]
    if all(populated):
        status = "exact"
    elif any(populated):
        status = "partial"
    else:
        status = "unavailable"
    sums = [
        sum(row[index] or 0 for row in fields) if any(row[index] is not None for row in fields) else None
        for index in range(3)
    ]
    return status, sums[0], sums[1], sums[2]


def _summary(
    *,
    context: AgentRuntimeContext,
    mode: str,
    rounds: int,
    stop_reason: str,
    traces: list[AgentRoundTrace],
    metrics: list[LLMCallMetrics],
    elapsed_ms: float,
    fallback_reason: str | None = None,
) -> AgentRunSummary:
    usage_status, prompt, completion, total = _usage_summary(metrics)
    return AgentRunSummary(
        mode=mode,
        rounds=rounds,
        executed_tool_calls=context.executed_call_count,
        baseline_source_count=context.baseline_source_count,
        stop_reason=stop_reason,
        round_traces=traces,
        usage_status=usage_status,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        elapsed_ms=elapsed_ms,
        fallback_reason=fallback_reason,
    )


def _check_cancel(cancel_signal: Event | None) -> None:
    if cancel_signal is not None and cancel_signal.is_set():
        raise AgentCancelled("Agent generation cancelled")


def _check_time(started: float, policy: AgentExecutionPolicy) -> bool:
    return time.perf_counter() - started >= policy.max_agent_seconds


def _is_progress(observations: list[AgentObservation]) -> bool:
    return any(
        observation.status in {"executed", "no_new_context"}
        for observation in observations
    )


def _controller_failure_result(
    *,
    context: AgentRuntimeContext,
    traces: list[AgentRoundTrace],
    metrics: list[LLMCallMetrics],
    rounds: int,
    started: float,
    progressed: bool,
    protocol: bool,
    exc: Exception,
) -> AgentRuntimeResult:
    stop_reason = (
        "protocol_error_after_progress" if protocol else "provider_error_after_progress"
    ) if progressed else "fallback_v1"
    fallback = not progressed
    summary = _summary(
        context=context,
        mode="bounded_agent_v2_fallback" if fallback else "bounded_agent_v2",
        rounds=rounds,
        stop_reason=stop_reason,
        traces=traces,
        metrics=metrics,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        fallback_reason=_safe_failure(exc),
    )
    return AgentRuntimeResult(
        bundle=context.bundle,
        summary=summary,
        fallback_requested=fallback,
    )


def run_bounded_agent(
    *,
    initial_messages: list[AgentMessage],
    tools: list[dict[str, Any]],
    context: AgentRuntimeContext,
    execute_calls: ExecuteCalls,
    policy: AgentExecutionPolicy,
    controller: Controller = complete_agent_turn,
    validate_calls: ValidateCalls | None = None,
    model: str | None = None,
    cancel_signal: Event | None = None,
) -> AgentRuntimeResult:
    """Run the bounded controller/tool loop and return sanitized provenance."""
    started = time.perf_counter()
    messages = list(initial_messages)
    metrics: list[LLMCallMetrics] = []
    round_traces: list[AgentRoundTrace] = []
    seen_call_ids: set[str] = set()
    progressed = False
    rounds = 0

    for round_number in range(1, policy.max_rounds + 1):
        _check_cancel(cancel_signal)
        if _check_time(started, policy):
            return AgentRuntimeResult(
                bundle=context.bundle,
                summary=_summary(
                    context=context,
                    mode="bounded_agent_v2",
                    rounds=rounds,
                    stop_reason="time_budget",
                    traces=round_traces,
                    metrics=metrics,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                ),
            )
        rounds = round_number
        round_started = time.perf_counter()
        round_metrics: list[LLMCallMetrics] = []
        try:
            turn = controller(
                messages,
                tools,
                model=model,
                metrics_callback=round_metrics.append,
            )
            metrics.extend(round_metrics)
        except AgentCancelled:
            raise
        except LLMAgentTurnProtocolError as exc:
            metrics.extend(round_metrics)
            return _controller_failure_result(
                context=context,
                traces=round_traces,
                metrics=metrics,
                rounds=rounds,
                started=started,
                progressed=progressed,
                protocol=True,
                exc=exc,
            )
        except LLMAgentTurnUnavailable as exc:
            metrics.extend(round_metrics)
            return _controller_failure_result(
                context=context,
                traces=round_traces,
                metrics=metrics,
                rounds=rounds,
                started=started,
                progressed=progressed,
                protocol=False,
                exc=exc,
            )
        except Exception as exc:
            metrics.extend(round_metrics)
            return _controller_failure_result(
                context=context,
                traces=round_traces,
                metrics=metrics,
                rounds=rounds,
                started=started,
                progressed=progressed,
                protocol=False,
                exc=exc,
            )

        _check_cancel(cancel_signal)
        usage_status, _prompt, _completion, total_tokens = _usage_summary(metrics)
        controller_metric = round_metrics[-1] if round_metrics else None
        if (
            policy.max_controller_tokens is not None
            and usage_status == "exact"
            and total_tokens is not None
            and total_tokens >= policy.max_controller_tokens
        ):
            round_traces.append(AgentRoundTrace(
                round_number=round_number,
                controller_metrics=controller_metric,
                elapsed_ms=(time.perf_counter() - round_started) * 1000,
            ))
            return AgentRuntimeResult(
                bundle=context.bundle,
                summary=_summary(
                    context=context,
                    mode="bounded_agent_v2",
                    rounds=rounds,
                    stop_reason="token_budget",
                    traces=round_traces,
                    metrics=metrics,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                ),
            )

        if _check_time(started, policy):
            round_traces.append(AgentRoundTrace(
                round_number=round_number,
                controller_metrics=controller_metric,
                elapsed_ms=(time.perf_counter() - round_started) * 1000,
            ))
            return AgentRuntimeResult(
                bundle=context.bundle,
                summary=_summary(
                    context=context,
                    mode="bounded_agent_v2",
                    rounds=rounds,
                    stop_reason="time_budget",
                    traces=round_traces,
                    metrics=metrics,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                ),
            )

        if not turn.tool_calls:
            exc = ToolRuntimeError("Agent turn returned no finish or read tool calls")
            return _controller_failure_result(
                context=context,
                traces=round_traces,
                metrics=metrics,
                rounds=rounds,
                started=started,
                progressed=progressed,
                protocol=True,
                exc=exc,
            )

        finish_calls = [call for call in turn.tool_calls if call.name == "finish_answer"]
        if finish_calls:
            if len(finish_calls) != 1 or len(turn.tool_calls) != 1:
                return _controller_failure_result(
                    context=context,
                    traces=round_traces,
                    metrics=metrics,
                    rounds=rounds,
                    started=started,
                    progressed=progressed,
                    protocol=True,
                    exc=ToolRuntimeError("finish_answer cannot be mixed with read calls"),
                )
            round_traces.append(AgentRoundTrace(
                round_number=round_number,
                controller_metrics=controller_metric,
                elapsed_ms=(time.perf_counter() - round_started) * 1000,
            ))
            return AgentRuntimeResult(
                bundle=context.bundle,
                summary=_summary(
                    context=context,
                    mode="bounded_agent_v2",
                    rounds=rounds,
                    stop_reason="finish_requested",
                    traces=round_traces,
                    metrics=metrics,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                ),
            )

        try:
            if len(turn.tool_calls) > policy.max_calls_per_round:
                return AgentRuntimeResult(
                    bundle=context.bundle,
                    summary=_summary(
                        context=context,
                        mode="bounded_agent_v2",
                        rounds=rounds,
                        stop_reason="tool_exhausted",
                        traces=round_traces,
                        metrics=metrics,
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                    ),
                )
            duplicate_ids = seen_call_ids.intersection(
                call.call_id for call in turn.tool_calls
            )
            if duplicate_ids:
                raise ToolRuntimeError("duplicate tool call id across Agent rounds")
            seen_call_ids.update(call.call_id for call in turn.tool_calls)
            validated_calls = (
                validate_calls(list(turn.tool_calls))
                if validate_calls is not None
                else list(turn.tool_calls)
            )
            new_call_count = sum(
                f"{call.name}:{json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)}"
                not in context.seen_call_keys
                for call in validated_calls
            )
            if context.executed_call_count + new_call_count > policy.max_total_tool_calls:
                round_traces.append(AgentRoundTrace(
                    round_number=round_number,
                    controller_metrics=controller_metric,
                    elapsed_ms=(time.perf_counter() - round_started) * 1000,
                ))
                return AgentRuntimeResult(
                    bundle=context.bundle,
                    summary=_summary(
                        context=context,
                        mode="bounded_agent_v2",
                        rounds=rounds,
                        stop_reason="max_total_tool_calls",
                        traces=round_traces,
                        metrics=metrics,
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                    ),
                )
            if any(call.name not in policy.allowed_tools for call in validated_calls):
                raise ToolRuntimeError("Agent selected a tool outside the allowlist")
            messages.append(AgentMessage(
                role="assistant",
                content=turn.assistant_text,
                tool_calls=validated_calls,
            ))
        except (ToolRuntimeError, ValueError) as exc:
            return _controller_failure_result(
                context=context,
                traces=round_traces,
                metrics=metrics,
                rounds=rounds,
                started=started,
                progressed=progressed,
                protocol=True,
                exc=exc,
            )

        observations: list[AgentObservation] = []
        traces: list[ToolTrace] = []
        for call in validated_calls:
            _check_cancel(cancel_signal)
            try:
                call_observations, call_traces = execute_calls([call])
            except Exception as exc:
                return _controller_failure_result(
                    context=context,
                    traces=round_traces,
                    metrics=metrics,
                    rounds=rounds,
                    started=started,
                    progressed=progressed,
                    protocol=True,
                    exc=exc,
                )
            observations.extend(call_observations)
            traces.extend(call_traces)
            observation = call_observations[0]
            messages.append(AgentMessage(
                role="tool",
                tool_call_id=call.call_id,
                content=observation.content_for_model or observation.error or observation.status,
            ))

        round_traces.append(AgentRoundTrace(
            round_number=round_number,
            tool_traces=traces,
            observation_source_count=sum(len(item.sources) for item in observations),
            observation_chars=sum(len(item.content_for_model) for item in observations),
            controller_metrics=controller_metric,
            elapsed_ms=(time.perf_counter() - round_started) * 1000,
        ))
        if _is_progress(observations):
            progressed = True
        if _check_time(started, policy):
            return AgentRuntimeResult(
                bundle=context.bundle,
                summary=_summary(
                    context=context,
                    mode="bounded_agent_v2",
                    rounds=rounds,
                    stop_reason="time_budget",
                    traces=round_traces,
                    metrics=metrics,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                ),
            )

    return AgentRuntimeResult(
        bundle=context.bundle,
        summary=_summary(
            context=context,
            mode="bounded_agent_v2",
            rounds=rounds,
            stop_reason="max_rounds",
            traces=round_traces,
            metrics=metrics,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        ),
    )


__all__ = [
    "AgentCancelled",
    "AgentRuntimeResult",
    "run_bounded_agent",
]

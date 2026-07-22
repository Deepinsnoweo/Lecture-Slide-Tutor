"""Provider-neutral, bounded Function Calling runtime.

The runtime owns tool schema construction, call normalization, argument
validation, trusted execution context, and structured results.  It deliberately
does not know about Streamlit, PDF loading, retrieval, or ``QAContextPlan`` so
future learning-state tools and agent loops can reuse it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from slide_tutor.models import (
    ToolCall,
    ToolExecutionContext,
    ToolResult,
    ToolTrace,
)


class ToolRuntimeError(ValueError):
    """Raised when a tool selection cannot be safely normalized or validated."""


ToolHandler = Callable[[ToolExecutionContext, BaseModel], ToolResult]


@dataclass(frozen=True)
class ToolDefinition:
    """One allowlisted tool and its validated handler."""

    name: str
    description: str
    arguments_model: type[BaseModel]
    handler: ToolHandler


class ToolRegistry:
    """Immutable-name registry used by Function Calling and future adapters."""

    def __init__(self, definitions: Sequence[ToolDefinition]) -> None:
        by_name: dict[str, ToolDefinition] = {}
        for definition in definitions:
            if not definition.name.strip():
                raise ValueError("tool name cannot be empty")
            if definition.name in by_name:
                raise ValueError(f"duplicate tool name: {definition.name}")
            by_name[definition.name] = definition
        self._by_name = by_name

    def get(self, name: str) -> ToolDefinition | None:
        return self._by_name.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __iter__(self):
        return iter(self._by_name.values())

    def names(self) -> tuple[str, ...]:
        return tuple(self._by_name)


def _strict_parameters_schema(arguments_model: type[BaseModel]) -> dict[str, Any]:
    """Build a conservative JSON Schema subset accepted by compatible APIs."""
    schema = arguments_model.model_json_schema()
    schema.setdefault("type", "object")
    schema["additionalProperties"] = False
    return schema


def build_openai_tool_schemas(registry: ToolRegistry) -> list[dict[str, Any]]:
    """Return OpenAI-compatible function definitions without SDK objects."""
    return [
        {
            "type": "function",
            "function": {
                "name": definition.name,
                "description": definition.description,
                "parameters": _strict_parameters_schema(definition.arguments_model),
            },
        }
        for definition in registry
    ]


def _read_member(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _validate_json_arguments(arguments: Any, call_id: str) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise ToolRuntimeError(
            f"tool call {call_id} arguments must be a JSON object"
        )
    try:
        json.dumps(arguments, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ToolRuntimeError(
            f"tool call {call_id} arguments must contain ordinary JSON values"
        ) from exc
    return arguments


def normalize_tool_calls(raw_calls: Sequence[Any]) -> list[ToolCall]:
    """Normalize OpenAI SDK objects or dict-shaped fakes to ``ToolCall``.

    Provider-specific response objects are accepted only at this boundary.  A
    missing id/name, non-object arguments, or malformed JSON is rejected before
    any handler runs so a partial batch cannot cross the scope boundary.
    """
    normalized: list[ToolCall] = []
    for index, raw_call in enumerate(raw_calls):
        if isinstance(raw_call, ToolCall):
            arguments = _validate_json_arguments(raw_call.arguments, raw_call.call_id)
            normalized.append(raw_call.model_copy(update={"arguments": arguments}))
            continue
        call_id = _read_member(raw_call, "id")
        function = _read_member(raw_call, "function") or raw_call
        name = _read_member(function, "name")
        raw_arguments = _read_member(function, "arguments")
        if not isinstance(call_id, str) or not call_id.strip():
            raise ToolRuntimeError(f"tool call {index} is missing call id")
        if not isinstance(name, str) or not name.strip():
            raise ToolRuntimeError(f"tool call {index} is missing function name")
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise ToolRuntimeError(
                    f"tool call {call_id} has malformed arguments JSON"
                ) from exc
        else:
            arguments = raw_arguments
        arguments = _validate_json_arguments(arguments, call_id)
        normalized.append(ToolCall(
            call_id=call_id,
            name=name,
            arguments=arguments,
        ))
    return normalized


def _call_key(call: ToolCall) -> str:
    return f"{call.name}:{json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)}"


def _safe_exception_summary(exc: Exception) -> str:
    """Avoid copying provider responses, keys, or local paths into a trace."""
    return f"{type(exc).__name__}: tool handler failed"


def execute_tool_calls(
    raw_calls: Sequence[Any],
    *,
    registry: ToolRegistry,
    context: ToolExecutionContext,
    max_calls: int = 4,
) -> tuple[list[ToolResult], list[ToolTrace]]:
    """Validate and execute one bounded batch in stable provider order.

    All call names and argument models are validated before the first handler
    runs.  This keeps an invalid batch from partially executing.  Handler
    failures become structured ``ToolResult(status='error')`` values; callers
    decide whether to fallback or surface the error.
    """
    calls = normalize_tool_calls(raw_calls)
    if len(calls) > max_calls:
        raise ToolRuntimeError(f"tool call batch exceeds limit {max_calls}")

    validated: list[tuple[ToolCall, ToolDefinition, BaseModel]] = []
    seen: set[str] = set()
    traces: list[ToolTrace] = []
    for call in calls:
        key = _call_key(call)
        if key in seen:
            traces.append(ToolTrace(
                call_id=call.call_id,
                name=call.name,
                status="deduplicated",
                arguments=call.arguments,
                summary="duplicate call skipped",
            ))
            continue
        definition = registry.get(call.name)
        if definition is None:
            raise ToolRuntimeError(f"unknown tool: {call.name}")
        try:
            arguments = definition.arguments_model.model_validate(call.arguments)
        except ValidationError as exc:
            raise ToolRuntimeError(
                f"invalid arguments for tool: {call.name}"
            ) from exc
        seen.add(key)
        validated.append((call, definition, arguments))
        traces.append(ToolTrace(
            call_id=call.call_id,
            name=call.name,
            status="accepted",
            arguments=call.arguments,
        ))

    results: list[ToolResult] = []
    trace_by_id = {trace.call_id: trace for trace in traces}
    for call, definition, arguments in validated:
        try:
            result = definition.handler(context, arguments)
            if not isinstance(result, ToolResult):
                raise TypeError("tool handler did not return ToolResult")
            result = result.model_copy(update={
                "call_id": call.call_id,
                "name": definition.name,
            })
        except Exception as exc:  # handlers must not break the QA worker
            result = ToolResult(
                call_id=call.call_id,
                name=definition.name,
                status="error",
                error=_safe_exception_summary(exc),
            )
        results.append(result)
        trace = trace_by_id[call.call_id]
        if result.status == "success":
            trace.status = "executed"  # type: ignore[misc]
            trace.summary = "tool executed"
        else:
            trace.status = "rejected"  # type: ignore[misc]
            trace.error = result.error or "tool returned error"
    return results, traces

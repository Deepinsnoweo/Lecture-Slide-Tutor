from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict, Field

from slide_tutor.models import RetrievalScope, ToolCall, ToolExecutionContext, ToolResult
from slide_tutor.tool_runtime import (
    ToolDefinition,
    ToolRegistry,
    ToolRuntimeError,
    build_openai_tool_schemas,
    execute_tool_calls,
    normalize_tool_calls,
)


class _Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: int = Field(ge=1)


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        project_id="project-1",
        current_file_id="file-1",
        current_page_number=3,
        max_page=10,
        retrieval_scope=RetrievalScope(
            kind="current_file",
            project_id="project-1",
            file_id="file-1",
        ),
        allowed_file_ids=["file-1"],
    )


def _success_handler(context, arguments):
    assert context.project_id == "project-1"
    return ToolResult(
        call_id="placeholder",
        name="echo_value",
        status="success",
        structured_data={"value": arguments.value},
    )


def _error_handler(context, arguments):
    raise RuntimeError("sensitive path should not enter trace")


def _registry() -> ToolRegistry:
    return ToolRegistry([
        ToolDefinition(
            name="echo_value",
            description="Return a validated value.",
            arguments_model=_Arguments,
            handler=_success_handler,
        ),
        ToolDefinition(
            name="error_tool",
            description="Return a structured error.",
            arguments_model=_Arguments,
            handler=_error_handler,
        ),
    ])


def test_build_openai_tool_schemas_is_strict_and_deterministic():
    schemas = build_openai_tool_schemas(_registry())

    assert [schema["function"]["name"] for schema in schemas] == [
        "echo_value", "error_tool",
    ]
    assert schemas[0]["function"]["parameters"]["additionalProperties"] is False
    assert schemas[0]["function"]["parameters"]["required"] == ["value"]


def test_normalize_tool_calls_accepts_dicts_and_sdk_like_objects():
    calls = normalize_tool_calls([
        {
            "id": "call-1",
            "function": {"name": "echo_value", "arguments": '{"value": 2}'},
        },
        SimpleNamespace(
            id="call-2",
            function=SimpleNamespace(name="echo_value", arguments={"value": 3}),
        ),
    ])

    assert [call.call_id for call in calls] == ["call-1", "call-2"]
    assert [call.arguments for call in calls] == [{"value": 2}, {"value": 3}]


def test_normalize_tool_calls_rejects_non_json_values_even_for_normalized_models():
    with pytest.raises(ToolRuntimeError, match="ordinary JSON"):
        normalize_tool_calls([
            ToolCall(call_id="call-1", name="echo_value", arguments={"value": {1}}),
        ])


@pytest.mark.parametrize("raw_call", [
    {"id": "call-1", "function": {"name": "echo_value", "arguments": "not-json"}},
    {"id": "call-1", "function": {"name": "echo_value", "arguments": [1]}},
    {"id": "call-1", "function": {"name": "echo_value", "arguments": {}}},
])
def test_normalize_or_validate_rejects_malformed_calls(raw_call):
    if raw_call["function"]["arguments"] == {}:
        with pytest.raises(ToolRuntimeError, match="invalid arguments"):
            execute_tool_calls([raw_call], registry=_registry(), context=_context())
    else:
        with pytest.raises(ToolRuntimeError):
            execute_tool_calls([raw_call], registry=_registry(), context=_context())


def test_unknown_tool_and_batch_limit_are_rejected_before_execution():
    with pytest.raises(ToolRuntimeError, match="unknown tool"):
        execute_tool_calls([
            {"id": "call-1", "function": {"name": "missing", "arguments": {}}},
        ], registry=_registry(), context=_context())

    raw_calls = [
        {"id": f"call-{i}", "function": {"name": "echo_value", "arguments": {"value": 1}}}
        for i in range(5)
    ]
    with pytest.raises(ToolRuntimeError, match="exceeds limit"):
        execute_tool_calls(raw_calls, registry=_registry(), context=_context(), max_calls=4)


def test_execute_calls_deduplicates_and_returns_trace_in_provider_order():
    raw_calls = [
        {"id": "call-1", "function": {"name": "echo_value", "arguments": {"value": 2}}},
        {"id": "call-2", "function": {"name": "echo_value", "arguments": {"value": 2}}},
        {"id": "call-3", "function": {"name": "echo_value", "arguments": {"value": 3}}},
    ]

    results, traces = execute_tool_calls(
        raw_calls, registry=_registry(), context=_context(),
    )

    assert [result.structured_data for result in results] == [
        {"value": 2}, {"value": 3},
    ]
    assert [trace.status for trace in traces] == [
        "executed", "deduplicated", "executed",
    ]


def test_handler_failure_becomes_safe_structured_error():
    results, traces = execute_tool_calls([
        {"id": "call-1", "function": {"name": "error_tool", "arguments": {"value": 1}}},
    ], registry=_registry(), context=_context())

    assert results[0].status == "error"
    assert results[0].error == "RuntimeError: tool handler failed"
    assert traces[0].status == "rejected"
    assert "sensitive" not in traces[0].error or traces[0].error is None


def test_execution_context_rejects_scope_identity_mismatch():
    with pytest.raises(ValueError, match="allowed_file_ids"):
        ToolExecutionContext(
            project_id="project-1",
            current_file_id="file-2",
            allowed_file_ids=["file-1"],
        )

    with pytest.raises(ValueError, match="execution project"):
        ToolExecutionContext(
            project_id="project-1",
            retrieval_scope=RetrievalScope(
                kind="whole_project",
                project_id="project-2",
            ),
        )

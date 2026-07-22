from __future__ import annotations

import pytest

from slide_tutor.models import RetrievalScope, ToolExecutionContext, ToolResult
from slide_tutor.qa_tools import (
    QAToolSelectionError,
    QA_TOOL_REGISTRY,
    tool_results_to_qa_context_plan,
)
from slide_tutor.tool_runtime import build_openai_tool_schemas, execute_tool_calls


def _context(**updates) -> ToolExecutionContext:
    values = {
        "project_id": "project-1",
        "current_file_id": "file-1",
        "current_page_number": 5,
        "max_page": 10,
        "retrieval_scope": RetrievalScope(
            kind="current_file",
            project_id="project-1",
            file_id="file-1",
        ),
        "allowed_file_ids": ["file-1"],
    }
    values.update(updates)
    return ToolExecutionContext(**values)


def _call(call_id: str, name: str, arguments: dict):
    return {"id": call_id, "function": {"name": name, "arguments": arguments}}


def test_registry_exposes_only_bounded_qa_tools():
    assert QA_TOOL_REGISTRY.names() == (
        "load_pages", "load_neighbor_pages", "search_slides",
    )


def test_tool_argument_schemas_encode_item_bounds():
    schemas = {
        item["function"]["name"]: item
        for item in build_openai_tool_schemas(QA_TOOL_REGISTRY)
    }
    page_items = schemas["load_pages"]["function"]["parameters"]["properties"][
        "page_numbers"
    ]["items"]
    offset_items = schemas["load_neighbor_pages"]["function"]["parameters"][
        "properties"
    ]["offsets"]["items"]

    assert page_items["minimum"] == 1
    assert offset_items["enum"] == [-3, -2, -1, 1, 2, 3]


def test_handlers_return_generic_results_and_adapter_builds_plan():
    results, traces = execute_tool_calls(
        [
            _call("call-1", "load_pages", {"page_numbers": [7, 7, 99]}),
            _call("call-2", "load_neighbor_pages", {"offsets": [-1, 1]}),
            _call("call-3", "search_slides", {"query": "  LTD mechanism  "}),
        ],
        registry=QA_TOOL_REGISTRY,
        context=_context(),
    )

    assert all(isinstance(result, ToolResult) for result in results)
    assert [trace.status for trace in traces] == ["executed"] * 3
    plan = tool_results_to_qa_context_plan(results)
    assert plan.include_current_page is True
    assert plan.page_numbers == [7]
    assert plan.page_offsets == [-1, 1]
    assert plan.semantic_query == "LTD mechanism"
    assert plan.use_memory is True


def test_page_tools_require_trusted_page_context():
    results, traces = execute_tool_calls(
        [_call("call-1", "load_pages", {"page_numbers": [2]})],
        registry=QA_TOOL_REGISTRY,
        context=ToolExecutionContext(project_id="project-1"),
    )

    assert results[0].status == "error"
    assert traces[0].status == "rejected"


def test_search_tool_rejects_blank_query():
    results, traces = execute_tool_calls(
        [_call("call-1", "search_slides", {"query": "   "})],
        registry=QA_TOOL_REGISTRY,
        context=_context(),
    )
    assert results[0].status == "error"
    assert traces[0].status == "rejected"


def test_page_tools_reject_when_dynamic_pdf_bounds_leave_no_selection():
    page_results, page_traces = execute_tool_calls(
        [_call("call-1", "load_pages", {"page_numbers": [99]})],
        registry=QA_TOOL_REGISTRY,
        context=_context(max_page=10),
    )
    neighbor_results, neighbor_traces = execute_tool_calls(
        [_call("call-2", "load_neighbor_pages", {"offsets": [1]})],
        registry=QA_TOOL_REGISTRY,
        context=_context(current_page_number=10, max_page=10),
    )

    assert page_results[0].status == "error"
    assert page_traces[0].status == "rejected"
    assert neighbor_results[0].status == "error"
    assert neighbor_traces[0].status == "rejected"


def test_adapter_rejects_failed_or_unknown_results():
    with pytest.raises(QAToolSelectionError, match="tool failed"):
        tool_results_to_qa_context_plan([
            ToolResult(
                call_id="call-1",
                name="search_slides",
                status="error",
                error="tool failed",
            ),
        ])

    with pytest.raises(QAToolSelectionError, match="unknown"):
        tool_results_to_qa_context_plan([
            ToolResult(call_id="call-1", name="other", status="success"),
        ])


def test_adapter_keeps_first_query_deterministically():
    results = [
        ToolResult(
            call_id="call-1", name="search_slides", status="success",
            structured_data={"semantic_query": "first"},
        ),
        ToolResult(
            call_id="call-2", name="search_slides", status="success",
            structured_data={"semantic_query": "second"},
        ),
    ]

    plan = tool_results_to_qa_context_plan(results)
    assert plan.semantic_query == "first"


def test_adapter_rejects_more_than_two_search_queries():
    results = [
        ToolResult(
            call_id=f"call-{index}", name="search_slides", status="success",
            structured_data={"semantic_query": f"query-{index}"},
        )
        for index in range(3)
    ]

    with pytest.raises(QAToolSelectionError, match="too many"):
        tool_results_to_qa_context_plan(results)

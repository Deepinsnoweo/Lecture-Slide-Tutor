"""Q&A-specific tools and the adapter into the existing context planner.

The handlers are intentionally selection-oriented in v1: they validate the
model's requested pages/offsets/query and return generic ``ToolResult`` data.
The existing context builder remains the single place that loads page content,
runs retrieval, applies the context budget, and records slide sources.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from slide_tutor.models import (
    QAContextPlan,
    ToolExecutionContext,
    ToolResult,
)
from slide_tutor.tool_runtime import ToolDefinition, ToolRegistry

_ALLOWED_PAGE_OFFSETS = {-3, -2, -1, 1, 2, 3}


class LoadPagesArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_numbers: list[Annotated[int, Field(ge=1)]] = Field(min_length=1, max_length=5)


class LoadNeighborPagesArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    offsets: list[Literal[-3, -2, -1, 1, 2, 3]] = Field(min_length=1, max_length=6)


class SearchSlidesArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500)


class QAToolSelectionError(ValueError):
    """Raised when tool results cannot produce a safe Q&A context plan."""


def _page_context_error(context: ToolExecutionContext) -> str | None:
    missing: list[str] = []
    if not context.current_file_id:
        missing.append("current_file_id")
    if context.current_page_number is None:
        missing.append("current_page_number")
    if context.max_page is None:
        missing.append("max_page")
    if missing:
        return f"Q&A page tool requires context: {', '.join(missing)}"
    return None


def _unique(values: list[int]) -> list[int]:
    return list(dict.fromkeys(values))


def _load_pages(
    context: ToolExecutionContext,
    arguments: BaseModel,
) -> ToolResult:
    """Validate page selection; content loading remains in context assembly."""
    error = _page_context_error(context)
    if error:
        raise ValueError(error)
    page_args = LoadPagesArguments.model_validate(arguments)
    assert context.max_page is not None
    page_numbers = [
        page for page in _unique(page_args.page_numbers)
        if 1 <= page <= context.max_page
    ]
    if not page_numbers:
        raise ValueError("load_pages selected no page within the current file")
    return ToolResult(
        call_id="pending",
        name="load_pages",
        status="success",
        structured_data={"page_numbers": page_numbers},
    )


def _load_neighbor_pages(
    context: ToolExecutionContext,
    arguments: BaseModel,
) -> ToolResult:
    """Validate bounded relative page selection for the current PDF."""
    error = _page_context_error(context)
    if error:
        raise ValueError(error)
    offset_args = LoadNeighborPagesArguments.model_validate(arguments)
    assert context.current_page_number is not None
    assert context.max_page is not None
    offsets = [
        offset for offset in _unique(offset_args.offsets)
        if (
            offset in _ALLOWED_PAGE_OFFSETS
            and 1 <= context.current_page_number + offset <= context.max_page
        )
    ]
    if not offsets:
        raise ValueError("load_neighbor_pages selected no valid offset")
    return ToolResult(
        call_id="pending",
        name="load_neighbor_pages",
        status="success",
        structured_data={"offsets": offsets},
    )


def _search_slides(
    context: ToolExecutionContext,
    arguments: BaseModel,
) -> ToolResult:
    """Validate a bounded semantic query; retrieval executes after planning."""
    if not context.current_file_id:
        raise ValueError("search_slides requires current_file_id")
    query_args = SearchSlidesArguments.model_validate(arguments)
    query = query_args.query.strip()
    if not query:
        raise ValueError("search_slides query cannot be blank")
    return ToolResult(
        call_id="pending",
        name="search_slides",
        status="success",
        structured_data={"semantic_query": query},
    )


QA_TOOL_REGISTRY = ToolRegistry([
    ToolDefinition(
        name="load_pages",
        description=(
            "Select explicit pages from the currently open PDF for grounded context. "
            "Do not provide file IDs or paths."
        ),
        arguments_model=LoadPagesArguments,
        handler=_load_pages,
    ),
    ToolDefinition(
        name="load_neighbor_pages",
        description=(
            "Select bounded previous or next pages in the currently open PDF. "
            "Use offsets -3,-2,-1,1,2,3; do not use 0."
        ),
        arguments_model=LoadNeighborPagesArguments,
        handler=_load_neighbor_pages,
    ),
    ToolDefinition(
        name="search_slides",
        description=(
            "Search the currently open PDF with hybrid retrieval using a concise "
            "3-10 term query. Preserve technical or multilingual terms; do not "
            "paste the whole question or include file IDs and paths."
        ),
        arguments_model=SearchSlidesArguments,
        handler=_search_slides,
    ),
])


def tool_results_to_qa_context_plan(
    results: list[ToolResult],
) -> QAContextPlan:
    """Convert successful Q&A tool results to the legacy plan contract.

    This adapter is deliberately outside ``tool_runtime``.  Future quiz,
    weakness, or progress tools can return their own structured results without
    pretending to be page selections.
    """
    if not results:
        raise QAToolSelectionError("Q&A tool batch returned no results")
    failed = next((result for result in results if result.status == "error"), None)
    if failed is not None:
        raise QAToolSelectionError(failed.error or "Q&A tool failed")

    page_numbers: list[int] = []
    page_offsets: list[int] = []
    queries: list[str] = []
    for result in results:
        data: dict[str, Any] = result.structured_data
        if result.name == "load_pages":
            raw_pages = data.get("page_numbers")
            if not isinstance(raw_pages, list):
                raise QAToolSelectionError("load_pages result has invalid data")
            page_numbers.extend(raw_pages)
        elif result.name == "load_neighbor_pages":
            raw_offsets = data.get("offsets")
            if not isinstance(raw_offsets, list):
                raise QAToolSelectionError("load_neighbor_pages result has invalid data")
            page_offsets.extend(raw_offsets)
        elif result.name == "search_slides":
            query = data.get("semantic_query")
            if not isinstance(query, str) or not query.strip():
                raise QAToolSelectionError("search_slides result has invalid data")
            queries.append(query.strip())
        else:
            raise QAToolSelectionError(f"unknown Q&A result: {result.name}")

    if len(queries) > 2:
        raise QAToolSelectionError("too many search_slides calls")

    return QAContextPlan(
        include_current_page=True,
        page_numbers=list(dict.fromkeys(page_numbers)),
        page_offsets=list(dict.fromkeys(page_offsets)),
        # v1 keeps one deterministic query because QAContextPlan has one field.
        semantic_query=queries[0] if queries else "",
        use_memory=True,
    )

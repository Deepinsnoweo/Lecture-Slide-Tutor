"""Worker-side Explain/Translate/Ask pipelines with no Streamlit dependency."""

from __future__ import annotations

import time

from slide_tutor import config
from slide_tutor.agent_policy import agent_policy_from_config
from slide_tutor.index_store import ensure_index_via_job
from slide_tutor.llm import stream_text
from slide_tutor.models import (
    QAContextBundle,
    QAContextSource,
    RetrievalScope,
    ToolTrace,
)
from slide_tutor.prompts import (
    PROMPT_EXPLAIN_PAGE,
    PROMPT_FREE_QA,
    PROMPT_TRANSLATE_PAGE,
    build_qa_system_prompt,
)
from slide_tutor.qa_context import (
    build_qa_context_bundle,
    plan_qa_context_with_tools,
)
from slide_tutor.qa_agent import run_free_ask_agent
from slide_tutor.qa_jobs import QAJobRequest, QAJobResult
from slide_tutor.retrieval import format_context, retrieve_for_current_page
from slide_tutor.page_understanding import get_page_text_for_prompt


def _single_page_bundle(request: QAJobRequest, text: str, *, reason: str) -> QAContextBundle:
    return QAContextBundle(
        prompt_context=f"--- Selected page ---\n{text}",
        sources=[QAContextSource(
            kind="selected_page",
            source_file=request.source_file,
            page_number=request.page_number,
            reason=reason,
            snippet=text[:200],
        )],
        memory_turns=[],
        planner_summary="selected page",
    )


def _stream_prompt(
    system_prompt: str,
    user_prompt: str,
    request: QAJobRequest,
    cancel_signal,
    emit,
    *,
    context_bundle: QAContextBundle | None,
    quality_warning: str | None = None,
    metrics_callback=None,
    stream_start_callback=None,
) -> QAJobResult:
    parts: list[str] = []
    if stream_start_callback is not None:
        stream_start_callback(time.perf_counter())
    for delta in stream_text(
        system_prompt,
        user_prompt,
        cancel_signal=cancel_signal,
        metrics_callback=metrics_callback,
    ):
        parts.append(delta)
        emit(delta)
    return QAJobResult(
        answer_markdown="".join(parts),
        context_bundle=context_bundle,
        quality_warning=quality_warning,
    )


def _agent_tool_traces(summary) -> list[ToolTrace]:
    traces: list[ToolTrace] = []
    for round_trace in summary.round_traces:
        traces.extend(round_trace.tool_traces)
    return traces


def _v1_ask_bundle(request: QAJobRequest, index):
    planning = plan_qa_context_with_tools(
        request.question,
        request.current_slide,
        request.file_page_count,
        request.memory,
    )
    bundle = build_qa_context_bundle(
        planning.plan,
        request.project_id,
        request.current_slide,
        request.file_id,
        request.file_page_count,
        request.memory,
        index,
        original_question=request.question,
    )
    return planning, bundle.model_copy(update={
        "planner_mode": planning.mode,
        "tool_traces": planning.tool_traces,
    })


def run_qa_pipeline(
    request: QAJobRequest,
    cancel_signal,
    emit,
    *,
    metrics_callback=None,
    stream_start_callback=None,
    agent_enabled: bool | None = None,
) -> QAJobResult:
    """Execute the immutable request snapshot for one of three actions."""
    page_resolution = get_page_text_for_prompt(
        request.project_id, request.file_id, request.current_slide,
    )
    page_text = page_resolution.text
    quality = page_resolution.quality
    quality_warning = page_resolution.warning or None
    if quality in {"fallback", "degraded", "sparse_valid", "needs_vlm", "empty"}:
        quality_warning = " · ".join(filter(None, [
            quality_warning, f"Page content quality: {quality}",
        ]))

    if request.action_type == "translate":
        bundle = _single_page_bundle(request, page_text, reason="Translation source page")
        prompt = PROMPT_TRANSLATE_PAGE.format(
            target_language=request.language,
            source_file=request.source_file,
            page_number=request.page_number,
            page_text=page_text,
        )
        return _stream_prompt(
            build_qa_system_prompt(request.language, translator=True), prompt, request,
            cancel_signal, emit, context_bundle=bundle, quality_warning=quality_warning,
            metrics_callback=metrics_callback,
            stream_start_callback=stream_start_callback,
        )

    index = ensure_index_via_job(request.project_id, cancel_signal=cancel_signal)
    if request.action_type == "explain":
        chunks = retrieve_for_current_page(
            request.current_slide,
            index,
            scope=RetrievalScope(
                kind="current_file", project_id=request.project_id, file_id=request.file_id,
            ),
        )
        context = format_context(chunks)
        sources = [QAContextSource(
            kind="selected_page" if chunk.slide_id == request.current_slide.slide_id else "retrieved_page",
            source_file=chunk.source_file,
            page_number=chunk.page_number,
            reason="Explain current page context",
            snippet=chunk.text[:200],
            retrieval_methods=chunk.retrieval_methods,
        ) for chunk in chunks]
        bundle = QAContextBundle(
            prompt_context=context,
            sources=sources,
            planner_summary="current page + deterministic retrieval",
        )
        prompt = PROMPT_EXPLAIN_PAGE.format(
            target_language=request.language,
            context=context,
            source_file=request.source_file,
            page_number=request.page_number,
            page_text=page_text,
        )
        return _stream_prompt(
            build_qa_system_prompt(request.language), prompt, request,
            cancel_signal, emit, context_bundle=bundle, quality_warning=quality_warning,
            metrics_callback=metrics_callback,
            stream_start_callback=stream_start_callback,
        )

    agent_summary = None
    agent_enabled_for_request = (
        config.QA_AGENT_ENABLED if agent_enabled is None else agent_enabled
    )
    if agent_enabled_for_request:
        agent_result = run_free_ask_agent(
            question=request.question,
            current_slide=request.current_slide,
            max_page=request.file_page_count,
            memory=request.memory,
            index=index,
            policy=agent_policy_from_config(),
            cancel_signal=cancel_signal,
        )
        agent_summary = agent_result.summary
        if agent_result.fallback_requested:
            planning, bundle = _v1_ask_bundle(request, index)
            bundle = bundle.model_copy(update={
                "planner_mode": planning.mode,
                "tool_traces": planning.tool_traces,
                "agent_summary": agent_summary,
            })
        else:
            planning = None
            bundle = agent_result.bundle.model_copy(update={
                "planner_mode": "bounded_agent_v2",
                "tool_traces": _agent_tool_traces(agent_summary),
                "agent_summary": agent_summary,
            })
    else:
        planning, bundle = _v1_ask_bundle(request, index)
    plan = planning.plan if planning is not None else None
    ambiguity_note = ""
    if plan is not None and plan.ambiguity_note:
        ambiguity_note = (
            f"\nTopic ambiguity: {plan.ambiguity_note}\n"
            "Address the ambiguity explicitly and stay grounded in the supplied context."
        )
    prompt = PROMPT_FREE_QA.format(
        target_language=request.language,
        context=bundle.prompt_context,
        question=request.question,
        chat_history="",
        ambiguity_note=ambiguity_note,
    )
    return _stream_prompt(
        build_qa_system_prompt(request.language), prompt, request,
        cancel_signal, emit, context_bundle=bundle, quality_warning=quality_warning,
        metrics_callback=metrics_callback,
        stream_start_callback=stream_start_callback,
    )

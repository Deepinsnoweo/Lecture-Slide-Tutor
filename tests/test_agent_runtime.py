from __future__ import annotations

import time
from threading import Event

import pytest

from slide_tutor import projects, qa_context, retrieval
from slide_tutor.agent_policy import AgentExecutionPolicy
from slide_tutor.agent_runtime import AgentCancelled, run_bounded_agent
from slide_tutor.models import (
    AgentMessage,
    AgentModelTurn,
    LLMCallMetrics,
    RetrievalChunk,
    SlidePage,
    ToolCall,
)
from slide_tutor.qa_agent import (
    create_agent_accumulator,
    execute_qa_selection_calls,
    validate_qa_selection_calls,
)
from slide_tutor.llm import LLMAgentTurnUnavailable


def _slide(page: int, text: str | None = None) -> SlidePage:
    return SlidePage(
        slide_id=f"slide-{page}",
        module_id="project-1",
        project_id="project-1",
        file_id="file-1",
        source_file="Lecture01.pdf",
        source_type="pdf",
        page_number=page,
        global_index=page,
        text=text or f"Page {page} text",
        text_quality="ok",
    )


def _chunk(page: int, text: str) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=f"chunk-{page}",
        project_id="project-1",
        file_id="file-1",
        slide_id=f"slide-{page}",
        source_file="Lecture01.pdf",
        page_number=page,
        text=text,
        snippet_reference=text[:80],
    )


def _accumulator(monkeypatch, *, policy=None, retrieved=None):
    slides = [_slide(page) for page in range(1, 21)]
    monkeypatch.setattr(projects, "load_file_slides", lambda *args: slides)
    monkeypatch.setattr(
        qa_context,
        "_slide_text_for_context",
        lambda _project, _file, slide: slide.text,
    )
    monkeypatch.setattr(
        retrieval,
        "retrieve_for_question",
        lambda *args, **kwargs: retrieved or [],
    )
    chosen_policy = policy or AgentExecutionPolicy()
    accumulator = create_agent_accumulator(
        "Explain this page",
        slides[4],
        20,
        [],
        ([], object(), object()),
        chosen_policy,
    )
    return accumulator


def _controller(turns, *, usage=(10, 5, 15), seen=None, on_call=None):
    remaining = list(turns)

    def call(messages, tools, *, model=None, metrics_callback=None):
        if seen is not None:
            seen.append(list(messages))
        if on_call is not None:
            on_call()
        if metrics_callback is not None:
            metrics_callback(LLMCallMetrics(
                operation="agent_turn",
                model=model or "agent",
                status="success",
                latency_ms=1,
                prompt_tokens=usage[0] if usage else None,
                completion_tokens=usage[1] if usage else None,
                total_tokens=usage[2] if usage else None,
            ))
        if not remaining:
            raise AssertionError("controller called more times than expected")
        return remaining.pop(0)

    return call


def _run(monkeypatch, turns, *, policy=None, controller=None, cancel=None, seen=None):
    accumulator = _accumulator(monkeypatch, policy=policy)
    selected_controller = controller or _controller(turns, seen=seen)
    result = run_bounded_agent(
        initial_messages=[AgentMessage(role="user", content="question")],
        tools=[],
        context=accumulator,
        execute_calls=lambda calls: execute_qa_selection_calls(
            calls,
            accumulator=accumulator,
        ),
        validate_calls=lambda calls: validate_qa_selection_calls(
            calls,
            accumulator=accumulator,
        ),
        policy=policy or AgentExecutionPolicy(),
        controller=selected_controller,
        cancel_signal=cancel,
    )
    return result, accumulator


def _read(call_id="call-1", page=7):
    return AgentModelTurn(tool_calls=[ToolCall(
        call_id=call_id,
        name="load_pages",
        arguments={"page_numbers": [page]},
    )])


def _search(call_id="call-2"):
    return AgentModelTurn(tool_calls=[ToolCall(
        call_id=call_id,
        name="search_slides",
        arguments={"query": "concept X"},
    )])


def _finish(call_id="finish-1"):
    return AgentModelTurn(tool_calls=[ToolCall(
        call_id=call_id,
        name="finish_answer",
        arguments={},
    )])


def test_two_round_read_then_finish_observes_real_result(monkeypatch):
    seen = []
    result, accumulator = _run(monkeypatch, [_read(), _finish()], seen=seen)

    assert result.summary.stop_reason == "finish_requested"
    assert result.summary.rounds == 2
    assert result.summary.executed_tool_calls == 1
    assert result.summary.mode == "bounded_agent_v2"
    assert any(source.page_number == 7 for source in accumulator.bundle.sources)
    assert "Page 7 text" in seen[1][-1].content


def test_three_round_page_search_finish_preserves_previous_observation(monkeypatch):
    retrieved = [_chunk(9, "concept X evidence")]
    accumulator = _accumulator(monkeypatch, retrieved=retrieved)
    seen = []
    controller = _controller([_read(), _search(), _finish()], seen=seen)
    result = run_bounded_agent(
        initial_messages=[AgentMessage(role="user", content="question")],
        tools=[],
        context=accumulator,
        execute_calls=lambda calls: execute_qa_selection_calls(calls, accumulator=accumulator),
        validate_calls=lambda calls: validate_qa_selection_calls(calls, accumulator=accumulator),
        policy=AgentExecutionPolicy(),
        controller=controller,
    )

    assert result.summary.rounds == 3
    assert result.summary.stop_reason == "finish_requested"
    assert result.summary.executed_tool_calls == 2
    assert any(source.page_number == 9 for source in accumulator.bundle.sources)
    assert "Page 7 text" in seen[1][-1].content
    assert "concept X evidence" in seen[2][-1].content


def test_multi_call_round_returns_one_tool_message_per_call(monkeypatch):
    seen = []
    multi = AgentModelTurn(tool_calls=[
        ToolCall(call_id="c1", name="load_pages", arguments={"page_numbers": [7]}),
        ToolCall(call_id="c2", name="load_neighbor_pages", arguments={"offsets": [1]}),
    ])
    result, _ = _run(monkeypatch, [multi, _finish()], seen=seen)

    assert result.summary.stop_reason == "finish_requested"
    assert result.summary.executed_tool_calls == 2
    assert [message.tool_call_id for message in seen[1][-2:]] == ["c1", "c2"]


def test_first_round_finish_is_valid_without_progress(monkeypatch):
    result, _ = _run(monkeypatch, [_finish()])
    assert result.fallback_requested is False
    assert result.summary.stop_reason == "finish_requested"
    assert result.summary.executed_tool_calls == 0


def test_max_rounds_forces_final(monkeypatch):
    policy = AgentExecutionPolicy(max_rounds=2, max_total_tool_calls=4)
    result, _ = _run(monkeypatch, [_read("c1", 7), _read("c2", 8)], policy=policy)
    assert result.summary.stop_reason == "max_rounds"
    assert result.summary.rounds == 2
    assert result.summary.executed_tool_calls == 2


def test_total_and_per_round_limits_do_not_execute_over_limit_batch(monkeypatch):
    policy = AgentExecutionPolicy(max_calls_per_round=1, max_total_tool_calls=1)
    result, accumulator = _run(
        monkeypatch,
        [AgentModelTurn(tool_calls=[
            ToolCall(call_id="c1", name="load_pages", arguments={"page_numbers": [7]}),
            ToolCall(call_id="c2", name="load_pages", arguments={"page_numbers": [8]}),
        ])],
        policy=policy,
    )
    assert result.fallback_requested is False
    assert result.summary.stop_reason == "tool_exhausted"
    assert accumulator.executed_call_count == 0


def test_malformed_arguments_first_round_fallback_and_after_progress_forces_final(
    monkeypatch,
):
    malformed = AgentModelTurn.model_construct(tool_calls=[ToolCall(
        call_id="bad",
        name="load_pages",
        arguments={"page_numbers": ["not-an-int"]},
    )])
    first, first_accumulator = _run(monkeypatch, [malformed])
    assert first.fallback_requested is True
    assert first.summary.stop_reason == "fallback_v1"
    assert first_accumulator.executed_call_count == 0

    after, after_accumulator = _run(monkeypatch, [_read(), malformed])
    assert after.fallback_requested is False
    assert after.summary.stop_reason == "protocol_error_after_progress"
    assert after_accumulator.executed_call_count == 1


def test_unknown_tool_is_rejected_without_execution(monkeypatch):
    unknown = AgentModelTurn(tool_calls=[ToolCall(
        call_id="unknown",
        name="delete_everything",
        arguments={},
    )])
    result, accumulator = _run(monkeypatch, [unknown])
    assert result.fallback_requested is True
    assert result.summary.stop_reason == "fallback_v1"
    assert accumulator.executed_call_count == 0


def test_out_of_range_page_is_error_observation_and_can_finish(monkeypatch):
    invalid_page = AgentModelTurn(tool_calls=[ToolCall(
        call_id="bad-page",
        name="load_pages",
        arguments={"page_numbers": [999]},
    )])
    seen = []
    result, accumulator = _run(monkeypatch, [invalid_page, _finish()], seen=seen)
    assert result.summary.stop_reason == "finish_requested"
    assert accumulator.executed_call_count == 1
    assert "tool handler failed" in seen[1][-1].content


def test_no_new_search_sources_is_legal_observation(monkeypatch):
    accumulator = _accumulator(monkeypatch, retrieved=[_chunk(5, "current evidence")])
    seen = []
    controller = _controller([_search(), _finish()], seen=seen)
    result = run_bounded_agent(
        initial_messages=[AgentMessage(role="user", content="question")],
        tools=[],
        context=accumulator,
        execute_calls=lambda calls: execute_qa_selection_calls(calls, accumulator=accumulator),
        validate_calls=lambda calls: validate_qa_selection_calls(calls, accumulator=accumulator),
        policy=AgentExecutionPolicy(),
        controller=controller,
    )
    assert result.summary.stop_reason == "finish_requested"
    assert "no new context" in seen[1][-1].content.lower()


def test_exact_token_budget_stops_before_tool_execution(monkeypatch):
    policy = AgentExecutionPolicy(max_controller_tokens=15)
    result, accumulator = _run(monkeypatch, [_read()], policy=policy)
    assert result.summary.stop_reason == "token_budget"
    assert accumulator.executed_call_count == 0
    assert result.summary.usage_status == "exact"


def test_time_budget_stops_at_boundary_without_claiming_http_was_killed(monkeypatch):
    policy = AgentExecutionPolicy(max_agent_seconds=0.01)

    def slow_controller(*args, **kwargs):
        time.sleep(0.02)
        return _read()

    result, accumulator = _run(monkeypatch, [], policy=policy, controller=slow_controller)
    assert result.summary.stop_reason == "time_budget"
    assert accumulator.executed_call_count == 0


def test_usage_unavailable_still_obeys_round_budget(monkeypatch):
    policy = AgentExecutionPolicy(
        max_rounds=1,
        max_calls_per_round=2,
        max_total_tool_calls=2,
    )
    result, accumulator = _run(
        monkeypatch,
        [_read()],
        policy=policy,
        controller=_controller([_read()], usage=None),
    )
    assert result.summary.stop_reason == "max_rounds"
    assert result.summary.usage_status == "unavailable"
    assert accumulator.executed_call_count == 1


def test_duplicate_call_id_after_progress_is_protocol_error(monkeypatch):
    result, accumulator = _run(
        monkeypatch,
        [_read("same", 7), _read("same", 8)],
    )
    assert result.summary.stop_reason == "protocol_error_after_progress"
    assert result.fallback_requested is False
    assert accumulator.executed_call_count == 1


def test_duplicate_arguments_return_dedup_observation(monkeypatch):
    seen = []
    result, accumulator = _run(
        monkeypatch,
        [_read("c1", 7), _read("c2", 7), _finish()],
        seen=seen,
    )
    assert result.summary.stop_reason == "finish_requested"
    assert accumulator.executed_call_count == 1
    assert "Duplicate selection" in seen[2][-1].content


def test_finish_mixed_with_read_is_protocol_error(monkeypatch):
    invalid = AgentModelTurn.model_construct(tool_calls=[
        ToolCall(call_id="finish", name="finish_answer", arguments={}),
        ToolCall(call_id="read", name="load_pages", arguments={"page_numbers": [7]}),
    ])
    result, accumulator = _run(monkeypatch, [invalid])
    assert result.summary.stop_reason == "fallback_v1"
    assert accumulator.executed_call_count == 0


def test_cancellation_before_provider_does_not_fallback(monkeypatch):
    cancel = Event()
    cancel.set()
    called = []

    def controller(*args, **kwargs):
        called.append(True)
        return _finish()

    with pytest.raises(AgentCancelled):
        _run(monkeypatch, [_finish()], controller=controller, cancel=cancel)
    assert called == []


def test_cancellation_after_provider_and_between_tools_is_cooperative(monkeypatch):
    cancel = Event()
    after_provider = _controller([_read()], on_call=cancel.set)
    with pytest.raises(AgentCancelled):
        _run(monkeypatch, [_read()], controller=after_provider, cancel=cancel)

    cancel.clear()
    accumulator = _accumulator(monkeypatch)
    call_count = 0

    def execute(calls):
        nonlocal call_count
        call_count += 1
        result = execute_qa_selection_calls(calls, accumulator=accumulator)
        if call_count == 1:
            cancel.set()
        return result

    with pytest.raises(AgentCancelled):
        run_bounded_agent(
            initial_messages=[AgentMessage(role="user", content="q")],
            tools=[],
            context=accumulator,
            execute_calls=execute,
            validate_calls=lambda calls: validate_qa_selection_calls(calls, accumulator=accumulator),
            policy=AgentExecutionPolicy(),
            controller=_controller([AgentModelTurn(tool_calls=[
                ToolCall(call_id="c1", name="load_pages", arguments={"page_numbers": [7]}),
                ToolCall(call_id="c2", name="load_pages", arguments={"page_numbers": [8]}),
            ])]),
            cancel_signal=cancel,
        )
    assert call_count == 1


def test_provider_error_before_and_after_progress_have_different_results(monkeypatch):
    def fail(*args, **kwargs):
        raise LLMAgentTurnUnavailable("unsupported")

    first, _ = _run(monkeypatch, [], controller=fail)
    assert first.fallback_requested is True
    assert first.summary.stop_reason == "fallback_v1"

    controller = _controller([_read()])
    calls = [0]

    def fail_after(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            return _read()
        raise LLMAgentTurnUnavailable("temporary")

    after, accumulator = _run(monkeypatch, [], controller=fail_after)
    assert after.fallback_requested is False
    assert after.summary.stop_reason == "provider_error_after_progress"
    assert accumulator.executed_call_count == 1


def test_failure_summary_does_not_leak_keys_paths_or_raw_provider_text(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("api_key=secret /Users/example/private.pdf raw response")

    result, _ = _run(monkeypatch, [], controller=fail)
    assert result.summary.fallback_reason is not None
    assert "secret" not in result.summary.fallback_reason
    assert "/Users" not in result.summary.fallback_reason
    assert "raw response" not in result.summary.fallback_reason

from __future__ import annotations

from types import SimpleNamespace

import pytest

from slide_tutor import llm
from slide_tutor.models import AgentMessage, ToolCall


class _Client:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def _response(*calls):
    tool_calls = [
        SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(name=name, arguments=arguments),
        )
        for call_id, name, arguments in calls
    ]
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=None, tool_calls=tool_calls),
        )],
        usage=SimpleNamespace(
            prompt_tokens=40,
            completion_tokens=10,
            total_tokens=50,
        ),
    )


def _tools():
    return [{
        "type": "function",
        "function": {
            "name": "load_pages",
            "description": "Load pages",
            "parameters": {
                "type": "object",
                "properties": {"page_numbers": {"type": "array"}},
                "required": ["page_numbers"],
                "additionalProperties": False,
            },
        },
    }]


def test_complete_tool_calls_sends_tools_and_returns_normalized_calls(monkeypatch):
    client = _Client(_response(
        ("call-1", "load_pages", '{"page_numbers": [3]}'),
        ("call-2", "load_pages", '{"page_numbers": [4]}'),
    ))
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    monkeypatch.setattr(llm.config, "QA_PLANNER_MODEL", "planner-light")

    metrics = []
    calls = llm.complete_tool_calls(
        "system", "question", _tools(), metrics_callback=metrics.append,
    )

    assert [call.call_id for call in calls] == ["call-1", "call-2"]
    assert [call.arguments for call in calls] == [
        {"page_numbers": [3]}, {"page_numbers": [4]},
    ]
    assert client.kwargs["model"] == "planner-light"
    assert client.kwargs["tools"] == _tools()
    assert client.kwargs["tool_choice"] == "auto"
    assert client.kwargs["temperature"] == 0.0
    assert "reasoning_effort" not in client.kwargs
    assert metrics[0].operation == "tool_calls"
    assert metrics[0].model == "planner-light"
    assert metrics[0].total_tokens == 50
    assert metrics[0].status == "success"


def test_complete_tool_calls_classifies_empty_calls_as_capability_fallback(monkeypatch):
    client = _Client(SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer", tool_calls=[]))],
    ))
    monkeypatch.setattr(llm, "_get_client", lambda: client)

    with pytest.raises(llm.LLMToolCallingUnavailable, match="no tool calls"):
        llm.complete_tool_calls("system", "question", _tools())


def test_complete_tool_calls_classifies_provider_errors(monkeypatch):
    client = _Client(error=RuntimeError("tools unsupported"))
    monkeypatch.setattr(llm, "_get_client", lambda: client)

    metrics = []
    with pytest.raises(llm.LLMToolCallingUnavailable, match="Provider does not support"):
        llm.complete_tool_calls(
            "system", "question", _tools(), metrics_callback=metrics.append,
        )
    assert metrics[0].status == "error"
    assert metrics[0].error_type == "RuntimeError"


def test_complete_tool_calls_rejects_malformed_provider_arguments(monkeypatch):
    client = _Client(_response(("call-1", "load_pages", "not-json")))
    monkeypatch.setattr(llm, "_get_client", lambda: client)

    with pytest.raises(llm.LLMToolCallingUnavailable, match="invalid tool-call"):
        llm.complete_tool_calls("system", "question", _tools())


def test_complete_text_reports_provider_usage(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=" answer "),
        )],
        usage={
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "total_tokens": 15,
        },
    )
    client = _Client(response)
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    metrics = []

    answer = llm.complete_text(
        "system", "question", model="answer-model",
        metrics_callback=metrics.append,
    )

    assert answer == "answer"
    assert client.kwargs["model"] == "answer-model"
    assert metrics[0].operation == "text"
    assert metrics[0].total_tokens == 15


def _agent_response(*calls, content=None, usage=None, finish_reason="tool_calls"):
    raw_calls = [
        SimpleNamespace(
            id=call_id,
            type="function",
            function=SimpleNamespace(name=name, arguments=arguments),
        )
        for call_id, name, arguments in calls
    ]
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=raw_calls),
            finish_reason=finish_reason,
        )],
    )
    if usage is not None:
        response.usage = usage
    return response


def test_complete_agent_turn_serializes_round_history_and_returns_calls(monkeypatch):
    response = _agent_response(
        ("call-2", "search_slides", '{"query":"concept X"}'),
        usage=SimpleNamespace(prompt_tokens=80, completion_tokens=12, total_tokens=92),
    )
    client = _Client(response)
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    monkeypatch.setattr(llm.config, "QA_AGENT_MODEL", "agent-light")
    messages = [
        AgentMessage(role="system", content="system"),
        AgentMessage(role="user", content="question"),
        AgentMessage(
            role="assistant",
            tool_calls=[ToolCall(
                call_id="call-1",
                name="load_pages",
                arguments={"page_numbers": [12]},
            )],
        ),
        AgentMessage(
            role="tool",
            tool_call_id="call-1",
            content="page 12 contains concept X",
        ),
    ]

    metrics = []
    turn = llm.complete_agent_turn(
        messages,
        _tools(),
        metrics_callback=metrics.append,
    )

    assert turn.tool_calls[0].call_id == "call-2"
    assert turn.tool_calls[0].arguments == {"query": "concept X"}
    sent = client.kwargs["messages"]
    assert sent[2]["role"] == "assistant"
    assert sent[2]["tool_calls"][0]["id"] == "call-1"
    assert sent[2]["tool_calls"][0]["function"]["arguments"] == (
        '{"page_numbers":[12]}'
    )
    assert sent[3] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "page 12 contains concept X",
    }
    assert client.kwargs["model"] == "agent-light"
    assert client.kwargs["tool_choice"] == "auto"
    assert metrics[0].operation == "agent_turn"
    assert metrics[0].total_tokens == 92


def test_complete_agent_turn_accepts_missing_usage(monkeypatch):
    client = _Client(_agent_response(
        ("call-1", "load_pages", '{"page_numbers":[1]}'),
    ))
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    metrics = []

    turn = llm.complete_agent_turn(
        [AgentMessage(role="user", content="question")],
        _tools(),
        metrics_callback=metrics.append,
    )

    assert turn.tool_calls[0].name == "load_pages"
    assert metrics[0].status == "success"
    assert metrics[0].total_tokens is None


def test_complete_agent_turn_classifies_empty_choices_and_malformed_calls(monkeypatch):
    empty = _Client(SimpleNamespace(choices=[]))
    monkeypatch.setattr(llm, "_get_client", lambda: empty)
    with pytest.raises(llm.LLMAgentTurnUnavailable, match="no choices"):
        llm.complete_agent_turn([AgentMessage(role="user", content="q")], _tools())

    malformed = _Client(_agent_response(
        ("call-1", "load_pages", "not-json"),
    ))
    monkeypatch.setattr(llm, "_get_client", lambda: malformed)
    with pytest.raises(llm.LLMAgentTurnProtocolError, match="malformed"):
        llm.complete_agent_turn([AgentMessage(role="user", content="q")], _tools())


def test_complete_agent_turn_returns_assistant_text_for_runtime_protocol_check(monkeypatch):
    client = _Client(_agent_response(content="I should inspect the page", finish_reason="stop"))
    monkeypatch.setattr(llm, "_get_client", lambda: client)

    turn = llm.complete_agent_turn(
        [AgentMessage(role="user", content="q")],
        _tools(),
    )

    assert turn.tool_calls == []
    assert turn.assistant_text == "I should inspect the page"
    assert turn.provider_finish_reason == "stop"

from __future__ import annotations

from threading import Event
from types import SimpleNamespace

import pytest

from slide_tutor import llm


class _Response:
    def __init__(self, chunks, error=None):
        self.chunks = iter(chunks)
        self.error = error
        self.closed = False

    def __iter__(self):
        if self.error:
            raise self.error
        return self

    def __next__(self):
        return next(self.chunks)

    def close(self):
        self.closed = True


def _chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def _usage_chunk(prompt=11, completion=7, total=18):
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        ),
    )


class _Client:
    def __init__(self, response):
        self.response = response
        self.kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_stream_text_stops_before_next_chunk_and_closes_response(monkeypatch):
    response = _Response([_chunk("a"), _chunk("b")])
    monkeypatch.setattr(llm, "_get_client", lambda: _Client(response))
    cancel = Event()
    metrics = []
    stream = llm.stream_text(
        "system", "user", cancel_signal=cancel, metrics_callback=metrics.append,
    )
    assert next(stream) == "a"
    cancel.set()
    with pytest.raises(llm.LLMCancelled):
        next(stream)
    assert response.closed is True
    assert len(metrics) == 1
    assert metrics[0].status == "cancelled"
    assert metrics[0].streamed is True
    assert metrics[0].first_token_latency_ms is not None


def test_stream_text_provider_error_is_not_cancelled_and_closes(monkeypatch):
    response = _Response([], error=RuntimeError("provider failure"))
    monkeypatch.setattr(llm, "_get_client", lambda: _Client(response))
    with pytest.raises(llm.LLMError) as exc_info:
        list(llm.stream_text("system", "user"))
    assert not isinstance(exc_info.value, llm.LLMCancelled)
    assert response.closed is True


def test_stream_text_records_first_nonempty_delta_and_exact_usage(monkeypatch):
    response = _Response([_chunk(""), _chunk("answer"), _usage_chunk()])
    monkeypatch.setattr(llm, "_get_client", lambda: _Client(response))
    metrics = []

    assert list(llm.stream_text("system", "user", metrics_callback=metrics.append)) == [
        "answer",
    ]
    assert len(metrics) == 1
    assert metrics[0].status == "success"
    assert metrics[0].first_token_latency_ms is not None
    assert metrics[0].prompt_tokens == 11
    assert metrics[0].completion_tokens == 7
    assert metrics[0].total_tokens == 18


def test_stream_text_cancellation_before_first_token_is_not_provider_error(monkeypatch):
    response = _Response([_chunk("answer")])
    monkeypatch.setattr(llm, "_get_client", lambda: _Client(response))
    cancel = Event()
    cancel.set()
    metrics = []

    with pytest.raises(llm.LLMCancelled):
        list(llm.stream_text(
            "system", "user", cancel_signal=cancel, metrics_callback=metrics.append,
        ))
    assert len(metrics) == 1
    assert metrics[0].status == "cancelled"
    assert metrics[0].first_token_latency_ms is None


def test_stream_text_pins_deepseek_v4_main_model_to_high_effort(monkeypatch):
    response = _Response([_chunk("answer")])
    client = _Client(response)
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    monkeypatch.setattr(llm.config, "LLM_MODEL", "deepseek-v4-pro[1m]")

    assert list(llm.stream_text("system", "user")) == ["answer"]
    assert client.kwargs["reasoning_effort"] == "high"
    assert client.kwargs["extra_body"] == {"thinking": {"type": "enabled"}}


def test_thinking_options_do_not_apply_to_explicit_planner_model():
    assert llm._main_model_thinking_options(
        "deepseek-v4-pro", enabled=False,
    ) == {}

"""LLM provider abstraction using the OpenAI-compatible API.

Works with any OpenAI-compatible endpoint (OpenAI, Groq, Ollama, DeepSeek, etc.).
Configuration is read from env vars (see slide_tutor.config).
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Generator
from threading import Event
from typing import Any

from openai import OpenAI

from slide_tutor import config
from slide_tutor.models import AgentMessage, AgentModelTurn, LLMCallMetrics, ToolCall
from slide_tutor.tool_runtime import ToolRuntimeError, normalize_tool_calls

logger = logging.getLogger(__name__)
LLMMetricsCallback = Callable[[LLMCallMetrics], None]


def _main_model_thinking_options(
    model: str, *, enabled: bool, reasoning_effort: str = "high",
) -> dict[str, Any]:
    """Pin DeepSeek V4 main-model calls to thinking/high without affecting planners."""
    normalized = model.lower().rsplit("/", 1)[-1]
    if not enabled or not normalized.startswith("deepseek-v4-"):
        return {}
    return {
        "reasoning_effort": reasoning_effort,
        "extra_body": {"thinking": {"type": "enabled"}},
    }


class LLMError(Exception):
    """Raised when the LLM is unreachable, unconfigured, or returns an error."""


class LLMCancelled(LLMError):
    """Raised when a caller cooperatively stops a streaming response."""


class LLMToolCallingUnavailable(LLMError):
    """Raised when a provider cannot return a valid tool-call selection."""


class LLMAgentTurnUnavailable(LLMToolCallingUnavailable):
    """Raised when a Provider cannot return a usable Agent controller turn."""


class LLMAgentTurnProtocolError(LLMError):
    """Raised when a Provider response violates the Agent turn contract."""


def _get_client() -> OpenAI:
    """Build an OpenAI client from the current env-var configuration."""
    return OpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        timeout=120.0,  # quiz generation can take longer than Q&A
    )


def _usage_values(response: Any) -> tuple[int | None, int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None and all(
        hasattr(response, name)
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
    ):
        usage = response
    if usage is None:
        return None, None, None
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
    return prompt_tokens, completion_tokens, total_tokens


def _emit_metrics(
    callback: LLMMetricsCallback | None,
    *,
    operation: str,
    model: str,
    started: float,
    response: Any = None,
    attempt: int = 1,
    error: Exception | None = None,
) -> None:
    if callback is None:
        return
    prompt_tokens, completion_tokens, total_tokens = _usage_values(response)
    try:
        callback(LLMCallMetrics(
            operation=operation,
            model=model,
            attempt=attempt,
            status="error" if error is not None else "success",
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            error_type=type(error).__name__ if error is not None else None,
        ))
    except Exception as exc:
        logger.debug("Unable to emit LLM metrics: %s", type(exc).__name__)


def _emit_stream_metrics(
    callback: LLMMetricsCallback | None,
    *,
    model: str,
    started: float,
    first_token_at: float | None,
    usage_source: Any = None,
    status: str,
    error: Exception | None = None,
) -> None:
    """Emit exactly one sanitized metric for a streaming request."""
    if callback is None:
        return
    prompt_tokens, completion_tokens, total_tokens = _usage_values(usage_source)
    try:
        callback(LLMCallMetrics(
            operation="stream",
            model=model,
            status=status,
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            first_token_latency_ms=(
                (first_token_at - started) * 1000
                if first_token_at is not None else None
            ),
            streamed=True,
            error_type=type(error).__name__ if error is not None else None,
        ))
    except Exception as exc:
        logger.debug("Unable to emit streaming metrics: %s", type(exc).__name__)


def complete_text(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.3,
    metrics_callback: LLMMetricsCallback | None = None,
) -> str:
    """Send a chat completion request and return the assistant's text response.

    Raises LLMError with a human-readable message on failure so the UI can
    display it directly."""
    selected_model = model or config.LLM_MODEL
    started = time.perf_counter()
    response = None
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            **_main_model_thinking_options(
                selected_model, enabled=model is None,
            ),
        )
        content = response.choices[0].message.content
        _emit_metrics(
            metrics_callback,
            operation="text",
            model=selected_model,
            started=started,
            response=response,
        )
        return content.strip() if content else ""
    except Exception as exc:
        _emit_metrics(
            metrics_callback,
            operation="text",
            model=selected_model,
            started=started,
            response=response,
            error=exc,
        )
        raise LLMError(
            f"LLM call failed. Check that your model server is running at "
            f"{config.LLM_BASE_URL} and the model '{config.LLM_MODEL}' is available. "
            f"Error: {exc}"
        ) from exc


def complete_tool_calls(
    system_prompt: str,
    user_prompt: str,
    tools: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    metrics_callback: LLMMetricsCallback | None = None,
) -> list[ToolCall]:
    """Select bounded tools and return provider-neutral ``ToolCall`` values.

    This is deliberately a non-streaming planner call.  Provider SDK objects
    are normalized at this boundary; tool handlers and Q&A code never depend on
    OpenAI response classes.  Callers can treat every failure as a safe
    capability fallback to the legacy JSON planner.
    """
    selected_model = model or config.QA_PLANNER_MODEL
    started = time.perf_counter()
    response = None
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            tools=tools,
            tool_choice="auto",
            # Explicit planner calls do not inherit main-model thinking flags.
            **_main_model_thinking_options(selected_model, enabled=False),
        )
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMToolCallingUnavailable("Provider returned no choices")
        message = getattr(choices[0], "message", None)
        raw_calls = getattr(message, "tool_calls", None) if message else None
        if not raw_calls:
            raise LLMToolCallingUnavailable("Provider returned no tool calls")
        try:
            calls = normalize_tool_calls(raw_calls)
        except ToolRuntimeError as exc:
            raise LLMToolCallingUnavailable(
                "Provider returned invalid tool-call arguments"
            ) from exc
        _emit_metrics(
            metrics_callback,
            operation="tool_calls",
            model=selected_model,
            started=started,
            response=response,
        )
        return calls
    except LLMToolCallingUnavailable as exc:
        _emit_metrics(
            metrics_callback,
            operation="tool_calls",
            model=selected_model,
            started=started,
            response=response,
            error=exc,
        )
        raise
    except Exception as exc:
        _emit_metrics(
            metrics_callback,
            operation="tool_calls",
            model=selected_model,
            started=started,
            response=response,
            error=exc,
        )
        raise LLMToolCallingUnavailable(
            f"Provider does not support usable tool calls: {type(exc).__name__}"
        ) from exc


def _agent_messages_to_openai(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    """Serialize validated provider-neutral messages at the SDK boundary."""
    serialized: list[dict[str, Any]] = []
    for message in messages:
        if message.role in {"system", "user"}:
            serialized.append({
                "role": message.role,
                "content": message.content,
            })
            continue
        if message.role == "tool":
            serialized.append({
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            })
            continue

        payload: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or None,
        }
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(
                            call.arguments,
                            ensure_ascii=False,
                            allow_nan=False,
                            separators=(",", ":"),
                        ),
                    },
                }
                for call in message.tool_calls
            ]
        serialized.append(payload)
    return serialized


def complete_agent_turn(
    messages: list[AgentMessage],
    tools: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    metrics_callback: LLMMetricsCallback | None = None,
) -> AgentModelTurn:
    """Request one bounded Agent controller turn.

    The function accepts only validated provider-neutral messages and returns
    only normalized Pydantic data.  It does not execute tools or interpret
    controller prose as a final user answer.
    """
    selected_model = model or config.QA_AGENT_MODEL
    started = time.perf_counter()
    response = None
    try:
        serialized_messages = _agent_messages_to_openai(messages)
        response = _get_client().chat.completions.create(
            model=selected_model,
            messages=serialized_messages,
            temperature=temperature,
            tools=tools,
            tool_choice="auto",
            **_main_model_thinking_options(selected_model, enabled=False),
        )
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMAgentTurnUnavailable("Provider returned no choices")
        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            raise LLMAgentTurnUnavailable("Provider returned no assistant message")
        raw_calls = getattr(message, "tool_calls", None) or []
        try:
            calls = normalize_tool_calls(raw_calls) if raw_calls else []
            turn = AgentModelTurn(
                tool_calls=calls,
                assistant_text=getattr(message, "content", None) or "",
                provider_finish_reason=getattr(choice, "finish_reason", None),
            )
        except (ToolRuntimeError, ValueError) as exc:
            raise LLMAgentTurnProtocolError(
                "Provider returned malformed Agent turn"
            ) from exc
        _emit_metrics(
            metrics_callback,
            operation="agent_turn",
            model=selected_model,
            started=started,
            response=response,
        )
        return turn
    except (LLMAgentTurnUnavailable, LLMAgentTurnProtocolError) as exc:
        _emit_metrics(
            metrics_callback,
            operation="agent_turn",
            model=selected_model,
            started=started,
            response=response,
            error=exc,
        )
        raise
    except Exception as exc:
        _emit_metrics(
            metrics_callback,
            operation="agent_turn",
            model=selected_model,
            started=started,
            response=response,
            error=exc,
        )
        raise LLMAgentTurnUnavailable(
            f"Provider Agent turn failed: {type(exc).__name__}"
        ) from exc


def stream_text(
    system_prompt: str,
    user_prompt: str,
    cancel_signal: Event | None = None,
    *,
    metrics_callback: LLMMetricsCallback | None = None,
    model: str | None = None,
) -> Generator[str, None, None]:
    """Same as complete_text but yields tokens as they arrive via SSE streaming.

    Use with st.write_stream(stream_text(...)) in Streamlit for real-time
    rendering — the user sees words appear one by one instead of staring at
    a static spinner for 3-10 seconds."""
    selected_model = model or config.LLM_MODEL
    started = time.perf_counter()
    first_token_at: float | None = None
    usage_source: Any = None
    status = "success"
    caught_error: Exception | None = None
    response = None
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            stream=True,
            **_main_model_thinking_options(selected_model, enabled=model is None),
        )
        for chunk in response:
            if cancel_signal is not None and cancel_signal.is_set():
                raise LLMCancelled("Generation cancelled by user")
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage_source = chunk_usage
            choices = getattr(chunk, "choices", None) or []
            if choices:
                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None) if delta else None
                if content:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    yield content
            if cancel_signal is not None and cancel_signal.is_set():
                raise LLMCancelled("Generation cancelled by user")
    except LLMCancelled as exc:
        status = "cancelled"
        caught_error = exc
        raise
    except Exception as exc:
        status = "error"
        caught_error = exc
        raise LLMError(
            f"LLM call failed. Check that your model server is running at "
            f"{config.LLM_BASE_URL} and the model '{selected_model}' is available. "
            f"Error: {exc}"
        ) from exc
    finally:
        _emit_stream_metrics(
            metrics_callback,
            model=selected_model,
            started=started,
            first_token_at=first_token_at,
            usage_source=(
                usage_source if usage_source is not None else response
            ),
            status=status,
            error=caught_error,
        )
        close = getattr(response, "close", None)
        if callable(close):
            try:
                close()
            except Exception as close_exc:
                logger.debug("Unable to close streaming response: %s", close_exc)


def complete_json(
    system_prompt: str, user_prompt: str,
    pydantic_model: type | None = None,
    max_retries: int = 2,
    model: str | None = None,
    temperature: float = 0.3,
    metrics_callback: LLMMetricsCallback | None = None,
    reasoning_effort: str | None = None,
) -> Any:
    """Call the LLM and parse the response as JSON.

    Pipeline:
      1. Call with response_format={"type": "json_object"}.
      2. If that fails or returns invalid JSON, extract ```json … ``` blocks.
      3. Retry once with the parse error included in the prompt.
      4. Optionally validate with a Pydantic model.

    Returns the parsed dict (or Pydantic instance if pydantic_model is given).
    Raises LLMError after all retries are exhausted."""
    last_error: str | None = None
    last_raw: str = ""
    selected_model = model or config.LLM_MODEL

    for attempt in range(max_retries):
        started = time.perf_counter()
        response = None
        try:
            client = _get_client()
            # Build messages — include retry hint if this is a retry
            messages: list[dict[str, str]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            if attempt > 0 and last_error:
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your previous response was not valid JSON.  "
                        f"Error: {last_error}\n"
                        f"Please output ONLY valid JSON this time.  "
                        f"Do NOT wrap it in markdown code fences."
                    ),
                })

            response = client.chat.completions.create(
                model=selected_model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
                # An explicit model belongs to a specialized caller such as the
                # lightweight planner; only default-model JSON jobs use main-model effort.
                **_main_model_thinking_options(
                    selected_model,
                    enabled=reasoning_effort is not None or model is None,
                    reasoning_effort=reasoning_effort or "high",
                ),
            )
            raw = (response.choices[0].message.content or "").strip()
            last_raw = raw

            # Try direct parse first
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # Try to extract a ```json ... ``` block
                match = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
                if match:
                    data = json.loads(match.group(1).strip())
                else:
                    raise

            # Pydantic validation if requested
            if pydantic_model is not None:
                result = pydantic_model.model_validate(data)
            else:
                result = data
            _emit_metrics(
                metrics_callback,
                operation="json",
                model=selected_model,
                attempt=attempt + 1,
                started=started,
                response=response,
            )
            return result

        except json.JSONDecodeError as exc:
            _emit_metrics(
                metrics_callback,
                operation="json",
                model=selected_model,
                attempt=attempt + 1,
                started=started,
                response=response,
                error=exc,
            )
            last_error = f"JSON decode error: {exc}\nRaw response (first 500 chars): {last_raw[:500]}"
            logger.warning("complete_json attempt %d failed: %s", attempt + 1, exc)
        except Exception as exc:
            _emit_metrics(
                metrics_callback,
                operation="json",
                model=selected_model,
                attempt=attempt + 1,
                started=started,
                response=response,
                error=exc,
            )
            last_error = f"LLM error: {exc}"
            logger.warning("complete_json attempt %d failed: %s", attempt + 1, exc)

    raise LLMError(
        f"Failed to get valid JSON from the LLM after {max_retries} attempts.  "
        f"Last error: {last_error}"
    )

from __future__ import annotations

import pytest
from pydantic import ValidationError

from slide_tutor.agent_policy import (
    DEFAULT_AGENT_TOOLS,
    AgentExecutionPolicy,
    default_agent_policy,
)


def test_default_policy_matches_v2_contract():
    policy = default_agent_policy()

    assert policy.max_rounds == 3
    assert policy.max_calls_per_round == 4
    assert policy.max_total_tool_calls == 6
    assert policy.max_agent_seconds == 45.0
    assert policy.max_observation_chars_per_call == 2500
    assert policy.max_total_observation_chars == 9000
    assert policy.max_controller_tokens is None
    assert policy.allowed_tools == DEFAULT_AGENT_TOOLS
    assert policy.allow_state_changes is False
    assert policy.fallback_mode == "single_turn_v1"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_rounds", 0),
        ("max_rounds", 6),
        ("max_calls_per_round", 0),
        ("max_total_tool_calls", 0),
        ("max_agent_seconds", 0),
        ("max_observation_chars_per_call", 199),
        ("max_total_observation_chars", 499),
        ("max_controller_tokens", 0),
    ],
)
def test_policy_rejects_invalid_scalar_limits(field, value):
    with pytest.raises(ValidationError):
        AgentExecutionPolicy(**{field: value})


def test_policy_rejects_total_calls_below_round_limit():
    with pytest.raises(ValidationError, match="max_total_tool_calls"):
        AgentExecutionPolicy(max_calls_per_round=4, max_total_tool_calls=3)


def test_policy_rejects_total_observation_budget_below_per_call():
    with pytest.raises(ValidationError, match="max_total_observation_chars"):
        AgentExecutionPolicy(
            max_observation_chars_per_call=1000,
            max_total_observation_chars=999,
        )


def test_policy_rejects_state_changes_and_finish_control_tool():
    with pytest.raises(ValidationError, match="state-changing"):
        AgentExecutionPolicy(allow_state_changes=True)
    with pytest.raises(ValidationError):
        AgentExecutionPolicy(allowed_tools=("finish_answer",))


def test_policy_rejects_duplicate_tools_and_returns_fresh_defaults():
    with pytest.raises(ValidationError, match="unique"):
        AgentExecutionPolicy(allowed_tools=("load_pages", "load_pages"))

    first = default_agent_policy()
    second = default_agent_policy()
    assert first is not second

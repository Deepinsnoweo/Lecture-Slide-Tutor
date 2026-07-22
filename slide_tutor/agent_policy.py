"""Validated, provider-independent execution limits for Agentic Q&A v2."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


AgentReadTool = Literal["load_pages", "load_neighbor_pages", "search_slides"]
DEFAULT_AGENT_TOOLS: tuple[AgentReadTool, ...] = (
    "load_pages",
    "load_neighbor_pages",
    "search_slides",
)


class AgentExecutionPolicy(BaseModel):
    """Hard limits used by the bounded Agent runtime.

    This module intentionally has no Provider, Q&A, Streamlit, or config
    imports.  Callers may inject a tiny policy in tests without mutating
    process-wide environment settings.
    """

    max_rounds: int = Field(default=3, ge=1, le=5)
    max_calls_per_round: int = Field(default=4, ge=1, le=8)
    max_total_tool_calls: int = Field(default=6, ge=1, le=20)
    max_agent_seconds: float = Field(default=45.0, gt=0, le=300)
    max_observation_chars_per_call: int = Field(default=2500, ge=200)
    max_total_observation_chars: int = Field(default=9000, ge=500)
    max_controller_tokens: int | None = Field(default=None, ge=1)
    allowed_tools: tuple[AgentReadTool, ...] = DEFAULT_AGENT_TOOLS
    allow_state_changes: bool = False
    fallback_mode: Literal["single_turn_v1"] = "single_turn_v1"

    @field_validator("allowed_tools")
    @classmethod
    def validate_allowed_tools(
        cls,
        value: tuple[AgentReadTool, ...],
    ) -> tuple[AgentReadTool, ...]:
        if not value:
            raise ValueError("allowed_tools cannot be empty")
        if len(set(value)) != len(value):
            raise ValueError("allowed_tools must be unique")
        if "finish_answer" in value:
            raise ValueError("finish_answer is a control action, not a read tool")
        return value

    @model_validator(mode="after")
    def validate_cross_field_limits(self) -> "AgentExecutionPolicy":
        if self.max_total_tool_calls < self.max_calls_per_round:
            raise ValueError(
                "max_total_tool_calls must be >= max_calls_per_round"
            )
        if self.max_total_observation_chars < self.max_observation_chars_per_call:
            raise ValueError(
                "max_total_observation_chars must be >= "
                "max_observation_chars_per_call"
            )
        if self.allow_state_changes:
            raise ValueError("state-changing tools are disabled in Agent v2")
        return self


def default_agent_policy() -> AgentExecutionPolicy:
    """Return a fresh default policy so callers cannot mutate shared state."""

    return AgentExecutionPolicy()


def agent_policy_from_config() -> AgentExecutionPolicy:
    """Build the v2 policy from config without importing config at module load."""
    from slide_tutor import config

    token_budget = config.QA_AGENT_MAX_CONTROLLER_TOKENS
    return AgentExecutionPolicy(
        max_rounds=config.QA_AGENT_MAX_ROUNDS,
        max_calls_per_round=config.QA_AGENT_MAX_CALLS_PER_ROUND,
        max_total_tool_calls=config.QA_AGENT_MAX_TOTAL_CALLS,
        max_agent_seconds=config.QA_AGENT_MAX_SECONDS,
        max_controller_tokens=token_budget or None,
    )


__all__ = [
    "AgentExecutionPolicy",
    "AgentReadTool",
    "DEFAULT_AGENT_TOOLS",
    "default_agent_policy",
    "agent_policy_from_config",
]

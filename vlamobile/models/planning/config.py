"""Configuration for the SmolVLM planning module."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlannerConfig:
    """Configuration for `Planner`.

    Plain dataclass (not a LeRobot policy config): the planner sits above the
    VLA policy and is not itself a policy, so no registration is needed.
    """
    vlm_kwargs: dict = field(default_factory=dict)

    # History is truncated to this many most-recent entries (oldest dropped
    # first; the plan header is always kept) to bound prompt length.
    max_history_entries: int = 30

    # Planning limits.
    max_subtasks: int = 8

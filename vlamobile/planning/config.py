"""Configuration for the planning, execution and monitoring harness."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness import Skills


@dataclass
class HarnessConfig:
    """Configuration shared by the planner, the monitor and the monitoring loop."""

    # MCP server started by the sim with `--policy_type=mcp`.
    mcp_url: str = "http://127.0.0.1:8000/mcp"

    # PydanticAI model strings. The monitor answers typed decisions, so it defaults to a decision model.
    planner_model: str = "anthropic:claude-sonnet-5-5"
    monitor_model: str = "openai-decisions:gpt-6-luna"

    # Folder of Agent Skills for the planner and the monitor: one subfolder per skill, each with a `SKILL.md`.
    skills_dir: str | None = None

    # Monitoring loop. A tick starts when the previous verdict arrives, but no
    # sooner than `min_tick_period_s` after the previous tick started.
    min_tick_period_s: float = 0.0
    monitor_image_size: int = 256

    # Guards: stop a subtask after `subtask_timeout_s`, or when progress has not
    # increased over the last `stall_checks` verdicts.
    subtask_timeout_s: float = 60.0
    stall_checks: int = 5

    # Retry limits.
    max_subtask_retries: int = 2
    request_limit: int = 200  # model requests per episode, both agents combined

    def skills(self) -> list[Skills]:
        """Agent capabilities that offer the skills in `skills_dir`, if one is set."""
        if not self.skills_dir:
            return []
        return [Skills(".", workspace=LocalWorkspaceBackend(self.skills_dir))]

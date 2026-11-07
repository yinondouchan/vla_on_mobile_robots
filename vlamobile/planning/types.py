"""Data contracts of the harness: agent outputs, subtask reports and the planner's run dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import logfire
from pydantic import BaseModel, Field

from vlamobile.planning.config import HarnessConfig
from vlamobile.planning.robot import Robot


class MonitorVerdict(BaseModel):
    """Judge the current subtask from the latest camera frames."""

    status: Literal["running", "succeeded", "failed"] = Field(
        description=(
            "Has the current subtask succeeded, failed, or is it still running? "
            "succeeded: the frames show it is complete. "
            "failed: it clearly cannot complete anymore, for example the cube was dropped "
            "or knocked away, or the robot is stuck or heading away from the target. "
            "running: otherwise, including when unsure."
        )
    )
    progress: float = Field(
        ge=0,
        le=1,
        description="How complete is the current subtask, from 0 (not started) to 1 (done)?",
    )
    cube: Literal["in_gripper", "visible", "not_visible"] = Field(
        description=(
            "Where is the cube in the current frames? "
            "in_gripper: held in the gripper. "
            "visible: in view but not held. "
            "not_visible: not in the frames."
        )
    )
    motion: Literal["toward_target", "away_from_target", "stopped"] = Field(
        description=(
            "How is the robot moving relative to the current subtask's target? "
            "toward_target: closing in on the cube or the named platform. "
            "away_from_target: heading away from it. "
            "stopped: not moving."
        )
    )

    def describe(self) -> str:
        """One line of decisions, for the next tick, the log and the subtask report."""
        return f"{self.status}, progress {self.progress:.2f}, cube {self.cube}, motion {self.motion}"


class SubtaskReport(BaseModel):
    """Outcome of one `run_subtask` call, handed back to the planner."""

    subtask: str
    outcome: Literal["succeeded", "failed", "timed_out", "stalled", "interrupted"]
    duration_s: float
    checks: int
    final_progress: float
    summary: str  # the last verdicts, as decisions


class EpisodeResult(BaseModel):
    """Final output of the planner run."""

    success: bool
    summary: str
    attempts: list[SubtaskReport]


class Events:
    """Live reporting of monitor verdicts; console output comes from `logfire.configure()`."""

    def emit(self, subtask: str, verdict: MonitorVerdict) -> None:
        logfire.info("{subtask}: {verdict}", subtask=subtask, verdict=verdict.describe())


@dataclass
class HarnessDeps:
    """Dependencies of a planner run (the PydanticAI `deps_type`), shared with `run_subtask`."""

    robot: Robot
    config: HarnessConfig
    plan: list[str] = field(default_factory=list)
    attempts: list[SubtaskReport] = field(default_factory=list)
    events: Events = field(default_factory=Events)

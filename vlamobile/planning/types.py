"""Data contracts of the harness: agent outputs, subtask reports and the planner's run dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import logfire
from pydantic import BaseModel, Field

from vlamobile.planning.config import HarnessConfig
from vlamobile.planning.robot import Robot


class MonitorVerdict(BaseModel):
    """The monitor's judgement of the running subtask from the latest frames."""

    status: Literal["running", "succeeded", "failed"]
    progress: float = Field(ge=0, le=1)
    rationale: str  # one sentence, what the frames show


class SubtaskReport(BaseModel):
    """Outcome of one `run_subtask` call, handed back to the planner."""

    subtask: str
    outcome: Literal["succeeded", "failed", "timed_out", "stalled", "interrupted"]
    duration_s: float
    checks: int
    final_progress: float
    summary: str  # rationale of the last verdicts


class EpisodeResult(BaseModel):
    """Final output of the planner run."""

    success: bool
    summary: str
    attempts: list[SubtaskReport]


class Events:
    """Live reporting of monitor verdicts; console output comes from `logfire.configure()`."""

    def emit(self, subtask: str, verdict: MonitorVerdict) -> None:
        logfire.info(
            "{subtask}: {status}, progress {progress}, {rationale}",
            subtask=subtask,
            **verdict.model_dump(),
        )


@dataclass
class HarnessDeps:
    """Dependencies of a planner run (the PydanticAI `deps_type`), shared with `run_subtask`."""

    robot: Robot
    config: HarnessConfig
    plan: list[str] = field(default_factory=list)
    attempts: list[SubtaskReport] = field(default_factory=list)
    events: Events = field(default_factory=Events)

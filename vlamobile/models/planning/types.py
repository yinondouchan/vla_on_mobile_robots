"""Data types defining the planning module's contract.

These are plain dataclasses/enums shared between the planner, the prompt
templates, and callers. Everything temporal lives in `PlannerState.history`
as text lines, so past camera frames never need to be stored or re-processed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SubtaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class Subtask:
    """A single step of the decomposed plan."""

    description: str
    status: SubtaskStatus = SubtaskStatus.PENDING
    attempts: int = 0


@dataclass
class SubtaskAssessment:
    """Result of one Assess call on the current subtask."""

    progress: float  # in [0, 1]
    status: SubtaskStatus
    rationale: str


@dataclass
class PlannerState:
    """Full planner state for one episode: the plan plus a text-only history."""

    task: str
    subtasks: list[Subtask] = field(default_factory=list)
    current_index: int = 0
    history: list[str] = field(default_factory=list)

    def current_subtask(self) -> Subtask | None:
        if 0 <= self.current_index < len(self.subtasks):
            return self.subtasks[self.current_index]
        return None

    def add_history(self, line: str) -> None:
        self.history.append(line)

    def render(self, max_history_entries: int | None = None) -> str:
        """Text block injected into prompts.

        The plan header (composite task + subtask statuses) is always kept in
        full; only the history is truncated, dropping the oldest entries first
        when it exceeds ``max_history_entries``.
        """
        lines = [f"Composite task: {self.task}", "Plan:"]
        for i, subtask in enumerate(self.subtasks):
            marker = "->" if i == self.current_index else "  "
            lines.append(f"{marker} {i + 1}. [{subtask.status.value}] {subtask.description}")

        history = self.history
        dropped = 0
        if max_history_entries is not None and len(history) > max_history_entries:
            dropped = len(history) - max_history_entries
            history = history[dropped:]

        lines.append("History:")
        if dropped:
            lines.append(f"  (... {dropped} older entries omitted ...)")
        if history:
            lines.extend(f"  {entry}" for entry in history)
        else:
            lines.append("  (empty)")
        return "\n".join(lines)

    def summary(self) -> str:
        """One-line composite-task summary for logging: done / in progress / remaining."""
        done = sum(s.status is SubtaskStatus.SUCCEEDED for s in self.subtasks)
        failed = sum(s.status is SubtaskStatus.FAILED for s in self.subtasks)
        current = self.current_subtask()
        in_progress = current.description if current is not None else "none"
        remaining = sum(s.status is SubtaskStatus.PENDING for s in self.subtasks)
        return (
            f"{done}/{len(self.subtasks)} done ({failed} failed) | "
            f"in progress: {in_progress} | {remaining} remaining"
        )

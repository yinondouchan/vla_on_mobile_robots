"""Task planner: public API over decompose + assess.

`Planner` owns a shared :class:`~.vlm.OpenAIChat` and the episode
:class:`~.types.PlannerState`. Decomposition and assessment logic live in
:mod:`.decomposition` and :mod:`.assessment`; this class wires them into a
stateful per-episode controller for the VLA policy.

Typical usage::

    planner = Planner()
    planner.decompose("Stack the small cube on the medium cube, then ...", frames)
    ...
    action = policy.predict(task=planner.current_subtask(), ...)
    ...
    if frame_idx % monitor_every == 0:
        assessment = planner.step(frames)
"""

from __future__ import annotations

import numpy as np

from .assessment import assess_subtask
from .config import PlannerConfig
from .decomposition import decompose
from .types import PlannerState, SubtaskAssessment
from .vlm import OpenAIChat


class Planner:
    """Stateful planner: holds the chat backend + plan state, delegates decompose/assess."""

    def __init__(
        self,
        cfg: PlannerConfig | None = None,
        vlm: OpenAIChat | None = None,
        predefined_tasks: list[str] | None = None,
    ) -> None:
        self.cfg = cfg if cfg is not None else PlannerConfig()
        self.vlm = vlm if vlm is not None else OpenAIChat()
        self.state: PlannerState | None = None
        self._predefined_tasks = predefined_tasks

    def reset(self) -> None:
        """Clear all per-episode state (plan, history, vocabulary)."""
        self.state = None
        self._predefined_tasks = None

    def decompose(
        self,
        environment: str,
        robot_structure: str,
        task: str,
        images: dict[str, np.ndarray],
        predefined_tasks: list[str] | None = None,
        notes: str = "",
    ) -> None:
        """
        Decompose the given task using the VLM and images, and initialize the planner state.

        Parameters
        ----------
        task : str
            The composite task that needs to be decomposed into subtasks.
        images : dict[str, np.ndarray]
            Current camera observations, with keys as camera names and values as image arrays.
        predefined_tasks : list[str], optional
            Restrict decomposition to the provided vocabulary of tasks, if given.

        Side effects
        ------------
        Updates self.state with a fresh PlannerState and self._predefined_tasks.
        """
        self.state = decompose(
            self.vlm,
            environment,
            robot_structure,
            task,
            images,
            predefined_tasks=predefined_tasks,
            max_subtasks=self.cfg.max_subtasks,
            notes=notes,
        )
        self._predefined_tasks = predefined_tasks

    def assess(self, environment: str, robot_structure: str, images: dict[str, np.ndarray], step: int | None = None) -> SubtaskAssessment:
        """
        Assess the current subtask using the VLM and images, and update the planner state.
        """
        self.state = assess_subtask(
            self.vlm,
            self.state,
            environment,
            robot_structure,
            images,
            max_history_entries=self.cfg.max_history_entries,
        )

    def current_subtask(self) -> str:
        """Subtask the VLA policy should be conditioned on right now.

        Returns an empty string once every subtask has terminated (plan complete).
        """
        current = self._require_state().current_subtask()
        return current.description if current is not None else ""

    def step(
        self,
        environment: str,
        robot_structure: str,
        images: dict[str, np.ndarray],
        *,
        step: int | None = None,
    ) -> SubtaskAssessment:
        """Periodic monitor call: assess the current subtask from ``images``.

        Delegates to :func:`~.assessment.assess_subtask`, which mutates
        ``self.state`` in place (history, and decide actions when enabled).
        The caller chooses the cadence (e.g. every N frames).
        """
        state = self._require_state()
        return assess_subtask(
            self.vlm,
            state,
            environment,
            robot_structure,
            images,
            step=step,
            predefined_tasks=self._predefined_tasks,
            max_history_entries=self.cfg.max_history_entries,
        )

    def history_text(self) -> str:
        """Rendered plan + history for logging / debugging."""
        state = self._require_state()
        return state.render(self.cfg.max_history_entries)

    def _require_state(self) -> PlannerState:
        if self.state is None:
            raise RuntimeError("no plan yet; call decompose(...) first")
        return self.state

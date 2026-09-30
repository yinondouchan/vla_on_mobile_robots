"""Task planner: decompose + assess behind a stateful ``Planner``.

``Planner.decompose`` turns a composite task and current frames into an
ordered subtask plan. ``Planner.assess`` / ``Planner.step`` monitor progress
from current frames only (temporal context lives in
``PlannerState.history``). ``Planner`` owns a shared
:class:`~.vlm.HuggingFaceChat` and the episode :class:`~.types.PlannerState`.

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

import difflib
import logging

import numpy as np

from .config import PlannerConfig
from .prompts import build_decompose_prompt
from .types import PlannerState, Subtask, SubtaskAssessment, SubtaskStatus
from .vlm import HuggingFaceChat
from .prompts import _ASSESS_USER_TEMPLATE, ASSESS_SYSTEM

logger = logging.getLogger(__name__)

# Minimum difflib similarity ratio for snapping a generated subtask to an
# entry of ``predefined_tasks``; below this the subtask is dropped.
_SNAP_THRESHOLD = 0.6

_VALID_DECIDE_ACTIONS = ("continue", "advance", "retry", "replan")


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    return " ".join(text.lower().split()).rstrip(".")


def _snap_to_vocabulary(subtask: str, vocabulary: list[str]) -> str | None:
    """Map a generated subtask onto the closest predefined task, or ``None``.

    Exact (case/whitespace-insensitive) matches are taken directly; otherwise
    the closest vocabulary entry by ``difflib`` ratio is used if it clears
    ``_SNAP_THRESHOLD``.
    """
    normalized = _normalize(subtask)
    by_normalized = {_normalize(t): t for t in vocabulary}
    if normalized in by_normalized:
        return by_normalized[normalized]

    matches = difflib.get_close_matches(
        normalized, list(by_normalized), n=1, cutoff=_SNAP_THRESHOLD
    )
    if matches:
        return by_normalized[matches[0]]
    return None


def parse_subtasks(
    reply: dict,
    predefined_tasks: list[str] | None,
    max_subtasks: int,
) -> list[str]:
    """Validate the Decompose JSON and return clean subtask strings.

    Drops non-string / empty entries and consecutive duplicates; when a
    vocabulary is given, snaps each subtask to it and drops anything that
    cannot be snapped. The result is capped at ``max_subtasks``.
    """
    raw = reply.get("subtasks")
    if not isinstance(raw, list):
        logger.warning("decompose reply has no 'subtasks' list: %r", reply)
        return []

    subtasks: list[str] = []
    for entry in raw:
        if not isinstance(entry, str) or not entry.strip():
            logger.warning("dropping non-string/empty subtask entry: %r", entry)
            continue
        subtask = entry.strip()
        if predefined_tasks:
            snapped = _snap_to_vocabulary(subtask, predefined_tasks)
            if snapped is None:
                logger.warning(
                    "dropping subtask %r: no close match in predefined tasks",
                    subtask,
                )
                continue
            subtask = snapped
        if subtasks and subtasks[-1] == subtask:
            continue  # collapse consecutive duplicates
        subtasks.append(subtask)
        if len(subtasks) == max_subtasks:
            break
    return subtasks


# ---------------------------------------------------------------------------
# Assessment helpers
# ---------------------------------------------------------------------------


def _image_list(images: dict[str, np.ndarray]) -> list[np.ndarray]:
    # Sort by camera name for deterministic ordering across calls.
    return [images[name] for name in sorted(images)]


def _build_messages(
    images: list[np.ndarray],
    system: str,
    user: str,
) -> list[dict]:
    """Build a chat-template message list with optional system + image user turn."""
    messages: list[dict] = []
    if system:
        messages.append(
            {"role": "system", "content": [{"type": "text", "text": system}]}
        )
    user_content: list[dict] = [
        {"type": "image", "image": img} for img in images
    ]
    user_content.append({"type": "text", "text": user})
    messages.append({"role": "user", "content": user_content})
    return messages


def _parse_assessment(reply: dict) -> SubtaskAssessment:
    try:
        progress = float(reply)
    except (TypeError, ValueError):
        logger.warning("failed to parse assessment progress: %r", reply)
        progress = 0.0
        
    progress = min(max(progress, 0.0), 1.0)
    status = SubtaskStatus.IN_PROGRESS if progress < 1.0 else SubtaskStatus.SUCCEEDED
    rationale = ""

    return SubtaskAssessment(progress=progress, status=status, rationale=rationale)


# def _decide(
#     vlm: HuggingFaceChat,
#     state: PlannerState,
#     current: Subtask,
#     assessment: SubtaskAssessment,
#     images: dict[str, np.ndarray],
#     *,
#     step_idx: int,
#     predefined_tasks: list[str] | None,
# ) -> None:
#     """Run Decide after a terminal assessment and apply the chosen action."""
#     system, user = build_decide_prompt(
#         state.render(max_history_entries),
#         assessment.progress,
#         assessment.status.value,
#         assessment.rationale,
#     )
#     default_action = (
#         "advance" if assessment.status is SubtaskStatus.SUCCEEDED else "retry"
#     )
#     reply = vlm.chat_json(
#         _build_messages(_image_list(images), system, user),
#         fallback={"action": default_action, "subtask": ""},
#     )

#     action = str(reply.get("action", "")).strip().lower()
#     if action not in _VALID_DECIDE_ACTIONS:
#         logger.warning(
#             "invalid decide action %r; defaulting to %r", action, default_action
#         )
#         action = default_action
#     replacement = str(reply.get("subtask", "")).strip()

#     if action == "continue":
#         state.add_history(
#             f"[step {step_idx}] decision: continue {current.description!r}"
#         )
#     elif action == "advance":
#         current.status = SubtaskStatus.SUCCEEDED
#         state.add_history(
#             f"[step {step_idx}] decision: subtask "
#             f"{current.description!r} succeeded, advancing"
#         )
#         next_index = state.current_index + 1
#         if next_index < len(state.subtasks):
#             _start_subtask(state, next_index, step_idx)
#         else:
#             state.current_index = next_index
#             state.add_history(f"[step {step_idx}] plan complete")
#     elif action == "retry":
#         current.status = SubtaskStatus.IN_PROGRESS
#         current.attempts += 1
#         state.add_history(
#             f"[step {step_idx}] decision: retrying "
#             f"{current.description!r} (attempt {current.attempts})"
#         )
#     elif action == "replan":
#         snapped = _snap_replan(replacement, predefined_tasks)
#         if snapped is None:
#             current.status = SubtaskStatus.IN_PROGRESS
#             current.attempts += 1
#             state.add_history(
#                 f"[step {step_idx}] decision: replan requested but no valid "
#                 f"replacement; retrying {current.description!r} "
#                 f"(attempt {current.attempts})"
#             )
#         else:
#             current.status = SubtaskStatus.FAILED
#             state.add_history(
#                 f"[step {step_idx}] decision: replacing failed subtask "
#                 f"{current.description!r} with {snapped!r}"
#             )
#             state.subtasks.insert(
#                 state.current_index + 1, Subtask(description=snapped)
#             )
#             _start_subtask(state, state.current_index + 1, step_idx)


# def _start_subtask(state: PlannerState, index: int, step_idx: int) -> None:
#     if not (0 <= index < len(state.subtasks)):
#         return
#     state.current_index = index
#     subtask = state.subtasks[index]
#     subtask.status = SubtaskStatus.IN_PROGRESS
#     subtask.attempts += 1
#     state.add_history(
#         f"[step {step_idx}] starting subtask {index + 1}: "
#         f"{subtask.description!r} (attempt {subtask.attempts})"
#     )


# def _snap_replan(
#     replacement: str, predefined_tasks: list[str] | None
# ) -> str | None:
#     """Accept a replan replacement, optionally constraining to a vocabulary."""
#     if not replacement:
#         return None
#     if not predefined_tasks:
#         return replacement
#     if replacement in predefined_tasks:
#         return replacement
#     lowered = replacement.casefold()
#     for entry in predefined_tasks:
#         if entry.casefold() == lowered:
#             return entry
#     # Soft fuzzy match; reuse the same cutoff as decomposition.
#     matches = difflib.get_close_matches(
#         replacement, predefined_tasks, n=1, cutoff=0.6
#     )
#     return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class Planner:
    """Stateful planner: holds the chat backend + plan state, delegates decompose/assess."""

    def __init__(
        self,
        cfg: PlannerConfig,
        vlm: HuggingFaceChat | None = None,
        predefined_tasks: list[str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.vlm = vlm if vlm is not None else HuggingFaceChat(**cfg.vlm_kwargs)
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
        notes: str = ""
    ) -> None:
        """Run one Decompose call and store a fresh plan on ``self.state``.

        Parameters
        ----------
        task : str
            The composite task (e.g. "Stack the small cube on the medium
            cube, then move the stack to the blue platform").
        images : dict[str, np.ndarray]
            Current camera frames, same shape as the ``frames`` dict in
            ``sim.py``. Used only for grounding this single call.
        predefined_tasks : list[str], optional
            When given, the plan is composed only from this vocabulary: the
            prompt constrains the model to it and each parsed subtask is
            snapped to its closest entry (fuzzy match); subtasks that cannot
            be snapped are dropped.

        Side effects
        ------------
        Sets ``self.state`` with a "plan created" history entry, the first
        subtask ``IN_PROGRESS``, and the rest ``PENDING``. If the model
        produces no usable subtasks, the plan degrades to the composite
        task itself as a single subtask. Also stores ``predefined_tasks``.
        """
        max_subtasks = self.cfg.max_subtasks
        system, user = build_decompose_prompt(
            environment,
            robot_structure,
            task,
            max_subtasks,
            predefined_tasks=predefined_tasks,
            notes=notes,
        )
        messages = _build_messages(list(images.values()), system, user)
        reply = self.vlm.chat_json(messages, fallback={"subtasks": [task]})
        subtasks = parse_subtasks(reply, predefined_tasks, max_subtasks)

        if not subtasks:
            logger.warning(
                "decomposition produced no usable subtasks for %r; "
                "falling back to the composite task as a single subtask",
                task,
            )
            subtasks = [task]

        state = PlannerState(
            task=task,
            subtasks=[Subtask(description=s) for s in subtasks],
        )
        state.subtasks[0].status = SubtaskStatus.IN_PROGRESS
        state.subtasks[0].attempts = 1
        state.add_history(
            f"plan created with {len(state.subtasks)} subtask(s): "
            + "; ".join(f"{i + 1}. {s}" for i, s in enumerate(subtasks))
        )
        self.state = state
        self._predefined_tasks = predefined_tasks

    def assess(
        self,
        environment: str,
        robot_structure: str,
        images: dict[str, np.ndarray],
        custom_system_prompt: str | None = None,
        custom_user_prompt: str | None = None,
    ) -> SubtaskAssessment:
        """Assess the current subtask from observations and update ``self.state``.

        Parameters
        ----------
        images : dict of camera name -> (H, W, 3) uint8 array
            Current camera frames only (same shape as the ``frames`` dict in
            ``sim.py``). Past frames are never stored or re-processed.
        step : int, optional
            Label for history lines (e.g. frame index). Defaults to the number
            of history entries already recorded.

        Returns
        -------
        SubtaskAssessment
            The Assess result for this call. When the plan is already complete
            (no current subtask), returns a synthetic succeeded assessment.
        """
        state = self._require_state()
        step_idx = self.state.current_index

        current = state.current_subtask()
        if current is None:
            return SubtaskAssessment(
                progress=1.0,
                status=SubtaskStatus.SUCCEEDED,
                rationale="All subtasks are already complete.",
            )

        system = ASSESS_SYSTEM.format(environment=environment, robot_structure=robot_structure) if custom_system_prompt is None else custom_system_prompt
        user = _ASSESS_USER_TEMPLATE.format(subtask=current.description) if custom_user_prompt is None else custom_user_prompt

        messages = _build_messages(_image_list(images), system, user)
        reply = self.vlm.chat(messages)
        assessment = _parse_assessment(reply)
        state.add_history(
            f"[step {step_idx}] {current.description!r}: "
            f"progress {assessment.progress:.2f}, {assessment.status.value}"
            f" — {assessment.rationale}"
        )

        # if assessment.status in (SubtaskStatus.SUCCEEDED, SubtaskStatus.FAILED):
        #     _decide(
        #         self.vlm,
        #         state,
        #         current,
        #         assessment,
        #         images,
        #         step_idx=step_idx,
        #         predefined_tasks=self._predefined_tasks,
        #     )

        return assessment,

    def current_subtask(self) -> str:
        """Subtask the VLA policy should be conditioned on right now.

        Returns an empty string once every subtask has terminated (plan complete).
        """
        current = self._require_state().current_subtask()
        return current.description if current is not None else ""

    def history_text(self) -> str:
        """Rendered plan + history for logging / debugging."""
        state = self._require_state()
        return state.render(self.cfg.max_history_entries)

    def _require_state(self) -> PlannerState:
        if self.state is None:
            raise RuntimeError("no plan yet; call decompose(...) first")
        return self.state

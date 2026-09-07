"""Subtask monitoring: current frames + PlannerState -> updated plan status.

``update`` runs one Assess call on the current camera frames, appends a
text-only history entry, and — when the assessment is terminal (succeeded /
failed) — runs Decide to advance, retry, or replan. Only the current frames
are ever sent to the VLM; everything temporal lives in
``PlannerState.history``.
"""

from __future__ import annotations

import difflib
import logging

import numpy as np

from .prompts import build_assess_prompt, build_decide_prompt
from .types import PlannerState, Subtask, SubtaskAssessment, SubtaskStatus
from .vlm import SmolVLMChat

logger = logging.getLogger(__name__)

_VALID_DECIDE_ACTIONS = ("continue", "advance", "retry", "replan")


def assess_subtask(
    vlm: SmolVLMChat,
    state: PlannerState,
    images: dict[str, np.ndarray],
    *,
    step: int | None = None,
    predefined_tasks: list[str] | None = None,
) -> SubtaskAssessment:
    """Assess the current subtask from observations and update ``state``.

    Parameters
    ----------
    vlm : SmolVLMChat
        Shared chat wrapper (same instance used by ``decompose``).
    state : PlannerState
        Episode plan + text history; mutated in place.
    images : dict of camera name -> (H, W, 3) uint8 array
        Current camera frames only (same shape as the ``frames`` dict in
        ``sim.py``). Past frames are never stored or re-processed.
    step : int, optional
        Label for history lines (e.g. frame index). Defaults to the number
        of history entries already recorded.
    predefined_tasks : list of str, optional
        When given, a Decide ``replan`` replacement is snapped to this
        vocabulary; unmatchable replacements fall back to retrying the
        current subtask.

    Returns
    -------
    SubtaskAssessment
        The Assess result for this call. When the plan is already complete
        (no current subtask), returns a synthetic succeeded assessment.
    """
    step_idx = len(state.history) if step is None else step

    current = state.current_subtask()
    if current is None:
        return SubtaskAssessment(
            progress=1.0,
            status=SubtaskStatus.SUCCEEDED,
            rationale="All subtasks are already complete.",
        )

    assessment = _assess(vlm, state, current, images)
    state.add_history(
        f"[step {step_idx}] {current.description!r}: "
        f"progress {assessment.progress:.2f}, {assessment.status.value}"
        f" — {assessment.rationale}"
    )

    # if assessment.status in (SubtaskStatus.SUCCEEDED, SubtaskStatus.FAILED):
    #     _decide(
    #         vlm,
    #         state,
    #         current,
    #         assessment,
    #         images,
    #         step_idx=step_idx,
    #         predefined_tasks=predefined_tasks,
    #     )

    return assessment


def _image_list(images: dict[str, np.ndarray]) -> list[np.ndarray]:
    # Sort by camera name for deterministic ordering across calls.
    return [images[name] for name in sorted(images)]


def _assess(
    vlm: SmolVLMChat,
    state: PlannerState,
    current: Subtask,
    images: dict[str, np.ndarray],
) -> SubtaskAssessment:
    system, user = build_assess_prompt(
        current.description,
        state.render(vlm.cfg.max_history_entries),
    )
    fallback = {
        "progress": 0.0,
        "status": SubtaskStatus.IN_PROGRESS.value,
        "rationale": "Assessment unavailable; continuing current subtask.",
    }
    reply = vlm.chat_json(_image_list(images), system, user, fallback)
    return _parse_assessment(reply)


def _parse_assessment(reply: dict) -> SubtaskAssessment:
    try:
        progress = float(reply.get("progress", 0.0))
    except (TypeError, ValueError):
        progress = 0.0
    progress = min(max(progress, 0.0), 1.0)

    try:
        status = SubtaskStatus(str(reply.get("status", "")).strip().lower())
    except ValueError:
        status = SubtaskStatus.IN_PROGRESS
    if status is SubtaskStatus.PENDING:
        status = SubtaskStatus.IN_PROGRESS

    rationale = str(reply.get("rationale", "")).strip() or "(no rationale)"
    return SubtaskAssessment(progress=progress, status=status, rationale=rationale)

# def _decide(
#     vlm: SmolVLMChat,
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
#         state.render(vlm.cfg.max_history_entries),
#         assessment.progress,
#         assessment.status.value,
#         assessment.rationale,
#     )
#     default_action = (
#         "advance" if assessment.status is SubtaskStatus.SUCCEEDED else "retry"
#     )
#     reply = vlm.chat_json(
#         _image_list(images),
#         system,
#         user,
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


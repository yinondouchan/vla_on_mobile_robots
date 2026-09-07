"""Task decomposition: composite task + current frames -> ordered subtask plan.

``decompose`` is stateless: each call runs one Decompose prompt on the current
camera frames and returns a fresh :class:`~.types.PlannerState`. It takes a
shared :class:`~.vlm.SmolVLMChat` instance so the quantized model is loaded
once and reused by the monitor.
"""

from __future__ import annotations

import difflib
import logging

import numpy as np

from .prompts import build_decompose_prompt
from .types import PlannerState, Subtask, SubtaskStatus
from .vlm import SmolVLMChat

logger = logging.getLogger(__name__)

# Minimum difflib similarity ratio for snapping a generated subtask to an
# entry of ``predefined_tasks``; below this the subtask is dropped.
_SNAP_THRESHOLD = 0.6


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


def decompose(
    vlm: SmolVLMChat,
    task: str,
    images: dict[str, np.ndarray],
    predefined_tasks: list[str] | None = None,
) -> PlannerState:
    """Run one Decompose call and return a fresh plan.

    Parameters
    ----------
    vlm : SmolVLMChat
        Shared chat wrapper (the same instance used by the monitor, so the
        quantized model is loaded once).
    task : str
        The composite task (e.g. "Stack the small cube on the medium
        cube, then move the stack to the blue platform").
    images : dict of camera name -> (H, W, 3) uint8 array
        Current camera frames, same shape as the ``frames`` dict produced
        in ``sim.py``. Used only for grounding this single call.
    predefined_tasks : list of str, optional
        When given, the plan is composed only from this vocabulary: the
        prompt constrains the model to it and each parsed subtask is
        snapped to its closest entry (fuzzy match); subtasks that cannot
        be snapped are dropped.

    Returns
    -------
    PlannerState
        Fresh state with a "plan created" history entry, the first
        subtask ``IN_PROGRESS``, and the rest ``PENDING``. If the model
        produces no usable subtasks, the plan degrades to the composite
        task itself as a single subtask.
    """
    max_subtasks = vlm.cfg.max_subtasks
    system, user = build_decompose_prompt(
        task, max_subtasks, predefined_tasks=predefined_tasks
    )
    reply = vlm.chat_json(
        list(images.values()), system, user, fallback={"subtasks": [task]}
    )
    subtasks = _parse_subtasks(reply, predefined_tasks, max_subtasks)

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
    return state


def _parse_subtasks(
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

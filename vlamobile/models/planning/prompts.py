"""Prompt templates for the three planning VLM calls: decompose, assess, decide.

Each builder returns a ``(system, user)`` pair for ``SmolVLMChat.chat_json``.
All prompts demand a single strict JSON object as the reply (no prose, no code
fences) so the tolerant extractor in :mod:`.vlm` can parse it reliably. Images
(the current camera frames) are attached to the user turn by the caller; the
prompts reference them but never embed them.
"""

from __future__ import annotations

_JSON_ONLY_RULE = (
    "Reply with ONLY a single valid JSON object. "
    "No markdown, no code fences, no explanation outside the JSON."
)

# ---------------------------------------------------------------------------
# Decompose: composite task -> ordered subtask list
# ---------------------------------------------------------------------------

DECOMPOSE_SYSTEM = (
    "You are a task planner for a mobile robot with an arm and a gripper. "
    "You are shown the robot's current camera views. Break a composite task "
    "into a short ordered list of concrete subtasks, each independently "
    "executable by a low-level visuomotor policy (e.g. picking up an object, "
    "placing it somewhere, moving to a location). Subtasks must be phrased as "
    "short imperative commands and must be in execution order. "
    f"{_JSON_ONLY_RULE}"
)

_DECOMPOSE_USER_TEMPLATE = """The images show the robot's current camera views.

Composite task: {task}

Break this task into at most {max_subtasks} subtasks.{vocabulary_block}

Answer with JSON of this exact shape:
{{"subtasks": ["<subtask 1>", "<subtask 2>", ...]}}"""

_VOCABULARY_TEMPLATE = """

The low-level policy only understands the following commands. Every subtask
MUST be copied verbatim from this list (same wording, same casing):
{vocabulary}"""


def build_decompose_prompt(
    task: str,
    max_subtasks: int,
    predefined_tasks: list[str] | None = None,
) -> tuple[str, str]:
    """Prompt pair for the initial plan decomposition.

    When ``predefined_tasks`` is given, the model is constrained to compose
    the plan only from that vocabulary (the caller additionally validates and
    snaps the parsed output to the list).
    """
    vocabulary_block = ""
    if predefined_tasks:
        vocabulary = "\n".join(f"- {t}" for t in predefined_tasks)
        vocabulary_block = _VOCABULARY_TEMPLATE.format(vocabulary=vocabulary)
    user = _DECOMPOSE_USER_TEMPLATE.format(
        task=task,
        max_subtasks=max_subtasks,
        vocabulary_block=vocabulary_block,
    )
    return DECOMPOSE_SYSTEM, user


# ---------------------------------------------------------------------------
# Assess: current frames + history -> progress / status of current subtask
# ---------------------------------------------------------------------------

ASSESS_SYSTEM = (
    "You are a progress monitor for a mobile robot with an arm and a gripper. "
    "You are shown the robot's current camera views, the plan, and a text log "
    "of what happened so far. Judge only the CURRENT subtask from the visual "
    "evidence in the images; the history is context, not proof of success. Be "
    "conservative: report 'succeeded' only if the images clearly show the "
    "subtask is complete, and 'failed' only if it clearly cannot succeed "
    "without intervention (e.g. object dropped out of reach, robot stuck). "
    f"{_JSON_ONLY_RULE}"
)

_ASSESS_USER_TEMPLATE = """The images show the robot's current camera views.

{state_text}

Current subtask to assess: {subtask}

Answer with JSON of this exact shape:
{{"progress": <number between 0 and 1>, "status": "in_progress" | "succeeded" | "failed", "rationale": "<one short sentence>"}}"""


def build_assess_prompt(subtask: str, state_text: str) -> tuple[str, str]:
    """Prompt pair for assessing the current subtask.

    ``state_text`` is ``PlannerState.render(...)`` — the plan header plus the
    (truncated) text history. Only current frames are attached by the caller.
    """
    user = _ASSESS_USER_TEMPLATE.format(state_text=state_text, subtask=subtask)
    return ASSESS_SYSTEM, user


# ---------------------------------------------------------------------------
# Decide: latest assessment -> continue / advance / retry / replan
# ---------------------------------------------------------------------------

DECIDE_SYSTEM = (
    "You are the executive controller of a mobile robot's task plan. Given "
    "the plan, the execution history, and the latest assessment of the "
    "current subtask, decide what to do next. Choose exactly one action:\n"
    "- \"continue\": keep executing the current subtask (it is still in progress).\n"
    "- \"advance\": the current subtask succeeded; move to the next subtask.\n"
    "- \"retry\": the current subtask failed but is worth attempting again as is.\n"
    "- \"replan\": the current subtask keeps failing or the plan no longer fits "
    "the situation; propose a replacement subtask in the \"subtask\" field.\n"
    "For actions other than \"replan\", set \"subtask\" to \"\". "
    f"{_JSON_ONLY_RULE}"
)

_DECIDE_USER_TEMPLATE = """{state_text}

Latest assessment of the current subtask:
- progress: {progress:.2f}
- status: {status}
- rationale: {rationale}

Answer with JSON of this exact shape:
{{"action": "continue" | "advance" | "retry" | "replan", "subtask": "<replacement subtask, or empty string>"}}"""


def build_decide_prompt(
    state_text: str,
    progress: float,
    status: str,
    rationale: str,
) -> tuple[str, str]:
    """Prompt pair for deciding how to proceed after an assessment.

    ``state_text`` is ``PlannerState.render(...)``; ``progress``, ``status``,
    and ``rationale`` come from the latest :class:`~.types.SubtaskAssessment`.
    """
    user = _DECIDE_USER_TEMPLATE.format(
        state_text=state_text,
        progress=progress,
        status=status,
        rationale=rationale,
    )
    return DECIDE_SYSTEM, user
